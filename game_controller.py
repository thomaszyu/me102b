"""
Game controller: bridges the ESP32 display with the air hockey robot.

Waits for START GAME from the display, initializes motors + vision,
runs the air hockey player loop, and streams state to the display.
PAUSE holds mallet position. STOP kills motors and returns to menu.

Usage:
    python game_controller.py
"""

import asyncio
import sys
import os
import numpy as np
import time
import moteus

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MOTOR_DIR = os.path.join(SCRIPT_DIR, 'motor_code', 'arbitrary_move')
VISION_DIR = os.path.join(SCRIPT_DIR, 'vision_code')
DISPLAY_DIR = os.path.join(SCRIPT_DIR, 'display_code')
sys.path.insert(0, MOTOR_DIR)
sys.path.insert(0, VISION_DIR)
sys.path.insert(0, DISPLAY_DIR)

from laptop_listener import DisplayLink
from air_hockey_player import (
    PuckTracker, decide_strategy, clamp_to_workspace, _attack,
    DEFEND_X, MALLET_X_MIN, MALLET_X_MAX, MALLET_Y_MIN, MALLET_Y_MAX,
    TABLE_X_MIN, TABLE_X_MAX, TABLE_Y_MIN, TABLE_Y_MAX,
)
from config import *
from kinematics_utils import xy_to_enc, xy_vel_to_enc_vel
from motor_utils_for_arbitrary_move import read_encoders
from ekf_controller import EKFController
from home_motors import initialize_and_calibrate
from vision import VisionSystem


UDP_PORT = 5005


async def stop_all_motors(motors):
    """Kill all motors immediately."""
    ids = [1, 2, 3, 4]
    await asyncio.gather(*[motors[mid].set_stop() for mid in ids])
    print("[game] All motors stopped.")


async def hold_position(motors, current_enc):
    """Hold mallet at current encoder positions (zero velocity)."""
    ids = [1, 2, 3, 4]
    await asyncio.gather(*[
        motors[mid].set_position(
            position=current_enc[mid - 1],
            velocity=0.0,
            maximum_torque=MAX_TORQUE * abs(MOTOR_TORQUE_SCALE[mid]),
            watchdog_timeout=np.nan,
            query=True,
        ) for mid in ids
    ])


async def game_loop(ctrl, link):
    """
    Main game loop. Runs the air hockey player and streams state to the display.
    Handles PAUSE and STOP commands from the display.
    """
    puck = PuckTracker(alpha_pos=0.5, alpha_vel=0.3, max_jump=100.0)

    ids = [1, 2, 3, 4]
    slack_ff = np.zeros(4)
    smooth_offset = None
    OFFSET_ALPHA = 0.2
    MAX_MALLET_LOST = 10
    mallet_lost_count = 0

    desired_xy = ctrl.ekf.position.copy()
    commanded_xy = ctrl.ekf.position.copy()
    smooth_target_enc = None
    CMD_ALPHA = 0.3
    MAX_SPEED_NORMAL = 400.0
    loop_time = time.time()
    current_enc = await read_encoders(ctrl.motors)

    last_strategy = 'IDLE'
    paused = False
    score_us, score_them = 0, 0
    tick = 0

    print("[game] Game loop started. Defending X={:.0f}mm".format(DEFEND_X))

    while True:
        now = time.time()
        dt = max(now - loop_time, 0.001)
        loop_time = now

        # --- Check for commands from display ---
        cmd = link.check_command()
        if cmd == "stop":
            print("[game] STOP received.")
            await stop_all_motors(ctrl.motors)
            return "stop"
        elif cmd == "pause":
            if not paused:
                paused = True
                print("[game] PAUSED — holding position.")
            else:
                paused = False
                print("[game] RESUMED.")
                _attack.reset()

        # --- If paused, hold position and keep streaming state ---
        if paused:
            await hold_position(ctrl.motors, current_enc)
            mallet_reading, puck_reading, _ = ctrl.vision.get_positions()
            puck.update(puck_reading, puck_reading[2])
            mallet_xy = ctrl.ekf.position

            link.send_state(
                puck_x=float(puck.pos[0]), puck_y=float(puck.pos[1]),
                puck_vx=float(puck.vel[0]), puck_vy=float(puck.vel[1]),
                puck_valid=puck.initialized,
                mallet_x=float(mallet_xy[0]), mallet_y=float(mallet_xy[1]),
                mallet_valid=(mallet_lost_count < MAX_MALLET_LOST),
                strategy="PAUSED",
                score_us=score_us, score_them=score_them,
            )
            await asyncio.sleep(TICK_RATE)
            continue

        # --- EKF predict + update ---
        ctrl.ekf.predict(dt)
        mallet_reading, puck_reading, _ = ctrl.vision.get_positions()

        if mallet_reading[2]:
            ctrl.ekf.update([mallet_reading[0], mallet_reading[1]])
            mallet_lost_count = 0
        else:
            mallet_lost_count += 1

        # --- Update puck tracker ---
        puck.update(puck_reading, puck_reading[2])

        # --- Safety freeze ---
        if mallet_lost_count >= MAX_MALLET_LOST:
            states = await asyncio.gather(*[
                ctrl.motors[mid].set_position(
                    position=current_enc[mid - 1], velocity=0.0,
                    maximum_torque=MAX_TORQUE * abs(MOTOR_TORQUE_SCALE[mid]),
                    watchdog_timeout=np.nan, query=True,
                ) for mid in ids
            ])
            await asyncio.sleep(dt)
            current_enc = np.array([
                states[i].values[moteus.Register.POSITION] for i in range(4)
            ])

            link.send_state(
                puck_x=float(puck.pos[0]), puck_y=float(puck.pos[1]),
                puck_vx=float(puck.vel[0]), puck_vy=float(puck.vel[1]),
                puck_valid=puck.initialized,
                mallet_x=float(ctrl.ekf.position[0]),
                mallet_y=float(ctrl.ekf.position[1]),
                mallet_valid=False,
                strategy="LOST",
                score_us=score_us, score_them=score_them,
            )
            continue

        # --- Decide strategy ---
        mallet_xy = ctrl.ekf.position
        strategy, target_data, target_vel = decide_strategy(puck, mallet_xy)

        # --- Compute commanded position + velocity ---
        prev_commanded = commanded_xy.copy()
        commanded_vel = np.zeros(2)

        if strategy == 'STRIKE' and target_data is not None:
            commanded_xy = clamp_to_workspace(target_data)
            if target_vel is not None:
                commanded_vel = target_vel
        else:
            if strategy == 'DEFEND' and target_data is not None:
                raw_desired = clamp_to_workspace(np.array([DEFEND_X, target_data]))
            else:
                raw_desired = clamp_to_workspace(np.array([DEFEND_X, 0.0]))

            desired_xy = (1 - 0.15) * desired_xy + 0.15 * raw_desired
            max_step = MAX_SPEED_NORMAL * dt
            direction = desired_xy - commanded_xy
            dist = np.linalg.norm(direction)
            if dist > max_step:
                commanded_xy = commanded_xy + direction * (max_step / dist)
            else:
                commanded_xy = desired_xy.copy()
            commanded_vel = (commanded_xy - prev_commanded) / dt

        # --- Compute motor target ---
        camera_xy = ctrl.ekf.position
        expected_enc = xy_to_enc(camera_xy)
        raw_offset = current_enc - expected_enc
        if smooth_offset is None:
            smooth_offset = raw_offset.copy()
        else:
            smooth_offset = (1 - OFFSET_ALPHA) * smooth_offset + OFFSET_ALPHA * raw_offset

        enc_delta = xy_to_enc(commanded_xy) - expected_enc
        target_enc = current_enc + MOVE_GAIN * enc_delta - SIGNS * TENSION_BIAS_REV

        if strategy == 'STRIKE':
            smooth_target_enc = target_enc.copy()
        elif smooth_target_enc is None:
            smooth_target_enc = target_enc.copy()
        else:
            smooth_target_enc = (1 - CMD_ALPHA) * smooth_target_enc + CMD_ALPHA * target_enc
            target_enc = smooth_target_enc

        MAX_ENC_STEP = 0.2
        enc_step = target_enc - current_enc
        clamped = np.clip(enc_step, -MAX_ENC_STEP, MAX_ENC_STEP)
        target_enc = current_enc + clamped

        enc_vel = xy_vel_to_enc_vel(commanded_xy, commanded_vel)

        # --- Send to motors ---
        states = await asyncio.gather(*[
            ctrl.motors[mid].set_position(
                position=target_enc[mid - 1],
                velocity=enc_vel[mid - 1],
                kp_scale=KP_SCALE,
                kd_scale=KD_SCALE,
                velocity_limit=VEL_LIMIT,
                feedforward_torque=slack_ff[mid - 1],
                maximum_torque=MAX_TORQUE * abs(MOTOR_TORQUE_SCALE[mid]),
                watchdog_timeout=np.nan,
                query=True,
            )
            for mid in ids
        ])
        await asyncio.sleep(TICK_RATE)

        current_enc = np.array([
            states[i].values[moteus.Register.POSITION] for i in range(4)
        ])

        # Slack detection
        for mid in ids:
            i = mid - 1
            torque = states[i].values[moteus.Register.TORQUE]
            if abs(torque) < 0.1:
                slack_ff[i] = TENSION_TORQUE * MOTOR_TORQUE_SCALE[mid]
            else:
                slack_ff[i] = 0.0

        # --- Build trajectory for display ---
        traj_for_display = None
        traj_contact = -1
        if _attack.phase is not None and _attack.traj_pos is not None:
            traj_for_display = [(float(p[0]), float(p[1])) for p in _attack.traj_pos]
            if _attack.contact_pos is not None:
                # Find closest trajectory point to contact
                contact = np.array(_attack.contact_pos)
                dists = [np.linalg.norm(np.array(p) - contact) for p in _attack.traj_pos]
                traj_contact = int(np.argmin(dists))

        # --- Send state to display ---
        link.send_state(
            puck_x=float(puck.pos[0]), puck_y=float(puck.pos[1]),
            puck_vx=float(puck.vel[0]), puck_vy=float(puck.vel[1]),
            puck_valid=puck.initialized,
            mallet_x=float(mallet_xy[0]), mallet_y=float(mallet_xy[1]),
            mallet_valid=True,
            strategy=strategy,
            score_us=score_us, score_them=score_them,
            trajectory=traj_for_display, contact_idx=traj_contact,
        )

        if strategy != last_strategy or tick % 40 == 0:
            print(f"  [{time.time():.1f}] {strategy}  "
                  f"cmd=({commanded_xy[0]:.0f},{commanded_xy[1]:.0f})  "
                  f"mallet=({mallet_xy[0]:.0f},{mallet_xy[1]:.0f})  "
                  f"puck=({puck.pos[0]:.0f},{puck.pos[1]:.0f})")
            last_strategy = strategy

        tick += 1


async def main():
    link = DisplayLink(UDP_PORT)

    vis = None
    motors = None
    ctrl = None

    try:
        while True:
            print("\n[game] Waiting for START GAME from display...")
            difficulty = link.wait_for_start()
            print(f"[game] Starting game (difficulty={difficulty})")

            # Initialize hardware if not already done
            if vis is None:
                print("[game] Starting vision...")
                vis = VisionSystem()
                vis.start(show_display=False)
                await asyncio.sleep(1.0)

            if motors is None:
                print("[game] Homing motors...")
                motors, _ = await initialize_and_calibrate()

            if ctrl is None:
                ctrl = EKFController(motors, vis)
                await ctrl.initialize_ekf()

            # Move to starting position
            print("[game] Moving to defend position...")
            await ctrl.move_to(np.array([DEFEND_X, 0.0]), duration=0.5)

            # Run game until STOP
            result = await game_loop(ctrl, link)

            if result == "stop":
                # Motors already stopped, reset attack state
                _attack.reset()
                print("[game] Returned to menu. Waiting for next game...")
                # Don't tear down hardware — reuse for next game

    except KeyboardInterrupt:
        print("\n[game] Ctrl+C — stopping motors...")
        if motors:
            await stop_all_motors(motors)
        if vis:
            vis.stop()
        print("[game] Done.")


if __name__ == "__main__":
    asyncio.run(main())
