"""
Puck Y-tracking mode.

The mallet follows the puck's Y position while staying on its current X line.
Uses the EKF controller's camera-based closed-loop control.

Usage:
    python puck_tracker.py
"""

import asyncio
import sys
import os
import numpy as np
import moteus

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'vision_code'))

from config import *
from kinematics_utils import xy_to_enc
from motor_utils_for_arbitrary_move import read_encoders
from ekf_controller import EKFController


async def track_puck_y(ctrl, x_line=None, duration=30.0):
    """
    Continuously move the mallet to match the puck's Y position.

    ctrl: initialized EKFController
    x_line: fixed X coordinate for the mallet (mm). None = use current X.
    duration: how long to track (seconds)
    """
    if not ctrl._initialized:
        await ctrl.initialize_ekf()

    # Use current mallet X if not specified
    if x_line is None:
        mallet, _, _ = ctrl.vision.get_positions()
        if mallet[2]:
            x_line = mallet[0]
        else:
            x_line = ctrl.ekf.position[0]
        print(f"Tracking on X line: {x_line:.1f} mm")

    ids = [1, 2, 3, 4]
    dt = TICK_RATE
    num_ticks = int(duration / dt)
    slack_ff = np.zeros(4)
    smooth_offset = None
    OFFSET_ALPHA = 0.2

    # Seed encoders
    current_enc = await read_encoders(ctrl.motors)

    PUCK_ALPHA = 0.3          # puck Y smoothing (0 = frozen, 1 = no filter)
    MAX_PUCK_JUMP_MM = 80.0   # reject single-frame jumps larger than this
    MAX_MALLET_LOST = 10      # freeze motors after this many consecutive mallet-lost frames

    print(f"Tracking puck Y for {duration:.0f}s on X={x_line:.0f}mm. Press Ctrl+C to stop.")

    filtered_puck_y = 0.0
    puck_initialized = False
    mallet_lost_count = 0

    for _ in range(num_ticks):
        # --- EKF predict + update (mallet) ---
        ctrl.ekf.predict(dt)
        mallet, puck, _ = ctrl.vision.get_positions()
        if mallet[2]:
            ctrl.ekf.update([mallet[0], mallet[1]])
            mallet_lost_count = 0
        else:
            mallet_lost_count += 1

        # --- Safety freeze: hold position if mallet lost too long ---
        if mallet_lost_count >= MAX_MALLET_LOST:
            # Just hold current position, don't chase stale EKF estimate
            states = await asyncio.gather(*[
                ctrl.motors[mid].set_position(
                    position         = current_enc[mid - 1],
                    velocity         = 0.0,
                    maximum_torque   = MAX_TORQUE * abs(MOTOR_TORQUE_SCALE[mid]),
                    watchdog_timeout = np.nan,
                    query            = True,
                )
                for mid in ids
            ])
            await asyncio.sleep(dt)
            current_enc = np.array([
                states[i].values[moteus.Register.POSITION] for i in range(4)
            ])
            continue

        # --- Get puck Y target (filtered + jump rejection) ---
        if puck[2]:  # puck visible
            raw_y = puck[1]
            if not puck_initialized:
                filtered_puck_y = raw_y
                puck_initialized = True
            elif abs(raw_y - filtered_puck_y) < MAX_PUCK_JUMP_MM:
                filtered_puck_y = (1 - PUCK_ALPHA) * filtered_puck_y + PUCK_ALPHA * raw_y

        last_puck_y = filtered_puck_y

        target_xy = np.array([x_line, last_puck_y])

        # --- Smoothed offset ---
        camera_xy = ctrl.ekf.position
        expected_enc = xy_to_enc(camera_xy)
        raw_offset = current_enc - expected_enc
        if smooth_offset is None:
            smooth_offset = raw_offset.copy()
        else:
            smooth_offset = (1 - OFFSET_ALPHA) * smooth_offset + OFFSET_ALPHA * raw_offset

        # --- Target encoder ---
        enc_delta = xy_to_enc(target_xy) - expected_enc
        target_enc = current_enc + MOVE_GAIN * enc_delta - SIGNS * TENSION_BIAS_REV

        # --- Send to motors ---
        states = await asyncio.gather(*[
            ctrl.motors[mid].set_position(
                position           = target_enc[mid - 1],
                velocity           = 0.0,
                feedforward_torque = slack_ff[mid - 1],
                maximum_torque     = MAX_TORQUE * abs(MOTOR_TORQUE_SCALE[mid]),
                watchdog_timeout   = np.nan,
                query              = True,
            )
            for mid in ids
        ])
        await asyncio.sleep(dt)

        # Read encoders from response
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

    print("Tracking complete.")


async def main():
    from home_motors import initialize_and_calibrate
    from vision import VisionSystem

    # Start vision
    vis = VisionSystem()
    vis.start(show_display=False)
    print("Waiting for camera...")
    await asyncio.sleep(1.0)

    # Home motors
    motors, _ = await initialize_and_calibrate()

    # Create controller
    ctrl = EKFController(motors, vis)
    await ctrl.initialize_ekf()

    # Track puck Y on the current X line
    await track_puck_y(ctrl, x_line=None, duration=60.0)

    vis.stop()


if __name__ == "__main__":
    asyncio.run(main())
