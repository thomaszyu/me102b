"""
Game controller wrapper.

Bridges the ESP32 display with the air hockey robot. Waits for START GAME from
the display, initializes motors + vision, runs the air hockey player loop, and
streams state to the display. PAUSE holds mallet position. STOP kills motors
and returns to menu. Communicates with the ESP32 over USB UART.

This file is intentionally as close to motor_code/arbitrary_move/air_hockey_player.py
main() as possible — same vision/motor/EKF init in the same order — with only
the ESP32 link wrapped around it.

Usage:
    python game_controller_new.py --serial /dev/cu.usbserial-XXXX
"""
import argparse
import asyncio
import os
import sys
import time

import numpy as np

# Make the rest of the project importable regardless of where we run from.
# IMPORTANT: this MUST happen before any `from air_hockey_player ...` /
# `from vision ...` / `from laptop_listener ...` imports, otherwise Python
# raises ModuleNotFoundError on those.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MOTOR_DIR = os.path.join(SCRIPT_DIR, "motor_code", "arbitrary_move")
VISION_DIR = os.path.join(SCRIPT_DIR, "vision_code")
DISPLAY_DIR = os.path.join(SCRIPT_DIR, "display_code")
sys.path.insert(0, MOTOR_DIR)
sys.path.insert(0, VISION_DIR)
sys.path.insert(0, DISPLAY_DIR)

from air_hockey_player import DEFEND_X, _attack, play_air_hockey  # noqa: E402
from ekf_controller import EKFController  # noqa: E402
from home_motors import initialize_and_calibrate  # noqa: E402
from laptop_listener import SERIAL_BAUD, SerialLink  # noqa: E402
from vision import VisionSystem  # noqa: E402


########################
## DISPLAY CALLBACK   ##
########################

# Match the display's white-rect bounds (display_code.ino TBL_*_MIN/MAX).
# These are physical table half-extents in mm. We clamp positions sent to
# the LCD into this rect so the puck/mallet always render inside the inner
# white rectangle, even when vision has a noisy detection just outside the
# table area or the EKF temporarily saturates against a goal wall.
#
# This only affects what the display sees. The motor control loop still
# operates on the raw EKF / vision values.
TABLE_X_MIN = -273.0
TABLE_X_MAX = 273.0
TABLE_Y_MIN = -240.0
TABLE_Y_MAX = 240.0
DISP_INSET = 5.0  # mm — keeps the dot just inside the white border


def _clamp_display_xy(x, y):
    cx = max(TABLE_X_MIN + DISP_INSET, min(TABLE_X_MAX - DISP_INSET, float(x)))
    cy = max(TABLE_Y_MIN + DISP_INSET, min(TABLE_Y_MAX - DISP_INSET, float(y)))
    return cx, cy


DIFFICULTY_MAX_SPEED = {
    "easy": 300.0,    # slow mallet
    "medium": 1200.0,  # current default behavior
    "hard": 4000.0,    # faster mallet
}


def speed_for_difficulty(difficulty: str) -> float:
    """Map display difficulty string to DEFEND/IDLE max mallet speed."""
    if difficulty is None:
        return DIFFICULTY_MAX_SPEED["medium"]
    return DIFFICULTY_MAX_SPEED.get(str(difficulty).lower(),
                                    DIFFICULTY_MAX_SPEED["medium"])


def make_display_callback(link, vis):
    """Build a per-tick callback that streams state to the ESP32 and relays
    PAUSE/STOP commands back to play_air_hockey().

    The callback is intentionally cheap:
      - link.send_state(...) is non-blocking (writer thread inside SerialLink)
      - link.check_command() is a non-blocking queue poll

    So this adds at most a few hundred microseconds per tick on top of the
    bare air_hockey_player loop.

    Args:
        link: SerialLink to the ESP32.
        vis:  VisionSystem instance — we read robot_score / player_score off
              of it, and on every increment fire a "goal" event so the LCD
              can pop a 3-second banner.
    """
    # We mirror the FSM-tracked scores out of the VisionSystem.
    state_box = {
        "prev_robot_score": int(getattr(vis, "robot_score", 0)),
        "prev_player_score": int(getattr(vis, "player_score", 0)),
        "last_diag_t": 0.0,
    }

    def callback(state):
        puck = state["puck"]
        mallet_xy = state["mallet_xy"]
        mallet_valid = state["mallet_valid"]
        strategy = state["strategy"]
        paused = state["paused"]

        # Pull any queued PAUSE/STOP/START from the display.
        cmd = link.check_command()

        # Live scores from the FSM in vision.
        score_us = int(getattr(vis, "robot_score", 0))
        score_them = int(getattr(vis, "player_score", 0))

        # Goal banner: fire on score increments.
        if score_us > state_box["prev_robot_score"]:
            link.send({
                "type": "goal", "by": "robot",
                "su": score_us, "st": score_them,
            })
            state_box["prev_robot_score"] = score_us
        if score_them > state_box["prev_player_score"]:
            link.send({
                "type": "goal", "by": "player",
                "su": score_us, "st": score_them,
            })
            state_box["prev_player_score"] = score_them

        # Optional: include current attack trajectory + contact index so the
        # display can render the planned strike line. Clamped into the
        # visible rect for the same reason puck/mallet are.
        traj = None
        tc = -1
        if _attack.phase is not None and _attack.traj_pos is not None:
            traj = [_clamp_display_xy(p[0], p[1]) for p in _attack.traj_pos]
            if _attack.contact_pos is not None:
                contact = np.array(_attack.contact_pos)
                dists = [np.linalg.norm(np.array(p) - contact)
                         for p in _attack.traj_pos]
                tc = int(np.argmin(dists))

        # Clamp the puck and mallet positions into the white-rect so a noisy
        # vision detection or a saturated EKF can't push them off-screen.
        puck_disp_x, puck_disp_y = _clamp_display_xy(puck.pos[0], puck.pos[1])
        mallet_disp_x, mallet_disp_y = _clamp_display_xy(
            mallet_xy[0], mallet_xy[1]
        )

        link.send_state(
            puck_x=puck_disp_x, puck_y=puck_disp_y,
            puck_vx=float(puck.vel[0]), puck_vy=float(puck.vel[1]),
            puck_valid=bool(puck.initialized),
            mallet_x=mallet_disp_x, mallet_y=mallet_disp_y,
            mallet_valid=mallet_valid,
            strategy="PAUSED" if paused else strategy,
            score_us=score_us, score_them=score_them,
            trajectory=traj, contact_idx=tc,
        )

        # ~1 Hz diagnostic so we can see why the mallet looks parked at the
        # goal or why the puck never appears: prints raw EKF mallet pos, raw
        # puck pos, and validity flags.
        now = time.monotonic()
        if now - state_box["last_diag_t"] > 1.0:
            state_box["last_diag_t"] = now
            print(
                f"[diag] mallet=({mallet_xy[0]:+7.1f},{mallet_xy[1]:+7.1f}) "
                f"valid={mallet_valid}  "
                f"puck=({puck.pos[0]:+7.1f},{puck.pos[1]:+7.1f}) "
                f"init={puck.initialized} vis={puck.visible}  "
                f"score=us:{score_us}/them:{score_them}"
            )

        # play_air_hockey understands None / "pause" / "stop". Anything else
        # ("start", "score", etc.) is ignored.
        return cmd

    return callback


########################
## MAIN               ##
########################

MOTOR_IDS = [1, 2, 3, 4]


async def shutdown_motors(motors):
    """Send set_stop() to every cable-robot motor. Tolerant of missing motors
    or transient failures so we always get as close to a clean shutdown as
    we can before the program exits."""
    if not motors:
        return
    print("[game] Stopping motors...")
    tasks = []
    for mid in MOTOR_IDS:
        m = motors.get(mid) if isinstance(motors, dict) else None
        if m is None:
            continue
        try:
            tasks.append(m.set_stop())
        except Exception as e:
            print(f"[game] motor {mid}: set_stop() raised {e!r}")
    if tasks:
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            print(f"[game] shutdown_motors gather: {e!r}")
    print("[game] Motors stopped.")


async def main(link):
    # 1) Wait for the user to tap a difficulty + START on the display.
    print("Waiting for START GAME from display...")
    difficulty = link.wait_for_start()
    print(f"Starting game (difficulty={difficulty})")
    max_speed = speed_for_difficulty(difficulty)
    print(f"Using max mallet speed: {max_speed:.0f} mm/s")

    # ---- From here on, this mirrors air_hockey_player.main() exactly. ----

    vis = None
    motors = None
    try:
        # Start vision
        vis = VisionSystem()
        vis.start(show_display=False)
        print("Waiting for camera...")
        await asyncio.sleep(1.0)

        # Home motors
        motors, _ = await initialize_and_calibrate()

        # Create controller + initialize EKF
        ctrl = EKFController(motors, vis)
        await ctrl.initialize_ekf()

        # Move to starting defense position
        await ctrl.move_to(np.array([DEFEND_X, 0.0]), duration=0.5)

        # Play! (same call as air_hockey_player.main(), just with a callback)
        callback = make_display_callback(link, vis)
        game_result = await play_air_hockey(
            ctrl,
            duration=120.0,
            tick_callback=callback,
            max_speed_normal=max_speed,
        )
        # Winner is decided strictly at the 120s buzzer.
        # If user hit STOP, skip winner announcement.
        if game_result != "stop":
            score_us = int(getattr(vis, "robot_score", 0))
            score_them = int(getattr(vis, "player_score", 0))
            if score_us > score_them:
                winner = "robot"
            elif score_them > score_us:
                winner = "player"
            else:
                winner = "draw"

            # Set winner in vision so OpenCV overlay can show it if enabled.
            vis.winner = winner
            link.send({
                "type": "win",
                "by": winner,
                "su": score_us,
                "st": score_them,
            })
            print(f"=== TIME UP (120s): winner={winner} final us:{score_us} / them:{score_them} ===")
    finally:
        # Always make sure the motors are de-energized when the game ends —
        # whether that's a win, a STOP press, the duration timer expiring,
        # or an exception.
        await shutdown_motors(motors)
        if vis is not None:
            vis.stop()


def _build_link():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serial", metavar="PORT", required=True,
        help="Serial port the ESP32 is connected to "
             "(e.g. /dev/cu.usbserial-0001, /dev/ttyUSB0, COM5).",
    )
    parser.add_argument(
        "--baud", type=int, default=SERIAL_BAUD,
        help=f"Serial baud rate (default {SERIAL_BAUD}).",
    )
    args = parser.parse_args()

    print(f"[game] Using UART link on {args.serial} @ {args.baud}")
    return SerialLink(args.serial, args.baud)


if __name__ == "__main__":
    link = _build_link()
    try:
        asyncio.run(main(link))
    finally:
        try:
            link.close()
        except Exception:
            pass
