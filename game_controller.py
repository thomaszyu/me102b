"""
Game controller: bridges the ESP32 display with the air hockey robot.

Waits for START GAME from the display, initializes motors + vision,
runs the air hockey player loop, and streams state to the display.
PAUSE holds mallet position. STOP kills motors and returns to menu.

you must kill 33071 in arduino ide if you accidentaly use the serial monitor and have a terminal open in vscode

Communicates with the ESP32 over USB UART.

Usage:
    python game_controller.py --serial /dev/cu.usbserial-XXXX
    
    testing: python display_code/laptop_listener.py --serial /dev/cu.usbserial-10
"""

from laptop_listener import SerialLink, SERIAL_BAUD
from air_hockey_player import play_air_hockey, _attack, DEFEND_X
from ekf_controller import EKFController
from home_motors import initialize_and_calibrate
from vision import VisionSystem
import asyncio
import sys
import os
import argparse
import numpy as np

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MOTOR_DIR = os.path.join(SCRIPT_DIR, 'motor_code', 'arbitrary_move')
VISION_DIR = os.path.join(SCRIPT_DIR, 'vision_code')
DISPLAY_DIR = os.path.join(SCRIPT_DIR, 'display_code')
sys.path.insert(0, MOTOR_DIR)
sys.path.insert(0, VISION_DIR)
sys.path.insert(0, DISPLAY_DIR)


def make_display_callback(link):
    """Create a tick callback that streams state to the ESP32 and handles commands."""
    score_us = 0
    score_them = 0

    def callback(state):
        puck = state["puck"]
        mallet_xy = state["mallet_xy"]
        mallet_valid = state["mallet_valid"]
        strategy = state["strategy"]
        paused = state["paused"]

        # Check for commands from display
        cmd = link.check_command()

        # Build trajectory for display
        traj = None
        tc = -1
        if _attack.phase is not None and _attack.traj_pos is not None:
            traj = [(float(p[0]), float(p[1])) for p in _attack.traj_pos]
            if _attack.contact_pos is not None:
                contact = np.array(_attack.contact_pos)
                dists = [np.linalg.norm(np.array(p) - contact)
                         for p in _attack.traj_pos]
                tc = int(np.argmin(dists))

        # Send state to display
        link.send_state(
            puck_x=float(puck.pos[0]), puck_y=float(puck.pos[1]),
            puck_vx=float(puck.vel[0]), puck_vy=float(puck.vel[1]),
            puck_valid=puck.initialized,
            mallet_x=float(mallet_xy[0]), mallet_y=float(mallet_xy[1]),
            mallet_valid=mallet_valid,
            strategy="PAUSED" if paused else strategy,
            score_us=score_us, score_them=score_them,
            trajectory=traj, contact_idx=tc,
        )

        return cmd  # None, "pause", or "stop"

    return callback


async def main(link):
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

            # Play with display callback
            callback = make_display_callback(link)
            result = await play_air_hockey(ctrl, duration=600.0, tick_callback=None)

            _attack.reset()
            if result == "stop":
                print("[game] Returned to menu. Waiting for next game...")

    except KeyboardInterrupt:
        print("\n[game] Ctrl+C — stopping motors...")
        if motors:
            ids = [1, 2, 3, 4]
            await asyncio.gather(*[motors[mid].set_stop() for mid in ids])
        if vis:
            vis.stop()
        print("[game] Done.")


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
