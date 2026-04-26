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

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MOTOR_DIR = os.path.join(SCRIPT_DIR, 'motor_code', 'arbitrary_move')
VISION_DIR = os.path.join(SCRIPT_DIR, 'vision_code')
DISPLAY_DIR = os.path.join(SCRIPT_DIR, 'display_code')
sys.path.insert(0, MOTOR_DIR)
sys.path.insert(0, VISION_DIR)
sys.path.insert(0, DISPLAY_DIR)

from laptop_listener import DisplayLink
from air_hockey_player import play_air_hockey, _attack, DEFEND_X
from ekf_controller import EKFController
from home_motors import initialize_and_calibrate
from vision import VisionSystem


UDP_PORT = 5005


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
                dists = [np.linalg.norm(np.array(p) - contact) for p in _attack.traj_pos]
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

            # Play with display callback
            callback = make_display_callback(link)
            result = await play_air_hockey(ctrl, duration=600.0, tick_callback=callback)

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


if __name__ == "__main__":
    asyncio.run(main())
