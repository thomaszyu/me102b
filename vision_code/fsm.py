"""
Air-hockey finite-state machine.

Changes vs. the original fsm.py:
  * Loads a per-environment homography (calibration_data.npz) and works in
    world millimetres (origin at centre dot). Pixel-based ROIs are gone.
  * Puck position is reported in mm; velocity is computed in mm/s and is
    what gets handed to the RL model.
  * Goal detection no longer requires the puck to be seen *inside* the goal
    box on the very last frame. We keep a short trajectory buffer and, when
    the puck is lost, extrapolate its last known velocity forward to decide
    whether it was heading into a goal. The 10-frame lost threshold is
    preserved so a puck that is merely "stuck" (still visible) never scores,
    and a brief occlusion can be recovered without a false point.

Controls: press 'q' to quit, 'c' to re-run calibration (exits with code 2).
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import deque
from typing import Optional, Tuple

import cv2 as cv
import numpy as np

import table_config as tc
import vision_utils as vu


# ---------------------------------------------------------------------------
# Configuration knobs
# ---------------------------------------------------------------------------

TRAJECTORY_BUFFER_FRAMES = 8       # number of recent (x,y,t) samples kept
LOST_FRAMES_THRESHOLD = 10         # same timeout you had -- don't shorten
EXTRAPOLATE_HORIZON_SEC = 0.25     # how far ahead (s) to project when lost
EXTRAPOLATE_STEPS = 30             # discretisation of the extrapolation ray
MIN_SPEED_FOR_EXTRAP_MM_S = 200.0  # below this we assume puck just stopped,
                                   # don't attribute a goal intent
MIN_VISIBLE_FRAMES = 3             # before entering TRACKING
POST_SCORE_FREEZE_FRAMES = 30      # cooldown after a goal is awarded

PUCK_TARGET_BGR = (62, 165, 84)    # your current green puck


# ---------------------------------------------------------------------------
# Puck colour mask helper (kept consistent with the old file)
# ---------------------------------------------------------------------------

def build_puck_hsv_windows(target_bgr) -> Tuple[np.ndarray, np.ndarray,
                                                Optional[np.ndarray],
                                                Optional[np.ndarray]]:
    target = np.uint8([[list(target_bgr)]])
    hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)[0][0]
    hue, sat, val = int(hsv[0]), int(hsv[1]), int(hsv[2])
    lower1 = np.array([max(0, hue - 10), max(50, sat - 60), max(50, val - 60)])
    upper1 = np.array([min(180, hue + 10), min(255, sat + 60), min(255, val + 60)])
    lower2, upper2 = None, None
    if hue < 10:
        lower2 = np.array([180 - (10 - hue), max(50, sat - 60), max(50, val - 60)])
        upper2 = np.array([180, min(255, sat + 60), min(255, val + 60)])
    elif hue > 170:
        lower2 = np.array([0, max(50, sat - 60), max(50, val - 60)])
        upper2 = np.array([10 - (180 - hue), min(255, sat + 60), min(255, val + 60)])
    return lower1, upper1, lower2, upper2


def detect_puck_pixel(mask: np.ndarray,
                      frame_for_draw: Optional[np.ndarray] = None
                      ) -> Optional[Tuple[float, float]]:
    """Return (u, v) in image pixels, or None if no puck found."""
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv.contourArea)
    if cv.contourArea(largest) < 50:
        return None
    M = cv.moments(largest)
    if M["m00"] <= 0:
        return None
    u = M["m10"] / M["m00"]
    v = M["m01"] / M["m00"]
    if frame_for_draw is not None:
        cv.circle(frame_for_draw, (int(u), int(v)), 30, (255, 0, 0), 5)
    return (u, v)


# ---------------------------------------------------------------------------
# Goal-intent check using recent trajectory
# ---------------------------------------------------------------------------

def classify_goal_intent(trajectory: deque) -> Optional[str]:
    """
    Look at the most recent samples (in mm, timestamped) and decide whether
    the puck -- at the moment we lost track of it -- was heading into either
    goal. Returns 'robot', 'player', or None.
    """
    if len(trajectory) < 2:
        return None

    # Average velocity over the last (up to 4) samples -> robust to noise.
    samples = list(trajectory)[-4:]
    x0, y0, t0 = samples[0]
    xN, yN, tN = samples[-1]
    dt = tN - t0
    if dt <= 1e-4:
        return None
    vx = (xN - x0) / dt
    vy = (yN - y0) / dt
    speed = math.hypot(vx, vy)
    if speed < MIN_SPEED_FOR_EXTRAP_MM_S:
        return None

    # Walk the ray forward in small steps, checking for a goal rect crossing.
    step_dt = EXTRAPOLATE_HORIZON_SEC / EXTRAPOLATE_STEPS
    px, py = xN, yN
    for _ in range(EXTRAPOLATE_STEPS):
        nx = px + vx * step_dt
        ny = py + vy * step_dt
        # Robot goal on +x rail.
        if vu.segment_hits_rect(px, py, nx, ny,
                                tc.ROBOT_GOAL_X_MIN, tc.ROBOT_GOAL_X_MAX,
                                tc.ROBOT_GOAL_Y_MIN, tc.ROBOT_GOAL_Y_MAX):
            return "robot"
        # Player goal on -x rail.
        if vu.segment_hits_rect(px, py, nx, ny,
                                tc.PLAYER_GOAL_X_MIN, tc.PLAYER_GOAL_X_MAX,
                                tc.PLAYER_GOAL_Y_MIN, tc.PLAYER_GOAL_Y_MAX):
            return "player"
        px, py = nx, ny
        # Stop if we've clearly left the table without hitting a goal.
        if abs(px) > tc.HALF_LENGTH_MM + 100 or abs(py) > tc.HALF_WIDTH_MM + 100:
            break
    return None


def is_in_goal(x_mm: float, y_mm: float) -> Optional[str]:
    if vu.point_in_rect(x_mm, y_mm,
                        tc.ROBOT_GOAL_X_MIN, tc.ROBOT_GOAL_X_MAX,
                        tc.ROBOT_GOAL_Y_MIN, tc.ROBOT_GOAL_Y_MAX):
        return "robot"
    if vu.point_in_rect(x_mm, y_mm,
                        tc.PLAYER_GOAL_X_MIN, tc.PLAYER_GOAL_X_MAX,
                        tc.PLAYER_GOAL_Y_MIN, tc.PLAYER_GOAL_Y_MAX):
        return "player"
    return None


# ---------------------------------------------------------------------------
# HUD drawing helpers
# ---------------------------------------------------------------------------

def draw_goal_regions_topdown(warped: np.ndarray) -> None:
    def mm_to_px(x_mm, y_mm):
        return (int(x_mm * tc.PX_PER_MM + tc.HALF_LENGTH_MM * tc.PX_PER_MM),
                int(y_mm * tc.PX_PER_MM + tc.HALF_WIDTH_MM * tc.PX_PER_MM))

    cv.rectangle(warped,
                 mm_to_px(tc.ROBOT_GOAL_X_MIN, tc.ROBOT_GOAL_Y_MIN),
                 mm_to_px(tc.ROBOT_GOAL_X_MAX, tc.ROBOT_GOAL_Y_MAX),
                 (0, 255, 0), 2)
    cv.rectangle(warped,
                 mm_to_px(tc.PLAYER_GOAL_X_MIN, tc.PLAYER_GOAL_Y_MIN),
                 mm_to_px(tc.PLAYER_GOAL_X_MAX, tc.PLAYER_GOAL_Y_MAX),
                 (255, 0, 255), 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        calib = vu.load_calibration()
    except FileNotFoundError as exc:
        print(f"[fsm] {exc}")
        return 2

    H = calib["H"]
    print(f"[fsm] loaded calibration: reprojection err = {calib['err_mm']:.2f} mm, "
          f"flip_x = {calib['flip_x']}")

    vid = cv.VideoCapture(0, cv.CAP_V4L2)
    vid.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    vid.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    vid.set(cv.CAP_PROP_FPS, 100)
    vid.set(cv.CAP_PROP_BUFFERSIZE, 1)

    lower1, upper1, lower2, upper2 = build_puck_hsv_windows(PUCK_TARGET_BGR)

    # FSM state
    current_state = "SEARCHING"
    robot_score = 0
    player_score = 0
    frames_visible = 0
    frames_lost = 0
    score_cooldown = 0

    # Trajectory (world-mm). Each entry: (x_mm, y_mm, t_s).
    trajectory: deque = deque(maxlen=TRAJECTORY_BUFFER_FRAMES)
    last_mm: Optional[Tuple[float, float]] = None
    last_vel_mm_s: Tuple[float, float] = (0.0, 0.0)
    potential_goal: Optional[str] = None

    tn = time.time()
    tElapsed = 0.0

    print("[fsm] Press q to quit, c to re-calibrate (exits with code 2).")

    while True:
        # --- timing ---------------------------------------------------------
        t_now = time.time() - tn
        deltaT = t_now - tElapsed
        tElapsed = t_now
        fps = 1.0 / deltaT if deltaT > 0 else 0.0

        # --- capture & preprocess ------------------------------------------
        ret, raw = vid.read()
        if not ret:
            print("[fsm] Failed to read frame.")
            break

        frame = vu.rotate_image(raw)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower1, upper1)
        if lower2 is not None:
            mask = cv.bitwise_or(mask, cv.inRange(hsv, lower2, upper2))

        # --- puck detection (pixels -> mm) ---------------------------------
        puck_px = detect_puck_pixel(mask, frame_for_draw=frame)
        puck_mm: Optional[Tuple[float, float]] = None
        if puck_px is not None:
            u, v = puck_px
            xm, ym = vu.pixel_to_mm(H, u, v)
            # Reject detections outside the table with a small guard band --
            # they're almost always false positives on the rails.
            if (abs(xm) < tc.HALF_LENGTH_MM + 40.0 and
                    abs(ym) < tc.HALF_WIDTH_MM + 40.0):
                puck_mm = (xm, ym)

        # --- state machine -------------------------------------------------
        if score_cooldown > 0:
            # Hold everything still for a moment after a goal so the HUD
            # flashes the score and the trajectory buffer can reset.
            score_cooldown -= 1
            if score_cooldown == 0:
                current_state = "SEARCHING"
                trajectory.clear()
                last_mm = None
                frames_visible = 0
                frames_lost = 0
                potential_goal = None

        elif current_state == "SEARCHING":
            if puck_mm is not None:
                frames_visible += 1
                if frames_visible > MIN_VISIBLE_FRAMES:
                    current_state = "TRACKING"
                    frames_lost = 0
                    potential_goal = None
                    trajectory.clear()
                    trajectory.append((puck_mm[0], puck_mm[1], t_now))
                    last_mm = puck_mm
            else:
                frames_visible = 0

        elif current_state == "TRACKING":
            if puck_mm is not None:
                trajectory.append((puck_mm[0], puck_mm[1], t_now))
                last_mm = puck_mm
                frames_lost = 0
                potential_goal = None  # puck re-appeared -> cancel goal intent

                # Compute instantaneous velocity (mm/s) from the last 2 samples.
                if len(trajectory) >= 2:
                    x0, y0, t0 = trajectory[-2]
                    x1, y1, t1 = trajectory[-1]
                    dt = t1 - t0
                    if dt > 1e-4:
                        last_vel_mm_s = ((x1 - x0) / dt, (y1 - y0) / dt)
            else:
                frames_lost += 1
                # On the very first lost frame, snapshot the trajectory intent.
                if frames_lost == 1:
                    potential_goal = classify_goal_intent(trajectory)
                # Also: if the last known position was already inside a goal
                # box (slow puck), that's obviously a goal too.
                if potential_goal is None and last_mm is not None:
                    potential_goal = is_in_goal(*last_mm)

                if frames_lost > LOST_FRAMES_THRESHOLD:
                    if potential_goal == "robot":
                        robot_score += 1
                        print(f"[fsm] ROBOT scores! Robot {robot_score} | Player {player_score}")
                    elif potential_goal == "player":
                        player_score += 1
                        print(f"[fsm] PLAYER scores! Robot {robot_score} | Player {player_score}")
                    else:
                        last_txt = f"({last_mm[0]:.1f}, {last_mm[1]:.1f}) mm" if last_mm else "unknown"
                        print(f"[fsm] Puck lost at {last_txt} -- no goal.")
                    score_cooldown = POST_SCORE_FREEZE_FRAMES

        # --- HUD (camera view) ---------------------------------------------
        speed_mm_s = math.hypot(*last_vel_mm_s)
        cv.putText(frame, f"STATE: {current_state}", (50, 70),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv.putText(frame, f"FPS: {int(fps)}", (50, 110),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(frame, f"Speed: {speed_mm_s:.0f} mm/s", (50, 150),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if last_mm is not None:
            cv.putText(frame,
                       f"Puck(mm): ({last_mm[0]:+.1f}, {last_mm[1]:+.1f})",
                       (50, 190), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv.putText(frame, f"Robot: {robot_score}   Player: {player_score}",
                   (50, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if potential_goal is not None:
            cv.putText(frame, f"intent: {potential_goal} goal",
                       (50, 280), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # --- top-down view with goal rects + puck ---------------------------
        warped = vu.warp_to_topdown(frame, H)
        draw_goal_regions_topdown(warped)
        if last_mm is not None:
            cx = int(last_mm[0] * tc.PX_PER_MM + tc.HALF_LENGTH_MM * tc.PX_PER_MM)
            cy = int(last_mm[1] * tc.PX_PER_MM + tc.HALF_WIDTH_MM * tc.PX_PER_MM)
            cv.circle(warped, (cx, cy), 12, (0, 255, 255), 2)
            # Velocity arrow (scaled for visibility).
            vx, vy = last_vel_mm_s
            cv.arrowedLine(warped, (cx, cy),
                           (int(cx + vx * 0.05), int(cy + vy * 0.05)),
                           (0, 255, 0), 2, tipLength=0.2)

        cv.imshow('frame', frame)
        cv.imshow('top-down (mm)', warped)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            print("[fsm] exiting so calibration can be re-run.")
            vid.release()
            cv.destroyAllWindows()
            return 2

    vid.release()
    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
