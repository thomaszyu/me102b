"""
Debug visualizer for the air hockey player.
Runs vision only (no motors) and draws:
- Puck position + velocity vector
- Predicted puck trajectory (with wall bounces)
- Mallet position
- Strategy decision + mallet target
- Table outline from calibration JSON (with rounded corners)
- Defense line, attack zone, goals

Usage:
    python air_hockey_debug.py
"""

import sys
import os
import cv2 as cv
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'vision_code'))

from air_hockey_player import (
    PuckTracker, predict_puck_trajectory, predict_intercept_y, decide_strategy,
    TABLE_X_MIN, TABLE_X_MAX, TABLE_Y_MIN, TABLE_Y_MAX, CORNER_RADIUS,
    PUCK_X_MIN, PUCK_X_MAX, PUCK_Y_MIN, PUCK_Y_MAX, PUCK_RADIUS,
    MALLET_X_MIN, MALLET_X_MAX, MALLET_Y_MIN, MALLET_Y_MAX,
    DEFEND_X, ATTACK_LIMIT_X, ATTACK_ZONE_X, OPPONENT_GOAL_X,
)

# Display config
DISPLAY_W = 960
DISPLAY_H = 540
MARGIN = 40


def mm_to_px(x_mm, y_mm):
    """Convert vision mm coords to display pixel coords."""
    pad = 20
    px = MARGIN + (x_mm - (TABLE_X_MIN - pad)) / ((TABLE_X_MAX + pad) - (TABLE_X_MIN - pad)) * (DISPLAY_W - 2 * MARGIN)
    py = MARGIN + ((TABLE_Y_MAX + pad) - y_mm) / ((TABLE_Y_MAX + pad) - (TABLE_Y_MIN - pad)) * (DISPLAY_H - 2 * MARGIN)
    return int(px), int(py)


def _draw_rounded_rect(frame, xn, xx, yn, yx, r, color, thickness):
    """Draw a rounded rectangle."""
    if r > 1:
        cv.line(frame, mm_to_px(xn + r, yx), mm_to_px(xx - r, yx), color, thickness)
        cv.line(frame, mm_to_px(xn + r, yn), mm_to_px(xx - r, yn), color, thickness)
        cv.line(frame, mm_to_px(xn, yx - r), mm_to_px(xn, yn + r), color, thickness)
        cv.line(frame, mm_to_px(xx, yx - r), mm_to_px(xx, yn + r), color, thickness)
        for cx, cy, a0 in [
            (xn + r, yx - r, 90),
            (xx - r, yx - r, 0),
            (xx - r, yn + r, 270),
            (xn + r, yn + r, 180),
        ]:
            pts = []
            for a in range(0, 91, 3):
                angle = np.radians(a0 + a)
                pts.append(mm_to_px(cx + r * np.cos(angle), cy + r * np.sin(angle)))
            for i in range(len(pts) - 1):
                cv.line(frame, pts[i], pts[i + 1], color, thickness)
    else:
        cv.rectangle(frame, mm_to_px(xn, yx), mm_to_px(xx, yn), color, thickness)


def draw_table(frame):
    """Draw table wall + puck boundary, center line, goals, defense line."""
    # Actual wall
    _draw_rounded_rect(frame, TABLE_X_MIN, TABLE_X_MAX, TABLE_Y_MIN, TABLE_Y_MAX,
                       CORNER_RADIUS, (100, 100, 100), 1)

    # Puck center boundary (where the puck center can actually reach)
    pr = max(0, CORNER_RADIUS - PUCK_RADIUS)
    _draw_rounded_rect(frame, PUCK_X_MIN, PUCK_X_MAX, PUCK_Y_MIN, PUCK_Y_MAX,
                       pr, (255, 255, 255), 2)

    # Mallet safe workspace (dashed cyan)
    tl = mm_to_px(MALLET_X_MIN, MALLET_Y_MAX)
    br = mm_to_px(MALLET_X_MAX, MALLET_Y_MIN)
    cv.rectangle(frame, tl, br, (255, 255, 0), 1, cv.LINE_AA)
    cv.putText(frame, "WORKSPACE", (tl[0] + 5, tl[1] + 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

    # Center line
    cv.line(frame, mm_to_px(0, TABLE_Y_MAX), mm_to_px(0, TABLE_Y_MIN), (100, 100, 100), 1, cv.LINE_AA)

    # Defense line
    dt, db = mm_to_px(DEFEND_X, TABLE_Y_MAX), mm_to_px(DEFEND_X, TABLE_Y_MIN)
    cv.line(frame, dt, db, (0, 200, 255), 1, cv.LINE_AA)
    cv.putText(frame, "DEFEND", (dt[0] - 25, dt[1] - 8),
               cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)

    # Attack limit
    cv.line(frame, mm_to_px(ATTACK_LIMIT_X, TABLE_Y_MAX), mm_to_px(ATTACK_LIMIT_X, TABLE_Y_MIN),
            (0, 100, 200), 1, cv.LINE_AA)

    # Attack zone
    azt = mm_to_px(ATTACK_ZONE_X, TABLE_Y_MAX)
    cv.line(frame, azt, mm_to_px(ATTACK_ZONE_X, TABLE_Y_MIN), (0, 80, 150), 1, cv.LINE_AA)
    cv.putText(frame, "ATK ZONE", (azt[0] - 30, azt[1] - 8),
               cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 80, 150), 1)

    # Goals
    goal_half = 60
    g1, g2 = mm_to_px(TABLE_X_MIN, goal_half), mm_to_px(TABLE_X_MIN, -goal_half)
    cv.line(frame, g1, g2, (0, 0, 255), 4)
    cv.putText(frame, "OUR GOAL", (g1[0] + 5, g1[1] - 5),
               cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    g1, g2 = mm_to_px(TABLE_X_MAX, goal_half), mm_to_px(TABLE_X_MAX, -goal_half)
    cv.line(frame, g1, g2, (255, 0, 255), 4)
    cv.putText(frame, "THEIR GOAL", (g1[0] - 80, g1[1] - 5),
               cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)


def draw_puck(frame, puck):
    if not puck.initialized:
        return
    px, py = mm_to_px(puck.pos[0], puck.pos[1])
    cv.circle(frame, (px, py), 12, (0, 255, 0), -1)
    cv.circle(frame, (px, py), 12, (0, 200, 0), 2)

    vel_scale = 0.3
    vx_px = int(puck.vel[0] * vel_scale * (DISPLAY_W - 2 * MARGIN) / (TABLE_X_MAX - TABLE_X_MIN + 40))
    vy_px = int(-puck.vel[1] * vel_scale * (DISPLAY_H - 2 * MARGIN) / (TABLE_Y_MAX - TABLE_Y_MIN + 40))
    cv.arrowedLine(frame, (px, py), (px + vx_px, py + vy_px), (0, 255, 100), 2, tipLength=0.3)

    speed = np.linalg.norm(puck.vel)
    cv.putText(frame, f"{speed:.0f}mm/s", (px + 15, py - 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)


def draw_predicted_trajectory(frame, puck):
    if not puck.initialized or np.linalg.norm(puck.vel) < 10:
        return
    trajectory = predict_puck_trajectory(puck.pos, puck.vel, dt=0.02, num_steps=150)
    pts = [mm_to_px(puck.pos[0], puck.pos[1])]
    for pos in trajectory:
        pts.append(mm_to_px(pos[0], pos[1]))
    for i in range(len(pts) - 1):
        alpha = max(0, 1.0 - i / len(pts))
        color = (0, int(255 * alpha), int(100 * alpha))
        cv.line(frame, pts[i], pts[i + 1], color, 1, cv.LINE_AA)


def draw_intercept(frame, puck):
    if not puck.initialized:
        return
    intercept_y = predict_intercept_y(puck.pos, puck.vel, DEFEND_X)
    if intercept_y is not None:
        ix, iy = mm_to_px(DEFEND_X, intercept_y)
        cv.circle(frame, (ix, iy), 8, (0, 200, 255), 2)
        cv.circle(frame, (ix, iy), 3, (0, 200, 255), -1)
        cv.putText(frame, f"Y={intercept_y:.0f}", (ix + 10, iy),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)


def draw_mallet(frame, mallet_xy, valid):
    if not valid:
        return
    mx, my = mm_to_px(mallet_xy[0], mallet_xy[1])
    cv.circle(frame, (mx, my), 15, (0, 165, 255), 2)
    cv.circle(frame, (mx, my), 5, (0, 165, 255), -1)


def draw_target(frame, strategy, target_xy):
    if target_xy is None:
        return
    if isinstance(target_xy, (int, float)):
        tx, ty = mm_to_px(DEFEND_X, target_xy)
    else:
        tx, ty = mm_to_px(target_xy[0], target_xy[1])
    color = (0, 0, 255) if strategy == 'DEFEND' else (255, 100, 0) if strategy == 'ATTACK' else (128, 128, 128)
    cv.drawMarker(frame, (tx, ty), color, cv.MARKER_CROSS, 20, 2)
    cv.putText(frame, strategy, (tx + 12, ty - 5),
               cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def build_debug_homography():
    """
    Build a homography that warps the rectified camera image (862x480, world coords)
    onto the debug display, mapping table mm coords to debug pixel coords.
    """
    # Source: 4 corners in the rectified camera image (pixel coords in 862x480 frame)
    # These correspond to the world_pts from vision.py calibration
    # world_pts maps to: TL(-273,240), TR(273,240), BR(273,-240), BL(-273,-240)
    # In the 862x480 rectified image, these map to display_pts from vision.py
    from vision import display_pts as cam_display_pts
    src_pts = cam_display_pts.copy()  # where the world corners appear in the camera image

    # Destination: where those same world corners appear in our debug display
    dst_pts = np.array([
        mm_to_px(-273,  240),
        mm_to_px( 273,  240),
        mm_to_px( 273, -240),
        mm_to_px(-273, -240),
    ], dtype="float32")

    return cv.getPerspectiveTransform(src_pts, dst_pts)


def main():
    from vision import VisionSystem

    vis = VisionSystem()
    vis.start(show_display=False)

    puck = PuckTracker(alpha_pos=0.5, alpha_vel=0.3, max_jump=100.0)

    src = "calibration" if CORNER_RADIUS > 0 else "defaults"
    print(f"Debug visualizer running (table bounds from {src}). Press 'q' to quit.")

    cv.namedWindow('Air Hockey Debug')
    last_time = time.time()

    # Build homography from camera rectified → debug display
    H_cam_to_debug = None
    if vis.H_display is not None:
        H_cam_to_debug = build_debug_homography()

    OVERLAY_ALPHA = 0.4  # camera feed dimming (0=invisible, 1=full brightness)

    while True:
        now = time.time()
        dt = now - last_time
        last_time = now
        fps = 1.0 / dt if dt > 0 else 0.0

        mallet_reading, puck_reading, _ = vis.get_positions()
        puck.update(puck_reading, puck_reading[2], now)

        mallet_xy = np.array([mallet_reading[0], mallet_reading[1]])
        strategy, target_data = decide_strategy(puck, mallet_xy)

        # --- Background: camera feed or black ---
        frame = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)

        if vis.frame is not None and vis.H_display is not None:
            # Build homography lazily (in case calibration loaded after start)
            if H_cam_to_debug is None:
                H_cam_to_debug = build_debug_homography()

            # Warp raw camera → rectified → debug display coords
            H_combined = H_cam_to_debug @ vis.H_display
            warped = cv.warpPerspective(vis.frame, H_combined, (DISPLAY_W, DISPLAY_H))
            frame = (warped * OVERLAY_ALPHA).astype(np.uint8)

        # --- Draw overlays ---
        draw_table(frame)
        draw_predicted_trajectory(frame, puck)
        draw_intercept(frame, puck)
        draw_puck(frame, puck)
        draw_mallet(frame, mallet_xy, mallet_reading[2])
        draw_target(frame, strategy, target_data)

        cv.putText(frame, f"FPS: {int(fps)}", (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv.putText(frame, f"Strategy: {strategy}", (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if puck.initialized:
            cv.putText(frame, f"Puck: ({puck.pos[0]:.0f}, {puck.pos[1]:.0f}) vel=({puck.vel[0]:.0f}, {puck.vel[1]:.0f})",
                       (10, DISPLAY_H - 40), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        if mallet_reading[2]:
            cv.putText(frame, f"Mallet: ({mallet_xy[0]:.0f}, {mallet_xy[1]:.0f})",
                       (10, DISPLAY_H - 20), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        cv.imshow('Air Hockey Debug', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    vis.stop()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
