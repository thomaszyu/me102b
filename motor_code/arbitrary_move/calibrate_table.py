"""
Table boundary calibration by tracing the outline with the puck.

Slowly drag the puck along all walls. The script records the path
and fits a rounded rectangle to it.

Usage:
    python calibrate_table.py

    Drag puck along all 4 walls + corners. Press 'f' to fit, 's' to save, 'q' to quit.
"""

import sys
import os
import cv2 as cv
import numpy as np
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'vision_code'))

CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "table_calibration.json")

DISPLAY_W = 960
DISPLAY_H = 540
MARGIN = 60

# How often to record a point (seconds)
SAMPLE_INTERVAL = 0.05
# Minimum puck movement between samples (mm) — skip if stationary
MIN_MOVE = 2.0
# Puck radius (mm) — traced path is offset from the actual wall by this amount
PUCK_RADIUS = 31.0


def fit_rounded_rect(points):
    """
    Fit a rounded rectangle to a set of boundary points.
    Returns (bounds_dict, corner_radius).
    """
    pts = np.array(points)

    # Step 1: rough bounds from extremes
    x_min_raw, y_min_raw = pts.min(axis=0)
    x_max_raw, y_max_raw = pts.max(axis=0)

    # Step 2: estimate corner radius
    # Points near the geometric corners will be further from the corner
    # than the wall points are from their respective walls.
    # The corner "cuts in" by the radius.
    cx = (x_min_raw + x_max_raw) / 2
    cy = (y_min_raw + y_max_raw) / 2

    # Split into 4 quadrants
    corners_radii = []
    for sx, sy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        # Points in this corner quadrant
        corner_pts = pts[(pts[:, 0] - cx) * sx > 0]
        corner_pts = corner_pts[(corner_pts[:, 1] - cy) * sy > 0]
        if len(corner_pts) < 3:
            continue

        # The geometric corner
        gx = x_max_raw if sx > 0 else x_min_raw
        gy = y_max_raw if sy > 0 else y_min_raw

        # Distance from geometric corner to each point
        dists = np.sqrt((corner_pts[:, 0] - gx)**2 + (corner_pts[:, 1] - gy)**2)

        # Points closest to the corner define the radius
        # The minimum distance from the corner to the traced arc ≈ r*(sqrt(2)-1)
        # but more practically, the traced points ON the arc have distance ≈ r from
        # the center of the arc, which is at (gx ∓ r, gy ∓ r)
        # Simplification: the "cut-in" at the corner ≈ min distance to geometric corner
        min_dist = np.percentile(dists, 10)  # use 10th percentile for robustness
        # For a rounded corner, dist_to_geometric_corner ≈ r * (sqrt(2) - 1)
        r = min_dist / (np.sqrt(2) - 1) if min_dist > 3 else 0
        corners_radii.append(r)

    corner_radius = float(np.median(corners_radii)) if corners_radii else 0.0

    # Step 3: shrink bounds by puck radius (traced path = wall + puck_radius)
    bounds = {
        'x_min': float(x_min_raw + PUCK_RADIUS),
        'x_max': float(x_max_raw - PUCK_RADIUS),
        'y_min': float(y_min_raw + PUCK_RADIUS),
        'y_max': float(y_max_raw - PUCK_RADIUS),
    }

    # Corner radius shrinks too (traced arc is puck_radius outside the actual wall arc)
    corner_radius = max(0.0, corner_radius - PUCK_RADIUS)

    return bounds, corner_radius


def mm_to_px(x_mm, y_mm, bounds):
    pad = 30
    x_min = bounds.get('x_min', -273) - pad
    x_max = bounds.get('x_max', 273) + pad
    y_min = bounds.get('y_min', -240) - pad
    y_max = bounds.get('y_max', 240) + pad
    px = MARGIN + (x_mm - x_min) / (x_max - x_min) * (DISPLAY_W - 2 * MARGIN)
    py = MARGIN + (y_max - y_mm) / (y_max - y_min) * (DISPLAY_H - 2 * MARGIN)
    return int(px), int(py)


def draw_rounded_rect(frame, bounds, r, b_ref, color=(255, 255, 255), thickness=2):
    def p(x, y):
        return mm_to_px(x, y, b_ref)

    xn, xx = bounds['x_min'], bounds['x_max']
    yn, yx = bounds['y_min'], bounds['y_max']

    # Straight edges
    cv.line(frame, p(xn + r, yx), p(xx - r, yx), color, thickness)
    cv.line(frame, p(xn + r, yn), p(xx - r, yn), color, thickness)
    cv.line(frame, p(xn, yx - r), p(xn, yn + r), color, thickness)
    cv.line(frame, p(xx, yx - r), p(xx, yn + r), color, thickness)

    # Corner arcs
    for cx, cy, a0 in [
        (xn + r, yx - r, 90),
        (xx - r, yx - r, 0),
        (xx - r, yn + r, 270),
        (xn + r, yn + r, 180),
    ]:
        arc_pts = []
        for a in range(0, 91, 3):
            angle = np.radians(a0 + a)
            arc_pts.append(p(cx + r * np.cos(angle), cy + r * np.sin(angle)))
        for i in range(len(arc_pts) - 1):
            cv.line(frame, arc_pts[i], arc_pts[i + 1], color, thickness)


def save_calibration(bounds, corner_radius, filepath=CALIBRATION_FILE):
    data = {'bounds': bounds, 'corner_radius': corner_radius}
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {filepath}")


def load_calibration(filepath=CALIBRATION_FILE):
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        return json.load(f)


def main():
    from vision import VisionSystem

    vis = VisionSystem()
    vis.start(show_display=False)

    trace_points = []
    last_sample_time = 0.0
    last_pos = None
    fitted_bounds = None
    fitted_radius = 0.0

    # Default display bounds
    display_bounds = {'x_min': -300, 'x_max': 300, 'y_min': -260, 'y_max': 260}

    print("Trace the puck along all walls and corners.")
    print("  'f' = fit    's' = save    'c' = clear    'q' = quit")

    cv.namedWindow('Table Calibration')
    last_time = time.time()

    while True:
        now = time.time()
        dt_frame = now - last_time
        last_time = now
        fps = 1.0 / dt_frame if dt_frame > 0 else 0.0

        _, puck_reading, _ = vis.get_positions()

        # Record trace points
        if puck_reading[2] and now - last_sample_time > SAMPLE_INTERVAL:
            pos = np.array([puck_reading[0], puck_reading[1]])
            if last_pos is None or np.linalg.norm(pos - last_pos) > MIN_MOVE:
                trace_points.append(pos)
                last_pos = pos
                last_sample_time = now

        # Display bounds
        b = fitted_bounds if fitted_bounds else display_bounds

        # Draw
        frame = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)

        # Fitted shape
        if fitted_bounds:
            draw_rounded_rect(frame, fitted_bounds, fitted_radius, b, (255, 255, 255), 2)

            # Dimensions
            w = fitted_bounds['x_max'] - fitted_bounds['x_min']
            h = fitted_bounds['y_max'] - fitted_bounds['y_min']
            cv.putText(frame, f"Bounds: X[{fitted_bounds['x_min']:.0f}, {fitted_bounds['x_max']:.0f}]  "
                       f"Y[{fitted_bounds['y_min']:.0f}, {fitted_bounds['y_max']:.0f}]",
                       (10, DISPLAY_H - 40), cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv.putText(frame, f"Size: {w:.0f} x {h:.0f} mm   Corner R: {fitted_radius:.0f} mm",
                       (10, DISPLAY_H - 20), cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Trace points
        for i, pt in enumerate(trace_points):
            px, py = mm_to_px(pt[0], pt[1], b)
            # Color fades from blue (old) to cyan (new)
            t = i / max(len(trace_points), 1)
            color = (int(255 * (1 - t)), int(200 * t + 55), 0)
            cv.circle(frame, (px, py), 2, color, -1)

        # Current puck position
        if puck_reading[2]:
            px, py = mm_to_px(puck_reading[0], puck_reading[1], b)
            cv.circle(frame, (px, py), 8, (0, 255, 0), -1)
            cv.putText(frame, f"({puck_reading[0]:.0f}, {puck_reading[1]:.0f})",
                       (px + 12, py - 5), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # HUD
        cv.putText(frame, f"FPS: {int(fps)}   Points: {len(trace_points)}", (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv.putText(frame, "'f'=fit  's'=save  'c'=clear  'q'=quit", (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        cv.imshow('Table Calibration', frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            if len(trace_points) < 20:
                print("Need more points — keep tracing.")
            else:
                fitted_bounds, fitted_radius = fit_rounded_rect(trace_points)
                display_bounds = fitted_bounds
                print(f"Fit: {fitted_bounds}  R={fitted_radius:.1f}mm")
        elif key == ord('s'):
            if fitted_bounds is None:
                if len(trace_points) >= 20:
                    fitted_bounds, fitted_radius = fit_rounded_rect(trace_points)
                else:
                    print("Need more points.")
                    continue
            save_calibration(fitted_bounds, fitted_radius)
        elif key == ord('c'):
            trace_points.clear()
            last_pos = None
            fitted_bounds = None
            fitted_radius = 0.0
            display_bounds = {'x_min': -300, 'x_max': 300, 'y_min': -260, 'y_max': 260}
            print("Cleared.")

    vis.stop()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
