"""
On-site calibration for the air-hockey table.

Run this once whenever the camera or table is moved. It opens the live camera
feed, auto-detects the three blue lines, centre circle and centre dot, then
computes a planar homography that converts camera pixels to world
millimetres (origin at the centre dot, +x toward the robot side).

Keys while running
------------------
    c   Confirm current detection and save calibration.
    r   Re-capture and re-detect (if something moved / the overlay looks off).
    f   Flip the x-axis assignment (use this if the warped top-down view
        shows the ROBOT goal on the left instead of the right).
    m   Toggle the HSV colour-mask debug window.
    q   Quit without saving.

Verification
------------
After each successful detection, a warped top-down view is displayed next to
the raw camera view. The table should appear as a clean rectangle, the three
blue lines should be perfectly vertical, the centre circle should be round,
and the centre dot should be at the exact middle of the output window. The
console also prints the mean reprojection error in mm -- a healthy
calibration is well under ~5 mm.
"""

from __future__ import annotations

import os
import sys
import time

import cv2 as cv
import numpy as np

import table_config as tc
import vision_utils as vu


# ---------------------------------------------------------------------------
# Camera open (matches fsm.py settings)
# ---------------------------------------------------------------------------

def open_camera() -> cv.VideoCapture:
    # Try the same backend the FSM uses; fall back for dev machines.
    try:
        cap = cv.VideoCapture(0, cv.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError
    except Exception:
        cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv.CAP_PROP_FPS, 100)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera. Check your /dev/video0 device.")
    return cap


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

def draw_detection_overlay(frame: np.ndarray, feat: vu.BlueFeatures,
                           err_mm: float | None, flip_x: bool) -> np.ndarray:
    out = frame.copy()
    # Lines in the sorted order (left, mid, right) drawn in distinct colours.
    colours = [(0, 255, 255), (255, 255, 0), (255, 0, 255)]
    labels = ["L", "C", "R"]
    for line, colour, lbl in zip(feat.lines_sorted_by_x, colours, labels):
        p1 = tuple(int(v) for v in line.p1)
        p2 = tuple(int(v) for v in line.p2)
        cv.line(out, p1, p2, colour, 3)
        cv.circle(out, p1, 6, colour, -1)
        cv.circle(out, p2, 6, colour, -1)
        cv.putText(out, f"{lbl}(top)", (p1[0] + 8, p1[1] - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        cv.putText(out, f"{lbl}(bot)", (p2[0] + 8, p2[1] + 18),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

    # Circle + dot.
    cc = (int(feat.circle_center[0]), int(feat.circle_center[1]))
    cv.circle(out, cc, int(feat.circle_radius), (0, 255, 0), 2)
    dc = (int(feat.dot_center[0]), int(feat.dot_center[1]))
    cv.drawMarker(out, dc, (0, 0, 255), markerType=cv.MARKER_CROSS,
                  markerSize=28, thickness=2)

    # Header.
    lines_ok = "OK" if err_mm is not None else "--"
    cv.putText(out, f"calibration: {lines_ok}   flip_x={flip_x}",
               (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    if err_mm is not None:
        cv.putText(out, f"mean reproj err: {err_mm:.2f} mm",
                   (20, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv.putText(out, "[c]=save  [r]=retry  [f]=flip x  [m]=mask  [q]=quit",
               (20, out.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6,
               (220, 220, 220), 2)
    return out


def draw_topdown_grid(warped: np.ndarray) -> np.ndarray:
    """Overlay a mm grid + expected feature positions on the warped image."""
    h, w = warped.shape[:2]
    out = warped.copy()

    def mm_to_px(x_mm: float, y_mm: float):
        return (int(x_mm * tc.PX_PER_MM + tc.HALF_LENGTH_MM * tc.PX_PER_MM),
                int(y_mm * tc.PX_PER_MM + tc.HALF_WIDTH_MM * tc.PX_PER_MM))

    # 100 mm grid.
    for x_mm in range(-400, 401, 100):
        p1 = mm_to_px(x_mm, -tc.HALF_WIDTH_MM)
        p2 = mm_to_px(x_mm, +tc.HALF_WIDTH_MM)
        cv.line(out, p1, p2, (60, 60, 60), 1)
    for y_mm in range(-200, 201, 100):
        p1 = mm_to_px(-tc.HALF_LENGTH_MM, y_mm)
        p2 = mm_to_px(+tc.HALF_LENGTH_MM, y_mm)
        cv.line(out, p1, p2, (60, 60, 60), 1)

    # Expected blue lines.
    for x_mm, lbl in [(tc.LEFT_LINE_X_MM, "L"), (0.0, "C"), (tc.RIGHT_LINE_X_MM, "R")]:
        p1 = mm_to_px(x_mm, -tc.HALF_WIDTH_MM)
        p2 = mm_to_px(x_mm, +tc.HALF_WIDTH_MM)
        cv.line(out, p1, p2, (255, 200, 0), 1)
        cv.putText(out, lbl, (p1[0] + 4, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 200, 0), 1)

    # Expected centre circle.
    centre_px = mm_to_px(0.0, 0.0)
    cv.circle(out, centre_px, int(tc.CENTER_CIRCLE_RADIUS_MM * tc.PX_PER_MM),
              (0, 255, 0), 1)
    cv.drawMarker(out, centre_px, (0, 0, 255), markerType=cv.MARKER_CROSS,
                  markerSize=20, thickness=2)

    # Goal regions.
    def draw_goal(xmin, xmax, ymin, ymax, colour):
        p1 = mm_to_px(xmin, ymin)
        p2 = mm_to_px(xmax, ymax)
        cv.rectangle(out, p1, p2, colour, 2)

    draw_goal(tc.ROBOT_GOAL_X_MIN, tc.ROBOT_GOAL_X_MAX,
              tc.ROBOT_GOAL_Y_MIN, tc.ROBOT_GOAL_Y_MAX, (0, 255, 0))
    draw_goal(tc.PLAYER_GOAL_X_MIN, tc.PLAYER_GOAL_X_MAX,
              tc.PLAYER_GOAL_Y_MIN, tc.PLAYER_GOAL_Y_MAX, (255, 0, 255))

    cv.putText(out, "warped top-down (1 px = 1 mm)", (10, h - 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    cap = open_camera()
    print("[calibration] Camera opened. Press c to save, r to retry, "
          "f to flip x-axis, m to toggle mask window, q to quit.")

    flip_x = False
    show_mask = False
    last_good = None   # (feat, H, err_mm, frame_size_hw)

    try:
        while True:
            ret, raw = cap.read()
            if not ret:
                print("[calibration] camera read failed"); time.sleep(0.1); continue

            frame = vu.rotate_image(raw)
            feat = vu.detect_blue_features(frame)

            H = None
            err_mm = None
            if feat is not None:
                try:
                    H, err_mm = vu.compute_homography(feat, flip_x=flip_x)
                    last_good = (feat, H, err_mm, frame.shape[:2])
                except Exception as exc:
                    print(f"[calibration] homography failed: {exc}")
                    feat = None

            overlay = (draw_detection_overlay(frame, feat, err_mm, flip_x)
                       if feat is not None else frame.copy())
            if feat is None:
                cv.putText(overlay, "no features detected",
                           (20, 70), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                           (0, 0, 255), 2)

            cv.imshow("calibration - camera", overlay)

            if H is not None:
                warped = vu.warp_to_topdown(frame, H)
                warped_overlay = draw_topdown_grid(warped)
                cv.imshow("calibration - top-down", warped_overlay)

            if show_mask:
                cv.imshow("blue-mask", vu.build_blue_mask(frame))
            else:
                try:
                    cv.destroyWindow("blue-mask")
                except cv.error:
                    pass

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[calibration] quit without saving.")
                return 1
            elif key == ord('r'):
                last_good = None
                print("[calibration] retrying...")
            elif key == ord('f'):
                flip_x = not flip_x
                print(f"[calibration] flip_x -> {flip_x}")
            elif key == ord('m'):
                show_mask = not show_mask
            elif key == ord('c'):
                if last_good is None:
                    print("[calibration] no valid detection yet -- cannot save.")
                    continue
                feat_saved, H_saved, err_saved, sz = last_good
                out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        tc.CALIBRATION_FILE)
                vu.save_calibration(out_path, H_saved, sz, err_saved, flip_x)
                print(f"[calibration] saved to {out_path}")
                print(f"[calibration] mean reprojection error = {err_saved:.3f} mm")
                return 0
    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
