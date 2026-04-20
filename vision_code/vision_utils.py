"""
Shared vision utilities: blue-feature detection for on-site calibration,
homography I/O, pixel <-> world-mm conversion, and top-down warping.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np

import table_config as tc


# ---------------------------------------------------------------------------
# Image rotation used by the camera pipeline (kept here so both the main FSM
# and the calibration script apply the *identical* transform before any
# coordinate work -- otherwise the homography would be meaningless).
# ---------------------------------------------------------------------------
CAMERA_ROTATION_DEG = -180.6


def rotate_image(mat: np.ndarray, angle: float = CAMERA_ROTATION_DEG) -> np.ndarray:
    """Rotate around image centre, expanding the canvas to avoid cropping."""
    height, width = mat.shape[:2]
    image_center = (width / 2.0, height / 2.0)
    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]
    return cv.warpAffine(mat, rotation_mat, (bound_w, bound_h))


# ---------------------------------------------------------------------------
# Blue-feature detection
# ---------------------------------------------------------------------------

@dataclass
class LineFeature:
    """A detected blue line. `p1` / `p2` are the two endpoints (pixel coords),
    ordered so that `p1.y < p2.y` (i.e. p1 is the 'top' endpoint in image)."""
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    center: Tuple[float, float]
    length: float


@dataclass
class BlueFeatures:
    """Result of detecting the three blue lines + circle + dot in a frame."""
    lines_sorted_by_x: List[LineFeature]    # length 3; leftmost first
    circle_center: Tuple[float, float]
    circle_radius: float
    dot_center: Tuple[float, float]


def build_blue_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Return a cleaned-up binary mask of blue pixels."""
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, np.array(tc.BLUE_HSV_LOWER), np.array(tc.BLUE_HSV_UPPER))
    # Close small gaps inside lines, then remove tiny speckle noise.
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k, iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k, iterations=1)
    return mask


def _contour_endpoints_from_min_area_rect(contour: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Given a contour representing a long-thin blue line, return the two
    endpoints at the centres of the *short* sides of its minAreaRect, plus
    the line length (long side of the rect)."""
    rect = cv.minAreaRect(contour)   # ((cx,cy),(w,h),angle)
    box = cv.boxPoints(rect).astype(np.float32)   # 4 x 2
    # Identify the long axis: pair the two edges and take the longer pair.
    e01 = np.linalg.norm(box[0] - box[1])
    e12 = np.linalg.norm(box[1] - box[2])
    if e01 >= e12:
        # Short sides are 1-2 and 3-0; midpoints of those are the endpoints.
        p1 = tuple(((box[1] + box[2]) / 2.0).tolist())
        p2 = tuple(((box[3] + box[0]) / 2.0).tolist())
        length = e01
    else:
        p1 = tuple(((box[0] + box[1]) / 2.0).tolist())
        p2 = tuple(((box[2] + box[3]) / 2.0).tolist())
        length = e12
    return p1, p2, length


def detect_blue_features(frame_bgr: np.ndarray) -> Optional[BlueFeatures]:
    """
    Detect the three blue lines, the centre circle, and the centre dot.
    Returns None if the scene cannot be classified with confidence.
    """
    mask = build_blue_mask(frame_bgr)
    h, w = mask.shape[:2]
    img_area = h * w

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) < 4:
        return None

    # Classify every contour.
    line_candidates = []
    dot_candidates = []
    circle_candidates = []

    for c in contours:
        area = cv.contourArea(c)
        if area < 50:
            continue  # noise

        # Shape descriptors.
        perim = cv.arcLength(c, True)
        if perim <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perim * perim)     # 1.0 for a perfect circle

        (cx, cy), _radius = cv.minEnclosingCircle(c)
        rect = cv.minAreaRect(c)
        (w_rect, h_rect) = rect[1]
        if min(w_rect, h_rect) <= 0:
            continue
        aspect = max(w_rect, h_rect) / min(w_rect, h_rect)

        # Long, thin -> line.
        if aspect > 6.0 and max(w_rect, h_rect) > 0.15 * min(h, w):
            p1, p2, length = _contour_endpoints_from_min_area_rect(c)
            # Order so p1 is the 'top' endpoint in image (smaller y).
            if p1[1] > p2[1]:
                p1, p2 = p2, p1
            line_candidates.append(
                LineFeature(p1=p1, p2=p2, center=(cx, cy), length=length)
            )
            continue

        # Round -> either circle outline or dot.
        if circularity > 0.55:
            # Circle outline is hollow, so its contour area is *small* relative
            # to its enclosing-circle area. The dot is filled, so area/enc ~1.
            enc_area = np.pi * _radius * _radius
            fill_ratio = area / enc_area if enc_area > 0 else 0.0
            if fill_ratio > 0.6:
                dot_candidates.append((cx, cy, _radius, area))
            else:
                circle_candidates.append((cx, cy, _radius, area))

    if len(line_candidates) < 3 or not dot_candidates or not circle_candidates:
        return None

    # Pick the 3 longest line candidates (there may be duplicates from noise).
    line_candidates.sort(key=lambda L: L.length, reverse=True)
    lines = line_candidates[:3]
    # Re-sort the survivors by x position of their centre (leftmost first).
    lines.sort(key=lambda L: L.center[0])

    # Centre dot = smallest filled round blob that is *close to the middle*
    # of the three lines (it sits on the centre line).
    middle_line_cx = lines[1].center[0]
    dot_candidates.sort(
        key=lambda d: (abs(d[0] - middle_line_cx), d[3])  # close to centre line, then smaller
    )
    dot_cx, dot_cy, _dot_r, _dot_a = dot_candidates[0]

    # Centre circle = largest hollow round blob, which should also enclose the dot.
    circle_candidates.sort(key=lambda c: c[3], reverse=True)
    circ_cx, circ_cy, circ_r, _circ_a = circle_candidates[0]

    # Sanity check: dot should be inside the circle.
    if np.hypot(dot_cx - circ_cx, dot_cy - circ_cy) > circ_r:
        return None

    # Sanity check: middle line should pass *through* the dot (within a few px).
    mid_line = lines[1]
    if _perp_distance_point_to_segment((dot_cx, dot_cy), mid_line.p1, mid_line.p2) > 0.05 * max(h, w):
        return None

    return BlueFeatures(
        lines_sorted_by_x=lines,
        circle_center=(circ_cx, circ_cy),
        circle_radius=circ_r,
        dot_center=(dot_cx, dot_cy),
    )


def _perp_distance_point_to_segment(p, a, b) -> float:
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    denom = np.linalg.norm(ab)
    if denom < 1e-6:
        return float(np.linalg.norm(p - a))
    return float(abs(np.cross(ab, p - a)) / denom)


# ---------------------------------------------------------------------------
# Homography construction & I/O
# ---------------------------------------------------------------------------

def world_points_for_features() -> List[Tuple[float, float]]:
    """World-frame (mm) coordinates that correspond, in order, to the image
    points returned by :func:`image_points_from_features`:

        0: left line, top endpoint         (LEFT_LINE_X,   -HALF_WIDTH)
        1: left line, bottom endpoint      (LEFT_LINE_X,   +HALF_WIDTH)
        2: center line, top endpoint       (0,             -HALF_WIDTH)
        3: center line, bottom endpoint    (0,             +HALF_WIDTH)
        4: right line, top endpoint        (RIGHT_LINE_X,  -HALF_WIDTH)
        5: right line, bottom endpoint     (RIGHT_LINE_X,  +HALF_WIDTH)
        6: centre dot                      (0, 0)

    "top"/"bottom" refer to the post-rotation camera image: the endpoint with
    the smaller pixel-y is called "top" and is assigned world y = -HALF_WIDTH.
    """
    hw = tc.HALF_WIDTH_MM
    return [
        (tc.LEFT_LINE_X_MM,  -hw),
        (tc.LEFT_LINE_X_MM,  +hw),
        (tc.CENTER_LINE_X_MM, -hw),
        (tc.CENTER_LINE_X_MM, +hw),
        (tc.RIGHT_LINE_X_MM, -hw),
        (tc.RIGHT_LINE_X_MM, +hw),
        (0.0, 0.0),
    ]


def image_points_from_features(feat: BlueFeatures) -> List[Tuple[float, float]]:
    """Return image-pixel points in the same order as world_points_for_features."""
    left, mid, right = feat.lines_sorted_by_x
    return [
        left.p1, left.p2,
        mid.p1,  mid.p2,
        right.p1, right.p2,
        feat.dot_center,
    ]


def compute_homography(feat: BlueFeatures,
                       flip_x: bool = False) -> Tuple[np.ndarray, float]:
    """
    Build the 3x3 homography H such that for a pixel (u,v):
        [X, Y, W] = H @ [u, v, 1],   with world (mm) = (X/W, Y/W).

    Parameters
    ----------
    flip_x : if True, swap the mapping so the leftmost image line is at +x
             instead of -x. Used when the robot side of the table happens to
             be on the left of the rotated camera image.

    Returns
    -------
    (H, reprojection_error_mm): the homography and the mean reprojection
    residual of the 7 calibration correspondences, in mm. A well-formed
    calibration should yield an error well under ~5 mm.
    """
    img_pts = np.asarray(image_points_from_features(feat), dtype=np.float64)
    world_pts = np.asarray(world_points_for_features(), dtype=np.float64)

    if flip_x:
        world_pts = world_pts.copy()
        world_pts[:, 0] *= -1.0

    H, _inliers = cv.findHomography(img_pts, world_pts, method=cv.RANSAC,
                                    ransacReprojThreshold=3.0)
    if H is None:
        raise RuntimeError("cv.findHomography failed")

    # Residual: map image pts through H, measure mm distance to world_pts.
    mapped = cv.perspectiveTransform(img_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    err_mm = float(np.mean(np.linalg.norm(mapped - world_pts, axis=1)))
    return H, err_mm


def save_calibration(path: str, H: np.ndarray, frame_size_hw: Tuple[int, int],
                     err_mm: float, flip_x: bool) -> None:
    np.savez(path,
             H=H,
             frame_size_hw=np.asarray(frame_size_hw, dtype=np.int32),
             err_mm=np.float64(err_mm),
             flip_x=np.bool_(flip_x))


def load_calibration(path: str = None) -> dict:
    """Load calibration bundle. Raises FileNotFoundError with a helpful hint."""
    path = path or tc.CALIBRATION_FILE
    if not os.path.isabs(path):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No calibration file at {path}. "
            f"Run calibration.py first (it writes {tc.CALIBRATION_FILE})."
        )
    data = np.load(path, allow_pickle=False)
    return {
        "H": data["H"].astype(np.float64),
        "frame_size_hw": tuple(int(v) for v in data["frame_size_hw"]),
        "err_mm": float(data["err_mm"]),
        "flip_x": bool(data["flip_x"]),
    }


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def pixel_to_mm(H: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    """Project a single image pixel through H into world-mm coordinates."""
    pt = np.array([[[float(u), float(v)]]], dtype=np.float64)
    out = cv.perspectiveTransform(pt, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def pixels_to_mm(H: np.ndarray, pts_uv: np.ndarray) -> np.ndarray:
    """Vectorised version. pts_uv shape (N, 2) -> returns (N, 2) in mm."""
    pts = pts_uv.reshape(-1, 1, 2).astype(np.float64)
    return cv.perspectiveTransform(pts, H).reshape(-1, 2)


def warp_to_topdown(frame: np.ndarray, H_pixel_to_mm: np.ndarray,
                    out_w_px: int = tc.WARPED_W_PX,
                    out_h_px: int = tc.WARPED_H_PX,
                    px_per_mm: float = tc.PX_PER_MM) -> np.ndarray:
    """
    Warp a camera frame into a top-down view where each pixel corresponds to
    1/px_per_mm mm and the table occupies the full output canvas.

    The output image origin (0,0) is the *top-left corner* of the table rink
    (world coord (-HALF_LENGTH, -HALF_WIDTH)). The centre dot appears at
    (out_w_px/2, out_h_px/2).
    """
    # H_pixel_to_mm maps pixel -> mm (world, centred at dot).
    # We need pixel -> output-pixel. Chain with a 2D affine that converts mm
    # to the output pixel grid.
    tx = tc.HALF_LENGTH_MM * px_per_mm
    ty = tc.HALF_WIDTH_MM * px_per_mm
    mm_to_out = np.array([
        [px_per_mm, 0.0, tx],
        [0.0, px_per_mm, ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    H_total = mm_to_out @ H_pixel_to_mm
    return cv.warpPerspective(frame, H_total, (out_w_px, out_h_px))


# ---------------------------------------------------------------------------
# Geometric helpers used by goal logic
# ---------------------------------------------------------------------------

def point_in_rect(x: float, y: float,
                  x_min: float, x_max: float, y_min: float, y_max: float) -> bool:
    return (x_min <= x <= x_max) and (y_min <= y <= y_max)


def segment_hits_rect(x0: float, y0: float, x1: float, y1: float,
                      x_min: float, x_max: float,
                      y_min: float, y_max: float) -> bool:
    """True iff the closed line segment (x0,y0)-(x1,y1) intersects the axis-
    aligned rectangle [x_min,x_max] x [y_min,y_max]. Used for trajectory
    extrapolation goal checks."""
    if point_in_rect(x0, y0, x_min, x_max, y_min, y_max):
        return True
    if point_in_rect(x1, y1, x_min, x_max, y_min, y_max):
        return True
    # Liang-Barsky clipping test.
    dx, dy = x1 - x0, y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0 - x_min, x_max - x0, y0 - y_min, y_max - y0]
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return False
        else:
            t = qi / pi
            if pi < 0:
                if t > u2:
                    return False
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return False
                if t < u2:
                    u2 = t
    return u1 <= u2
