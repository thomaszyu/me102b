"""
CV-based cable slack detection for cable robot.

Detects slack by checking whether each red cable follows the expected
straight-line path from the mallet to its motor corner. A taut cable
is a straight line; a slack cable sags away from it.

Detection uses an annular region around the mallet — inner radius
excludes the puck/mallet clutter, outer radius captures enough cable.

The mallet is found directly in the camera frame (yellow blob detection,
same as VisionSystem), so no coordinate transforms are needed for
the annulus center.

Usage:
    from cable_slack_cv import CableSlackDetector

    detector = CableSlackDetector(vision_system)
    slack, sag = detector.detect()
    # slack = [False, True, False, False]  — cable 2 is slack
    # sag   = [1.2, 9.5, 0.8, 1.1]        — perpendicular pixel deviation
"""

import cv2
import numpy as np
from config import CORNERS


class CableSlackDetector:
    """
    Detects cable slack from the camera feed.

    Finds the mallet directly in the raw frame (yellow blob), then
    samples red pixels in a narrow strip along each expected cable line
    within an annular region around the mallet.
    """

    def __init__(self, vision_system,
                 inner_radius_px=45,
                 outer_radius_px=130,
                 strip_half_width_px=15,
                 sag_threshold_px=20.0,
                 min_red_fraction=0.10):
        """
        vision_system:       VisionSystem with .frame and .H_matrix
        inner_radius_px:     inner annulus radius (exclude mallet/puck)
        outer_radius_px:     outer annulus radius
        strip_half_width_px: half-width of perpendicular sampling strip
        sag_threshold_px:    mean offset above this = slack
        min_red_fraction:    red pixel density below this = slack
        """
        self.vision = vision_system
        self.inner_r = inner_radius_px
        self.outer_r = outer_radius_px
        self.strip_hw = strip_half_width_px
        self.sag_thresh = sag_threshold_px
        self.min_red_frac = min_red_fraction

        # Red cable HSV ranges (red wraps around 0/180 in OpenCV HSV)
        self.red_lower1 = np.array([0, 70, 50])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 70, 50])
        self.red_upper2 = np.array([180, 255, 255])

        # Mallet HSV range (same as vision.py)
        self._mallet_lower = np.array([12, 100, 70])
        self._mallet_upper = np.array([32, 255, 200])
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Cache corner pixel positions (computed once from H_matrix)
        self._corner_pxs = None

    def _detect_mallet_pixel(self, frame):
        """
        Find the mallet in the raw frame using yellow HSV detection.
        Same method as VisionSystem — guaranteed to match.

        Returns (np.array([px_x, px_y]), found_bool)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._mallet_lower, self._mallet_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 50:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    return np.array([cx, cy]), True
        return np.array([0.0, 0.0]), False

    def _get_corner_pixels(self):
        """Get corner positions in pixel space (cached). Uses inv(H_matrix)."""
        if self._corner_pxs is not None:
            return self._corner_pxs
        if self.vision.H_matrix is None:
            return None
        H_inv = np.linalg.inv(self.vision.H_matrix)
        pxs = []
        for c in CORNERS:
            pt = np.array([[[float(c[0]), float(c[1])]]], dtype='float32')
            px = cv2.perspectiveTransform(pt, H_inv)
            pxs.append(px[0][0])
        self._corner_pxs = pxs
        return self._corner_pxs

    def detect(self, _mallet_world_xy=None):
        """
        Detect slack in each of the 4 cables.

        The mallet_world_xy argument is accepted for API compatibility
        but ignored — the mallet is detected directly in the frame.

        Returns:
            slack: list of 4 bools — True if that cable is slack
            sag:   list of 4 floats — mean perpendicular deviation (pixels)
        """
        frame = self.vision.frame
        if frame is None:
            return [False] * 4, [0.0] * 4

        corner_pxs = self._get_corner_pixels()
        if corner_pxs is None:
            return [False] * 4, [0.0] * 4

        # Find mallet directly in the frame (yellow blob)
        mallet_px, found = self._detect_mallet_pixel(frame)
        if not found:
            return [False] * 4, [0.0] * 4

        mx, my = int(mallet_px[0]), int(mallet_px[1])
        fh, fw = frame.shape[:2]

        # Crop ROI around mallet (faster than processing full frame)
        pad = self.outer_r + 10
        x1 = max(0, mx - pad)
        y1 = max(0, my - pad)
        x2 = min(fw, mx + pad)
        y2 = min(fh, my + pad)
        if x2 - x1 < 20 or y2 - y1 < 20:
            return [False] * 4, [0.0] * 4

        roi = frame[y1:y2, x1:x2]
        rh, rw = roi.shape[:2]

        # Mallet center in ROI coordinates
        cx = mx - x1
        cy = my - y1

        # HSV threshold for red cables in the ROI
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_mask = (
            cv2.inRange(hsv_roi, self.red_lower1, self.red_upper1) |
            cv2.inRange(hsv_roi, self.red_lower2, self.red_upper2)
        )

        # Annular mask (exclude mallet center, limit outer extent)
        Y, X = np.ogrid[:rh, :rw]
        dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
        annular = (dist_sq >= self.inner_r ** 2) & (dist_sq <= self.outer_r ** 2)
        red_annular = red_mask.astype(bool) & annular

        slack = []
        sag_values = []

        for corner_px in corner_pxs:
            # Direction from mallet to this corner (pixel space)
            direction = corner_px - mallet_px
            dist = np.linalg.norm(direction)
            if dist < 1.0:
                slack.append(False)
                sag_values.append(0.0)
                continue

            unit_dir = direction / dist
            perp = np.array([-unit_dir[1], unit_dir[0]])  # perpendicular

            # Build vectorized sample grid:
            #   r_values = distances along cable direction (inner→outer)
            #   o_values = offsets perpendicular to cable
            r_values = np.arange(self.inner_r, self.outer_r, 2, dtype=float)
            o_values = np.arange(-self.strip_hw, self.strip_hw + 1, dtype=float)
            R, O = np.meshgrid(r_values, o_values, indexing='ij')

            # Pixel coordinates of every sample point (in ROI space)
            sample_x = cx + unit_dir[0] * R + perp[0] * O
            sample_y = cy + unit_dir[1] * R + perp[1] * O

            # Bounds check
            in_bounds = (
                (sample_x >= 0) & (sample_x < rw) &
                (sample_y >= 0) & (sample_y < rh)
            )
            # Clip for safe indexing (out-of-bounds locations masked by in_bounds)
            sx = np.clip(sample_x.astype(int), 0, rw - 1)
            sy = np.clip(sample_y.astype(int), 0, rh - 1)

            # Sample red mask at all points
            is_red = red_annular[sy, sx] & in_bounds

            n_valid = np.sum(in_bounds)
            n_red = np.sum(is_red)

            if n_red == 0 or n_valid == 0:
                # No red pixels at all — cable not visible (very slack or occluded)
                slack.append(True)
                sag_values.append(float(self.strip_hw))
                continue

            # Mean absolute perpendicular offset of the red pixels
            mean_sag = float(np.mean(np.abs(O[is_red])))
            red_fraction = n_red / n_valid

            is_slack = mean_sag > self.sag_thresh or red_fraction < self.min_red_frac
            slack.append(is_slack)
            sag_values.append(mean_sag)

        return slack, sag_values

    def debug_frame(self, _mallet_world_xy=None):
        """
        Return a copy of the current frame with detection overlays:
          - Annular region outline (cyan)
          - Expected cable lines (green=taut, red=slack)
          - Sag values and labels

        mallet_world_xy is ignored — mallet found directly in frame.
        """
        frame = self.vision.frame
        if frame is None:
            return None

        corner_pxs = self._get_corner_pixels()
        if corner_pxs is None:
            return frame.copy()

        mallet_px, found = self._detect_mallet_pixel(frame)
        if not found:
            out = frame.copy()
            cv2.putText(out, "Mallet not found (yellow)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return out

        out = frame.copy()
        mx, my = int(mallet_px[0]), int(mallet_px[1])

        # Draw annular region
        cv2.circle(out, (mx, my), self.inner_r, (255, 255, 0), 1)
        cv2.circle(out, (mx, my), self.outer_r, (255, 255, 0), 1)
        cv2.drawMarker(out, (mx, my), (255, 255, 0), cv2.MARKER_CROSS, 10, 1)

        # Detect and annotate
        slack, sag = self.detect()

        for i, corner_px in enumerate(corner_pxs):
            color = (0, 0, 255) if slack[i] else (0, 255, 0)
            label = f"C{i+1} sag={sag[i]:.1f}{'  SLACK' if slack[i] else ''}"

            # Draw expected cable line (inner_r to outer_r from mallet)
            direction = corner_px - mallet_px
            dist = np.linalg.norm(direction)
            if dist < 1:
                continue
            unit_dir = direction / dist

            p1 = mallet_px + unit_dir * self.inner_r
            p2 = mallet_px + unit_dir * self.outer_r
            cv2.line(out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)

            # Label
            label_pos = mallet_px + unit_dir * (self.outer_r + 15)
            cv2.putText(out, label, (int(label_pos[0]), int(label_pos[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return out
