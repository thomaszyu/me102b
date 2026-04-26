"""
Live test for CV-based cable slack detection. No motors needed.

Shows the camera feed with slack detection overlay:
  - Cyan annular region around the mallet (detected via yellow blob)
  - Green line = taut cable, Red line = slack cable
  - Sag values and SLACK labels

Bottom half shows the red HSV mask for tuning cable detection.

Press 'q' to quit. Press 't' to print current slack/sag to terminal.

Usage:
    python test_slack_cv.py
"""

import sys
import os
import cv2
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'vision_code'))

from cable_slack_cv import CableSlackDetector
from config import CORNERS


def main():
    from vision import VisionSystem

    vis = VisionSystem()
    vis.start(show_display=False)
    print("Waiting for camera...")
    time.sleep(1.5)

    detector = CableSlackDetector(vis, sag_threshold_px=20.0)

    print("Live slack detection. Press 'q' to quit, 't' to print values.")
    print(f"  inner_r={detector.inner_r}px  outer_r={detector.outer_r}px")
    print(f"  sag_thresh={detector.sag_thresh}px  min_red_frac={detector.min_red_frac}")

    slack = [False] * 4
    sag = [0.0] * 4

    while True:
        frame = vis.frame
        if frame is None:
            time.sleep(0.05)
            continue

        # Run detection (finds mallet internally via yellow blob)
        slack, sag = detector.detect()

        # Draw debug overlay
        out = detector.debug_frame()
        if out is None:
            out = frame.copy()

        # Add status text at top
        for i in range(4):
            status = "SLACK" if slack[i] else "taut"
            color = (0, 0, 255) if slack[i] else (0, 255, 0)
            text = f"Cable {i+1}: {status}  sag={sag[i]:.1f}px"
            cv2.putText(out, text, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        # Also show the red mask for HSV tuning
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = (
            cv2.inRange(hsv, detector.red_lower1, detector.red_upper1) |
            cv2.inRange(hsv, detector.red_lower2, detector.red_upper2)
        )
        red_vis = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)

        # Stack: main view on top, red mask on bottom
        h1, w1 = out.shape[:2]
        h2, w2 = red_vis.shape[:2]
        if w1 != w2:
            red_vis = cv2.resize(red_vis, (w1, h1))
        combined = np.vstack([out, red_vis])

        cv2.imshow("Cable Slack Detection (top: overlay, bottom: red mask)", combined)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            print()
            for i in range(4):
                status = "SLACK" if slack[i] else "taut"
                print(f"  Cable {i+1}: {status}  sag={sag[i]:.1f}px")

    cv2.destroyAllWindows()
    vis.stop()
    print("Done.")


if __name__ == "__main__":
    main()
