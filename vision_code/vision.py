"""
Vision system for air hockey table.
Tracks mallet (yellow) and puck (green) positions in mm.

Usage:
    Standalone:  python vision.py
    From motor code:
        from vision import VisionSystem
        vis = VisionSystem()
        vis.start()                    # starts camera + processing in background thread
        mallet, puck = vis.get_positions()  # returns latest (x,y) in mm, thread-safe
        vis.stop()
"""

import cv2 as cv
import numpy as np
import time
import json
import os
import threading
from dataclasses import dataclass, field

CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")


#################
## CONFIG      ##
#################

CAMERA_HEIGHT_MM = 305.0
MALLET_HEIGHT_MM = 22.175

# Winning score — first side to reach this wins. Tweak as desired.
WIN_SCORE = 7

# Calibration target coords
world_pts = np.array([
    [-273,  240],
    [ 273,  240],
    [ 273, -240],
    [-273, -240],
], dtype="float32")

display_pts = np.array([
    [158,   0],
    [704,   0],
    [704, 480],
    [158, 480],
], dtype="float32")

# HSV ranges
lower_puck = np.array([35, 50, 30])
upper_puck = np.array([85, 255, 255])

lower_mallet = np.array([12, 100, 70])
upper_mallet = np.array([32, 255, 200])

# Hard offset for mallet detection (mm) — corrects for marker-to-center offset
MALLET_OFFSET_X = -20.0
MALLET_OFFSET_Y = 23.0

# Goal constraints (raw pixel coords)
robot_goal_left_x = 950
robot_goal_right_x = 1200
robot_goal_top_y = 40
robot_goal_bottom_y = 150

player_goal_left_x = 10
player_goal_right_x = 150
player_goal_top_y = 300
player_goal_bottom_y = 600

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))


#################
## HELPERS     ##
#################

def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)
    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]
    return cv.warpAffine(mat, rotation_mat, (bound_w, bound_h))


def detect_object(mask, draw_color=(0, 255, 0), target_frame=None):
    xf, yf = 0, 0
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        if cv.contourArea(largest_contour) > 50:
            M = cv.moments(largest_contour)
            if M["m00"] > 0:
                xf = int(M["m10"] / M["m00"])
                yf = int(M["m01"] / M["m00"])
                if target_frame is not None:
                    cv.circle(target_frame, (xf, yf), 30, draw_color, 5)
    return xf, yf


def get_real_world_coords(px, py, H):
    if H is None:
        return 0.0, 0.0
    pixel_vector = np.array([[[px, py]]], dtype="float32")
    world_vector = cv.perspectiveTransform(pixel_vector, H)
    return world_vector[0][0][0], world_vector[0][0][1]


def correct_parallax_error(tracked_x, tracked_y):
    scale_factor = (CAMERA_HEIGHT_MM - MALLET_HEIGHT_MM) / CAMERA_HEIGHT_MM
    true_base_x = tracked_x * scale_factor
    true_base_y = tracked_y * scale_factor
    return true_base_x, true_base_y


#######################
## POSITION OUTPUT   ##
#######################

@dataclass
class TrackedPositions:
    """Thread-safe container for latest tracked positions (mm)."""
    mallet_x: float = 0.0
    mallet_y: float = 0.0
    puck_x: float = 0.0
    puck_y: float = 0.0
    mallet_valid: bool = False
    puck_valid: bool = False
    timestamp: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update_mallet(self, x, y):
        with self._lock:
            self.mallet_x = x
            self.mallet_y = y
            self.mallet_valid = True
            self.timestamp = time.time()

    def update_puck(self, x, y):
        with self._lock:
            self.puck_x = x
            self.puck_y = y
            self.puck_valid = True
            self.timestamp = time.time()

    def mark_mallet_lost(self):
        with self._lock:
            self.mallet_valid = False

    def mark_puck_lost(self):
        with self._lock:
            self.puck_valid = False

    def get(self):
        """Returns ((mallet_x, mallet_y, valid), (puck_x, puck_y, valid), timestamp)."""
        with self._lock:
            return (
                (self.mallet_x, self.mallet_y, self.mallet_valid),
                (self.puck_x, self.puck_y, self.puck_valid),
                self.timestamp,
            )


#######################
## VISION SYSTEM     ##
#######################

class VisionSystem:
    """
    Runs the camera + CV pipeline. Can run standalone (with display)
    or as a background thread exposing positions to motor control.
    """

    def __init__(self, calibration_file=None):
        self.positions = TrackedPositions()
        self._thread = None
        self._stop_event = threading.Event()

        # Calibration state
        self.calibration_clicks = []
        self.H_matrix = None
        self.H_display = None
        self.frame = None  # current frame, needed by onMouse

        # Try to load saved calibration
        self._cal_file = calibration_file or CALIBRATION_FILE
        self._load_calibration()

        # FSM state
        self.current_state = "SEARCHING"
        self.robot_score = 0
        self.player_score = 0
        # Winner once either side reaches WIN_SCORE: "robot" | "player" | None
        self.winner = None
        self.frames_visible = 0
        self.frames_lost = 0
        self.evaluate_timer = 0
        self.puck_in_play = False
        self.x1, self.x2, self.x3 = 0, 0, 0
        self.y1, self.y2, self.y3 = 0, 0, 0
        self.last_x, self.last_y = 0, 0

        # Tracked world coords (internal, for display)
        self.real_x, self.real_y = 0.0, 0.0
        self.real_mallet_x, self.real_mallet_y = 0.0, 0.0
        self._mallet_lost_streak = 0

    def get_positions(self):
        """Thread-safe read of latest positions.
        Returns ((mallet_x, mallet_y, valid), (puck_x, puck_y, valid), timestamp)
        """
        return self.positions.get()

    def _check_winner(self):
        """Latch a winner once either side reaches WIN_SCORE."""
        if self.winner is not None:
            return  # already decided — don't overwrite
        if self.robot_score >= WIN_SCORE:
            self.winner = "robot"
            print(f"=== ROBOT WINS! Final {self.robot_score}-{self.player_score} ===")
        elif self.player_score >= WIN_SCORE:
            self.winner = "player"
            print(f"=== PLAYER WINS! Final {self.robot_score}-{self.player_score} ===")

    def reset_scores(self):
        """Reset scores and winner state. Call this between games."""
        self.robot_score = 0
        self.player_score = 0
        self.winner = None

    def start(self, show_display=False):
        """Start vision in a background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(show_display,),
            daemon=True
        )
        self._thread.start()

    def stop(self):
        """Stop the background thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _save_calibration(self):
        """Save calibration clicks to JSON so they persist across runs."""
        with open(self._cal_file, 'w') as f:
            json.dump(self.calibration_clicks, f)
        print(f"Calibration saved to {self._cal_file}")

    def _load_calibration(self):
        """Load saved calibration if it exists."""
        if not os.path.exists(self._cal_file):
            return
        try:
            with open(self._cal_file, 'r') as f:
                clicks = json.load(f)
            if len(clicks) == 4:
                self.calibration_clicks = clicks
                pixel_pts = np.array(clicks, dtype="float32")
                self.H_matrix = cv.getPerspectiveTransform(pixel_pts, world_pts)
                self.H_display = cv.getPerspectiveTransform(pixel_pts, display_pts)
                print(f"Loaded calibration from {self._cal_file}: {clicks}")
            else:
                print(f"Invalid calibration file (need 4 points, got {len(clicks)})")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to load calibration: {e}")

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            if self.frame is not None:
                bgrPixel = self.frame[y, x]
                print(f"Clicked x = {x}, y = {y}, BGR = {bgrPixel}")

            if len(self.calibration_clicks) < 4:
                self.calibration_clicks.append([x, y])
                print(f"Recorded calibration point {len(self.calibration_clicks)}/4: ({x}, {y})")

                if len(self.calibration_clicks) == 4:
                    pixel_pts = np.array(self.calibration_clicks, dtype="float32")
                    self.H_matrix = cv.getPerspectiveTransform(pixel_pts, world_pts)
                    self.H_display = cv.getPerspectiveTransform(pixel_pts, display_pts)
                    self._save_calibration()
                    print("H matrix done. Origin Set.")

    def _update_fsm(self, xf, yf):
        if self.current_state == "SEARCHING":
            if xf != 0 and yf != 0:
                self.frames_visible += 1
                if self.frames_visible > 3:
                    self.current_state = "TRACKING"
                    self.puck_in_play = True
                    self.frames_lost = 0
            else:
                self.frames_visible = 0

        elif self.current_state == "TRACKING":
            if xf != 0 and yf != 0:
                self.last_x, self.last_y = xf, yf
                self.x3, self.y3 = self.x2, self.y2
                self.x2, self.y2 = self.x1, self.y1
                self.x1, self.y1 = xf, yf
                self.frames_lost = 0
            else:
                self.frames_lost += 1
                if self.frames_lost > 10:
                    self.current_state = "EVALUATE_SCORE"
                    self.evaluate_timer = 0

        elif self.current_state == "EVALUATE_SCORE":
            if self.evaluate_timer == 0:
                dx = self.last_x - self.x3
                dy = self.last_y - self.y3
                predicted_x = self.last_x + (dx * 3)
                predicted_y = self.last_y + (dy * 3)

                robot_x_check = (robot_goal_left_x <= self.last_x <= robot_goal_right_x) or \
                                (robot_goal_left_x <= predicted_x <= robot_goal_right_x)
                if robot_x_check and (self.last_y <= robot_goal_bottom_y or predicted_y <= robot_goal_bottom_y):
                    self.robot_score += 1
                    print(f"Robot scores! Robot: {self.robot_score} | Player: {self.player_score}")
                elif (player_goal_left_x <= self.last_x <= player_goal_right_x) or \
                     (player_goal_left_x <= predicted_x <= player_goal_right_x):
                    if self.last_y >= player_goal_top_y or predicted_y >= player_goal_top_y:
                        self.player_score += 1
                        print(f"Player scores! Robot: {self.robot_score} | Player: {self.player_score}")
                else:
                    print(f"Puck lost at ({self.last_x}, {self.last_y})")

            self.evaluate_timer += 1
            if self.evaluate_timer > 30:
                self.current_state = "SEARCHING"
                self.puck_in_play = False
                self.frames_visible = 0
                self.frames_lost = 0
                self.x1, self.x2, self.x3 = 0, 0, 0
                self.y1, self.y2, self.y3 = 0, 0, 0

    def _run_loop(self, show_display=True):
        vid = cv.VideoCapture(0)
        vid.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        vid.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        vid.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        vid.set(cv.CAP_PROP_FPS, 100)
        vid.set(cv.CAP_PROP_BUFFERSIZE, 1)

        tn = time.time()
        tElapsed = 0

        if show_display:
            print("Click CLOCKWISE to calibrate: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left.")
            print("Press 'q' to quit.")

        while not self._stop_event.is_set():
            deltaT = (time.time() - tn) - tElapsed
            tElapsed = time.time() - tn
            fps = 1.0 / deltaT if deltaT > 0 else 0.0

            ret, raw = vid.read()
            if not ret:
                break

            frame = rotate_image(raw, -180.6)
            self.frame = frame  # single atomic assignment after rotation

            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            # Puck mask
            mask_puck = cv.inRange(hsv, lower_puck, upper_puck)
            mask_puck = cv.morphologyEx(mask_puck, cv.MORPH_OPEN, kernel)

            # Mallet mask
            mask_mallet_orig = cv.inRange(hsv, lower_mallet, upper_mallet)
            mask_mallet_orig = cv.morphologyEx(mask_mallet_orig, cv.MORPH_OPEN, kernel)

            # Mallet in unrectified image
            orig_mallet_xf, orig_mallet_yf = detect_object(
                mask_mallet_orig, draw_color=(0, 165, 255),
                target_frame=frame if show_display else None)

            # Rectified tracking
            if self.H_display is not None:
                warped_frame = cv.warpPerspective(frame, self.H_display, (862, 480))
                warped_hsv = cv.cvtColor(warped_frame, cv.COLOR_BGR2HSV)

                # Mallet rectified
                mask_mallet_warped = cv.inRange(warped_hsv, lower_mallet, upper_mallet)
                mask_mallet_warped = cv.morphologyEx(mask_mallet_warped, cv.MORPH_OPEN, kernel)
                mallet_xf, mallet_yf = detect_object(
                    mask_mallet_warped, draw_color=(0, 165, 255),
                    target_frame=warped_frame if show_display else None)

                if mallet_xf != 0 and mallet_yf != 0:
                    raw_mallet_real_x = mallet_xf - 431.0
                    raw_mallet_real_y = 240.0 - mallet_yf
                    mx, my = correct_parallax_error(
                        raw_mallet_real_x, raw_mallet_real_y)
                    new_x = mx + MALLET_OFFSET_X
                    new_y = my + MALLET_OFFSET_Y

                    # Reject jumps — mallet can't teleport
                    MALLET_MAX_JUMP_MM = 60.0  # max plausible movement per frame
                    MALLET_REACQUIRE_FRAMES = 5  # after this many lost frames, accept any detection
                    dx = new_x - self.real_mallet_x
                    dy = new_y - self.real_mallet_y
                    jump = np.sqrt(dx*dx + dy*dy)
                    if (jump < MALLET_MAX_JUMP_MM
                            or not hasattr(self, '_mallet_ever_seen')
                            or self._mallet_lost_streak >= MALLET_REACQUIRE_FRAMES):
                        self.real_mallet_x = new_x
                        self.real_mallet_y = new_y
                        self._mallet_ever_seen = True
                        self._mallet_lost_streak = 0
                        self.positions.update_mallet(self.real_mallet_x, self.real_mallet_y)
                    else:
                        # Jump detected — don't update, treat as lost
                        self._mallet_lost_streak = getattr(self, '_mallet_lost_streak', 0) + 1
                        self.positions.mark_mallet_lost()
                else:
                    self.positions.mark_mallet_lost()

                if show_display:
                    if mallet_xf != 0 and mallet_yf != 0:
                        cv.circle(warped_frame, (431, 240), 10, (0, 0, 255), -1)
                        cv.putText(warped_frame, "(0,0)", (445, 235),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv.putText(warped_frame,
                                   f"Base: {int(self.real_mallet_x)}, {int(self.real_mallet_y)}",
                                   (mallet_xf + 35, mallet_yf),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                # Puck rectified
                mask_puck_warped = cv.inRange(warped_hsv, lower_puck, upper_puck)
                mask_puck_warped = cv.morphologyEx(mask_puck_warped, cv.MORPH_OPEN, kernel)
                puck_warped_xf, puck_warped_yf = detect_object(
                    mask_puck_warped, draw_color=(0, 255, 0),
                    target_frame=warped_frame if show_display else None)

                if puck_warped_xf != 0 and puck_warped_yf != 0:
                    puck_warped_real_x = puck_warped_xf - 431.0
                    puck_warped_real_y = 240.0 - puck_warped_yf
                    self.real_x, self.real_y = puck_warped_real_x, puck_warped_real_y
                    self.positions.update_puck(puck_warped_real_x, puck_warped_real_y)
                else:
                    self.positions.mark_puck_lost()

                if show_display:
                    if puck_warped_xf != 0 and puck_warped_yf != 0:
                        cv.putText(warped_frame,
                                   f"Puck: {int(puck_warped_real_x)}, {int(puck_warped_real_y)}",
                                   (puck_warped_xf + 35, puck_warped_yf),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv.imshow('Top-Down Verification', warped_frame)

            # FSM
            xf, yf = detect_object(mask_puck, draw_color=(0, 255, 0),
                                   target_frame=frame if show_display else None)
            self._update_fsm(xf, yf)

            # Display
            if show_display:
                cv.putText(frame, f"STATE: {self.current_state}", (50, 40),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv.putText(frame, f"FPS: {int(fps)}", (50, 70),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if self.H_matrix is not None and self.current_state == "TRACKING" and deltaT > 0:
                    cv.putText(frame, f"Puck (mm): ({int(self.real_x)}, {int(self.real_y)})",
                               (50, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv.putText(frame, f"Mallet (mm): ({int(self.real_mallet_x)}, {int(self.real_mallet_y)})",
                               (50, 130), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                if len(self.calibration_clicks) < 4:
                    cv.putText(frame, f"Calibration Clicks: {len(self.calibration_clicks)}/4",
                               (50, 160), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    pts = np.array(self.calibration_clicks, dtype=np.int32)
                    cv.polylines(frame, [pts], True, (255, 255, 255), 2)

                cv.putText(frame, f"Robot Score: {self.robot_score}", (50, 210),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv.putText(frame, f"Player Score: {self.player_score}", (50, 250),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

                cv.rectangle(frame, (robot_goal_left_x, robot_goal_top_y),
                             (robot_goal_right_x, robot_goal_bottom_y), (0, 255, 0), 2)
                cv.putText(frame, "Robot Goal", (robot_goal_left_x, robot_goal_top_y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv.rectangle(frame, (player_goal_left_x, player_goal_top_y),
                             (player_goal_right_x, player_goal_bottom_y), (255, 0, 255), 2)
                cv.putText(frame, "Player Goal", (player_goal_left_x, player_goal_top_y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                for pt in self.calibration_clicks:
                    cv.circle(frame, tuple(pt), 5, (0, 165, 255), -1)

                # Winner banner — drawn last so it sits on top of everything.
                if self.winner is not None:
                    if self.winner == "robot":
                        banner_text = "ROBOT WINS!"
                        banner_color = (0, 255, 0)
                    else:
                        banner_text = "PLAYER WINS!"
                        banner_color = (255, 0, 255)

                    h, w = frame.shape[:2]
                    bw, bh = 700, 130
                    bx = (w - bw) // 2
                    by = (h - bh) // 2

                    overlay = frame.copy()
                    cv.rectangle(overlay, (bx, by), (bx + bw, by + bh),
                                 (0, 0, 0), -1)
                    cv.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
                    cv.rectangle(frame, (bx, by), (bx + bw, by + bh),
                                 banner_color, 4)
                    cv.rectangle(frame, (bx + 4, by + 4),
                                 (bx + bw - 4, by + bh - 4),
                                 banner_color, 2)

                    text_size, _ = cv.getTextSize(
                        banner_text, cv.FONT_HERSHEY_SIMPLEX, 2.4, 5)
                    tx = bx + (bw - text_size[0]) // 2
                    ty = by + (bh + text_size[1]) // 2 - 10
                    cv.putText(frame, banner_text, (tx, ty),
                               cv.FONT_HERSHEY_SIMPLEX, 2.4, banner_color, 5)

                    score_text = f"{self.robot_score} - {self.player_score}"
                    sts, _ = cv.getTextSize(
                        score_text, cv.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv.putText(frame, score_text,
                               (bx + (bw - sts[0]) // 2, by + bh - 12),
                               cv.FONT_HERSHEY_SIMPLEX, 0.9,
                               (255, 255, 255), 2)

                cv.imshow('frame', frame)
                cv.setMouseCallback('frame', self._on_mouse)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        vid.release()
        if show_display:
            cv.destroyAllWindows()


#################
## STANDALONE  ##
#################

if __name__ == "__main__":
    vis = VisionSystem()
    vis._run_loop(show_display=True)
