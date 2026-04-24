import cv2 as cv
import numpy as np
import time
import math

tn = time.time()
tElapsed = 0

prevCommandX, prevCommandY = 101, 360
x1, x2, x3, y1, y2, y3 = 0, 0, 0, 0, 0, 0

current_state = "SEARCHING"
robot_score = 0
player_score = 0
frames_visible = 0
frames_lost = 0
evaluate_timer = 0
puck_in_play = False

# real world coords
last_x, last_y = 0, 0
real_x, real_y = 0.0, 0.0
last_real_x, last_real_y = 0.0, 0.0

# mallet real world corrds
real_mallet_x, real_mallet_y = 0.0, 0.0
last_real_mallet_x, last_real_mallet_y = 0.0, 0.0

# fixed goal constraints
robot_goal_left_x = 950
robot_goal_right_x = 1200
robot_goal_top_y = 40
robot_goal_bottom_y = 150

player_goal_left_x = 10
player_goal_right_x = 150
player_goal_top_y = 300
player_goal_bottom_y = 600

CAMERA_HEIGHT_MM = 305.0   # 30.5 cm
MALLET_HEIGHT_MM = 22.175

# rect verification
calibration_clicks = []
H_matrix = None
H_display = None

# Cartesian: Center is 0,0. Y goes UP , table is 862x480. center X is 431, Center Y is 240.
world_pts = np.array([
    [-273,  240],  # Top-Left (158, 480)
    [273,  240],  # Top-Right (704, 480)
    [273, -240],  # Bottom-Right (704, 0)
    [-273, -240],  # Bottom-Left (158, 0)
], dtype="float32")

# Display coordinates
display_pts = np.array([
    [158,   0],
    [704,   0],
    [704, 480],
    [158, 480],
], dtype="float32")


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
    rotated_mat = cv.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def onMouse(event, x, y, flags, param):
    global calibration_clicks, H_matrix, H_display
    if event == cv.EVENT_LBUTTONDOWN:
        bgrPixel = frame[y, x]
        print(f"Clicked x = {x}, y = {y}, BGR = {bgrPixel}")

        if len(calibration_clicks) < 4:
            calibration_clicks.append([x, y])
            print(
                f"Recorded calibration point {len(calibration_clicks)}/4: ({x}, {y})")

            if len(calibration_clicks) == 4:
                pixel_pts = np.array(calibration_clicks, dtype="float32")
                H_matrix = cv.getPerspectiveTransform(pixel_pts, world_pts)
                H_display = cv.getPerspectiveTransform(pixel_pts, display_pts)
                print("H matrix done. Origin Set.")


def get_real_world_coords(px, py, H):
    if H is None:
        return 0.0, 0.0
    pixel_vector = np.array([[[px, py]]], dtype="float32")
    world_vector = cv.perspectiveTransform(pixel_vector, H)
    return world_vector[0][0][0], world_vector[0][0][1]


def detect_object(mask, draw_color=(0, 255, 0), target_frame=None):
    if target_frame is None:
        global frame
        target_frame = frame

    xf, yf = 0, 0
    contours, _ = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        if cv.contourArea(largest_contour) > 50:
            M = cv.moments(largest_contour)
            if M["m00"] > 0:
                xf = int(M["m10"] / M["m00"])
                yf = int(M["m01"] / M["m00"])
                cv.circle(target_frame, (xf, yf), 30, draw_color, 5)
    return xf, yf


def correct_parallax_error(tracked_x, tracked_y):
    scale_factor = (CAMERA_HEIGHT_MM - MALLET_HEIGHT_MM) / CAMERA_HEIGHT_MM
    true_base_x = tracked_x * scale_factor
    true_base_y = tracked_y * scale_factor
    return true_base_x, true_base_y


vid = cv.VideoCapture(0)
vid.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
vid.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
vid.set(cv.CAP_PROP_FPS, 100)
vid.set(cv.CAP_PROP_BUFFERSIZE, 1)

# Puck colour green
lower_puck = np.array([40, 100, 40])
upper_puck = np.array([80, 255, 255])

# Robot colour yellow
lower_mallet = np.array([12, 100, 70])
upper_mallet = np.array([32, 255, 200])

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

print("Click CLOCKWISE to calibrate: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left.")
print("Press 'q' to quit.")

while True:
    deltaT = (time.time() - tn) - tElapsed
    tElapsed = time.time() - tn
    fps = 1.0 / deltaT if deltaT > 0 else 0.0

    ret, frame = vid.read()
    if not ret:
        break

    frame = rotate_image(frame, -180.6)

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Puck tracking mask
    mask_puck = cv.inRange(hsv, lower_puck, upper_puck)
    mask_puck = cv.morphologyEx(mask_puck, cv.MORPH_OPEN, kernel)

    # Mallet tracking mask
    mask_mallet_orig = cv.inRange(hsv, lower_mallet, upper_mallet)
    mask_mallet_orig = cv.morphologyEx(mask_mallet_orig, cv.MORPH_OPEN, kernel)

    # mallet in unrectified image
    orig_mallet_xf, orig_mallet_yf = detect_object(
        mask_mallet_orig, draw_color=(0, 165, 255))

    # rectified tracking
    mallet_xf, mallet_yf = 0, 0
    if H_display is not None:
        warped_frame = cv.warpPerspective(frame, H_display, (862, 480))
        warped_hsv = cv.cvtColor(warped_frame, cv.COLOR_BGR2HSV)

        # tracking mallet rectified
        mask_mallet_warped = cv.inRange(warped_hsv, lower_mallet, upper_mallet)
        mask_mallet_warped = cv.morphologyEx(
            mask_mallet_warped, cv.MORPH_OPEN, kernel)

        mallet_xf, mallet_yf = detect_object(
            mask_mallet_warped, draw_color=(0, 165, 255), target_frame=warped_frame)

        if mallet_xf != 0 and mallet_yf != 0:
            last_real_mallet_x, last_real_mallet_y = real_mallet_x, real_mallet_y
            raw_mallet_real_x = mallet_xf - 431.0
            raw_mallet_real_y = 240.0 - mallet_yf
            real_mallet_x, real_mallet_y = correct_parallax_error(
                raw_mallet_real_x, raw_mallet_real_y)

            cv.circle(warped_frame, (431, 240), 10, (0, 0, 255), -1)
            cv.putText(warped_frame, "(0,0)", (445, 235),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv.putText(warped_frame, f"Base: {int(real_mallet_x)}, {int(real_mallet_y)}",
                       (mallet_xf + 35, mallet_yf), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # tracking puck rectified
        mask_puck_warped = cv.inRange(warped_hsv, lower_puck, upper_puck)
        mask_puck_warped = cv.morphologyEx(
            mask_puck_warped, cv.MORPH_OPEN, kernel)

        puck_warped_xf, puck_warped_yf = detect_object(
            mask_puck_warped, draw_color=(0, 255, 0), target_frame=warped_frame)

        if puck_warped_xf != 0 and puck_warped_yf != 0:
            puck_warped_real_x = puck_warped_xf - 431.0
            puck_warped_real_y = 240.0 - puck_warped_yf
            cv.putText(warped_frame, f"Puck: {int(puck_warped_real_x)}, {int(puck_warped_real_y)}",
                       (puck_warped_xf + 35, puck_warped_yf), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv.imshow('Top-Down Verification', warped_frame)

    # finite state machine
    if current_state == "SEARCHING":
        xf, yf = detect_object(
            mask_puck, draw_color=(0, 255, 0))  # Green circle
        if xf != 0 and yf != 0:
            frames_visible += 1
            if frames_visible > 3:
                current_state = "TRACKING"
                puck_in_play = True
                frames_lost = 0
        else:
            frames_visible = 0

    elif current_state == "TRACKING":
        xf, yf = detect_object(mask_puck, draw_color=(0, 255, 0))

        if xf != 0 and yf != 0:
            last_x, last_y = xf, yf
            x3, y3 = x2, y2
            x2, y2 = x1, y1
            x1, y1 = xf, yf
            frames_lost = 0
        else:
            frames_lost += 1
            if frames_lost > 10:
                current_state = "EVALUATE_SCORE"
                evaluate_timer = 0

    elif current_state == "EVALUATE_SCORE":
        if evaluate_timer == 0:
            dx = last_x - x3
            dy = last_y - y3
            predicted_x = last_x + (dx * 3)
            predicted_y = last_y + (dy * 3)

            robot_x_check = (robot_goal_left_x <= last_x <= robot_goal_right_x) or (
                robot_goal_left_x <= predicted_x <= robot_goal_right_x)
            if robot_x_check and (last_y <= robot_goal_bottom_y or predicted_y <= robot_goal_bottom_y):
                robot_score += 1
                print(
                    f"Robot scores! Robot: {robot_score} | Player: {player_score}")

            elif (player_goal_left_x <= last_x <= player_goal_right_x) or (player_goal_left_x <= predicted_x <= player_goal_right_x):
                if (last_y >= player_goal_top_y or predicted_y >= player_goal_top_y):
                    player_score += 1
                    print(
                        f"Player scores! Robot: {robot_score} | Player: {player_score}")
            else:
                print(f"Puck lost at ({last_x}, {last_y})")

        evaluate_timer += 1
        if evaluate_timer > 30:
            current_state = "SEARCHING"
            puck_in_play = False
            frames_visible = 0
            frames_lost = 0
            x1, x2, x3, y1, y2, y3 = 0, 0, 0, 0, 0, 0

    cv.putText(frame, f"STATE: {current_state}", (50, 40),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.putText(frame, f"FPS: {int(fps)}", (50, 70),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if H_matrix is not None and current_state == "TRACKING" and deltaT > 0:
        cv.putText(frame, f"Puck (mm): ({int(real_x)}, {int(real_y)})", (
            50, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(frame, f"Mallet (mm): ({int(real_mallet_x)}, {int(real_mallet_y)})", (
            50, 130), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    if len(calibration_clicks) < 4:
        cv.putText(frame, f"Calibration Clicks: {len(calibration_clicks)}/4",
                   (50, 160), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    else:
        # white box from calibration for rectangular verification
        pts = np.array(calibration_clicks, dtype=np.int32)
        cv.polylines(frame, [pts], True, (255, 255, 255), 2)

    cv.putText(frame, f"Robot Score: {robot_score}", (50, 210),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv.putText(frame, f"Player Score: {player_score}", (50, 250),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    cv.rectangle(frame, (robot_goal_left_x, robot_goal_top_y),
                 (robot_goal_right_x, robot_goal_bottom_y), (0, 255, 0), 2)
    cv.putText(frame, "Robot Goal", (robot_goal_left_x, robot_goal_top_y - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv.rectangle(frame, (player_goal_left_x, player_goal_top_y),
                 (player_goal_right_x, player_goal_bottom_y), (255, 0, 255), 2)
    cv.putText(frame, "Player Goal", (player_goal_left_x, player_goal_top_y - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    for pt in calibration_clicks:
        cv.circle(frame, tuple(pt), 5, (0, 165, 255), -1)

    cv.imshow('frame', frame)

    cv.setMouseCallback('frame', onMouse)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()
