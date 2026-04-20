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
last_x, last_y = 0, 0
real_x, real_y = 0.0, 0.0
last_real_x, last_real_y = 0.0, 0.0

robot_goal_left_x = 950
robot_goal_right_x = 1200
robot_goal_top_y = 40
robot_goal_bottom_y = 150

player_goal_left_x = 10
player_goal_right_x = 150
player_goal_top_y = 300
player_goal_bottom_y = 600

# rect verification
calibration_clicks = []
H_matrix = None
H_display = None

# cartesion coords bottom-Left is 0,0, y goes up
world_pts = np.array([
    [158, 480],  # Top-Left
    [704, 480],  # Top-Right
    [704,   0],  # Bottom-Right
    [158,   0],  # Bottom-Left
], dtype="float32")

# same cartesian coords as above, used to display rectified image under transform
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
                print("Homography Matrices Calculated! Bottom-Left Origin Set.")


def get_real_world_coords(px, py, H):
    if H is None:
        return 0.0, 0.0
    pixel_vector = np.array([[[px, py]]], dtype="float32")
    world_vector = cv.perspectiveTransform(pixel_vector, H)
    return world_vector[0][0][0], world_vector[0][0][1]


def detectPuck(puck_mask):
    global frame
    xf, yf = 0, 0
    roi_mask = np.zeros_like(puck_mask)
    roi_mask[45:670, 10:1265] = puck_mask[45:670, 10:1265]
    contours, _ = cv.findContours(
        roi_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        if cv.contourArea(largest_contour) > 50:
            M = cv.moments(largest_contour)
            if M["m00"] > 0:
                xf = int(M["m10"] / M["m00"])
                yf = int(M["m01"] / M["m00"])
                cv.circle(frame, (xf, yf), 30, (255, 0, 0), 5)
    return xf, yf


vid = cv.VideoCapture(0, cv.CAP_V4L2)
vid.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
vid.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
vid.set(cv.CAP_PROP_FPS, 100)
vid.set(cv.CAP_PROP_BUFFERSIZE, 1)

lower_bound1 = np.array([40, 100, 40])
upper_bound1 = np.array([80, 255, 255])
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

print("Click 4 points CLOCKWISE to calibrate: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left.")
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

    mask = cv.inRange(hsv, lower_bound1, upper_bound1)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    green = cv.bitwise_and(frame, frame, mask=mask)

    if current_state == "SEARCHING":
        xf, yf = detectPuck(mask)
        if xf != 0 and yf != 0:
            frames_visible += 1
            if frames_visible > 3:
                current_state = "TRACKING"
                puck_in_play = True
                frames_lost = 0
        else:
            frames_visible = 0

    elif current_state == "TRACKING":
        xf, yf = detectPuck(mask)

        if xf != 0 and yf != 0:
            last_x, last_y = xf, yf
            x3, y3 = x2, y2
            x2, y2 = x1, y1
            x1, y1 = xf, yf
            frames_lost = 0

            if H_matrix is not None:
                last_real_x, last_real_y = real_x, real_y
                real_x, real_y = get_real_world_coords(xf, yf, H_matrix)
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
                    f"ROBOT scores! Robot: {robot_score} | Player: {player_score}")

            elif (player_goal_left_x <= last_x <= player_goal_right_x) or (player_goal_left_x <= predicted_x <= player_goal_right_x):
                if (last_y >= player_goal_top_y or predicted_y >= player_goal_top_y):
                    player_score += 1
                    print(
                        f"PLAYER scores! Robot: {robot_score} | Player: {player_score}")
            else:
                print(f"Puck lost at ({last_x}, {last_y}) - No goal detected.")

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
        distance_mm = math.hypot(real_x - last_real_x, real_y - last_real_y)
        speed = distance_mm / deltaT
        cv.putText(frame, f"Speed: {speed:.2f} mm/s", (50, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv.putText(frame, f"RL Puck (mm): ({int(real_x)}, {int(real_y)})", (
            50, 130), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        pixel_speed = abs(x1 - x2) + \
            abs(y1 - y2) if current_state == "TRACKING" else 0
        cv.putText(frame, f"Speed (px/frame): {pixel_speed:.2f}",
                   (50, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv.putText(frame, f"Puck (px): ({x1}, {y1})", (50, 130),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if len(calibration_clicks) < 4:
        cv.putText(frame, f"Calibration Clicks: {len(calibration_clicks)}/4",
                   (50, 160), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

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

    # Draw calibration clicks
    for pt in calibration_clicks:
        cv.circle(frame, tuple(pt), 5, (0, 165, 255), -1)

    cv.imshow('frame', frame)
    cv.imshow('green mask', green)

    # Top-down verification view
    if H_display is not None:
        warped_frame = cv.warpPerspective(frame, H_display, (862, 480))

        cv.circle(warped_frame, (15, 465), 10, (0, 0, 255), -1)
        cv.putText(warped_frame, "(0,0) RL Origin", (30, 470),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv.imshow('Top-Down Verification', warped_frame)

    cv.setMouseCallback('frame', onMouse)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()
