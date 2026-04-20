import cv2 as cv
import numpy as np
import time
import math

tn = time.time()
tElapsed = 0

prevCommandX, prevCommandY = 101, 360

x1, x2, x3, y1, y2, y3 = 0, 0, 0, 0, 0, 0
movingStationary = False
goodVal = False

current_state = "SEARCHING"
robot_score = 0  
player_score = 0 
frames_visible = 0
frames_lost = 0
evaluate_timer = 0
puck_in_play = False
last_x, last_y = 0, 0

robot_goal_left_x = 950     
robot_goal_right_x = 1200
robot_goal_top_y = 100        
robot_goal_bottom_y = 250   

player_goal_left_x = 10     
player_goal_right_x = 150    
player_goal_top_y = 450        
player_goal_bottom_y = 700   

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
    if event == cv.EVENT_LBUTTONDOWN:
        bgrPixel = frame[y, x]
        print(f"Clicked x = {x}, y = {y}, BGR = {bgrPixel}")

def detectPuck(puck_mask):
    global frame
    xf, yf = 0, 0
    
    roi_mask = np.zeros_like(puck_mask)
    roi_mask[45:670, 10:1265] = puck_mask[45:670, 10:1265]

    contours, _ = cv.findContours(roi_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

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

# target BGR colour of puck RED
#target_bgr = np.uint8([[[68, 82, 232]]]) 
# target BGR for green
#target_bgr = np.uint8([[[118, 224, 127]]]) 
target_bgr = np.uint8([[[62, 165, 84]]]) 
hsv_color = cv.cvtColor(target_bgr, cv.COLOR_BGR2HSV)[0][0]

hue, sat, val = hsv_color[0], hsv_color[1], hsv_color[2]

# window tried (+/- 10 for Hue, +/- 60 for Sat/Val)
lower_bound1 = np.array([max(0, hue - 10), max(50, sat - 60), max(50, val - 60)])
upper_bound1 = np.array([min(180, hue + 10), min(255, sat + 60), min(255, val + 60)])

needs_wrap = False
if hue < 10:
    lower_bound2 = np.array([180 - (10 - hue), max(50, sat - 60), max(50, val - 60)])
    upper_bound2 = np.array([180, min(255, sat + 60), min(255, val + 60)])
    needs_wrap = True
elif hue > 170:
    lower_bound2 = np.array([0, max(50, sat - 60), max(50, val - 60)])
    upper_bound2 = np.array([10 - (180 - hue), min(255, sat + 60), min(255, val + 60)])
    needs_wrap = True

print("Press q to quit.")

while True:
    # 1. Universal Operations (Run every frame)
    deltaT = (time.time() - tn) - tElapsed
    tElapsed = time.time() - tn
    fps = 1.0 / deltaT if deltaT > 0 else 0.0

    ret, frame = vid.read()
    if not ret:
        print("Failed to read frame.")
        break

    frame = rotate_image(frame, -180.6)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask1 = cv.inRange(hsv, lower_bound1, upper_bound1)
    if needs_wrap:
        mask2 = cv.inRange(hsv, lower_bound2, upper_bound2)
        mask = cv.bitwise_or(mask1, mask2)
    else:
        mask = mask1
    
    green = cv.bitwise_and(frame, frame, mask=mask)

    if current_state == "SEARCHING":
        xf, yf = detectPuck(mask)
        
        # puck found
        if xf != 0 and yf != 0:
            frames_visible += 1
            if frames_visible > 3:  # 3 frames min of detected puck
                current_state = "TRACKING"
                puck_in_play = True
                frames_lost = 0
        else:
            frames_visible = 0

    elif current_state == "TRACKING":
        xf, yf = detectPuck(mask)
        
        if xf != 0 and yf != 0:
            # save last known position before vanishes
            last_x, last_y = xf, yf 
            
            x3, y3 = x2, y2
            x2, y2 = x1, y1
            x1, y1 = xf, yf 
            frames_lost = 0
        
        else:
            # Puck lost
            frames_lost += 1
            if frames_lost > 10:
                current_state = "EVALUATE_SCORE"
                evaluate_timer = 0

    elif current_state == "EVALUATE_SCORE":
        if evaluate_timer == 0:
            # check if it went into the robot's Goal
            if (robot_goal_top_y <= last_y <= robot_goal_bottom_y) and (robot_goal_left_x <= last_x <= robot_goal_right_x):
                robot_score += 1
                print(f"ROBOT scores! Robot: {robot_score} | Player: {player_score}")
            
            # check if it went into the player's Goal
            elif (player_goal_top_y <= last_y <= player_goal_bottom_y) and (player_goal_left_x <= last_x <= player_goal_right_x):
                player_score += 1
                print(f"PLAYER scores! Robot: {robot_score} | Player: {player_score}")
            
            # left the frame
            else:
                print(f"Puck lost at ({last_x}, {last_y}) - No goal detected.")
        
        evaluate_timer += 1
        
        if evaluate_timer > 30:
            current_state = "SEARCHING"
            puck_in_play = False
            frames_visible = 0
            frames_lost = 0
            x1, x2, x3, y1, y2, y3 = 0, 0, 0, 0, 0, 0
            sddx, sddy = 0, 0

    speed = abs(x1 - x2) + abs(y1 - y2) if current_state == "TRACKING" else 0
    
    cv.putText(frame, f"STATE: {current_state}", (50, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv.putText(frame, f"FPS: {int(fps)}", (50, 110), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.putText(frame, f"Speed: {speed:.2f}", (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.putText(frame, f"Puck: ({x1}, {y1})", (50, 190), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    cv.putText(frame, f"Robot Score: {robot_score}", (50, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.putText(frame, f"Player Score: {player_score}", (50, 280), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    cv.rectangle(frame, (robot_goal_left_x, robot_goal_top_y), (robot_goal_right_x, robot_goal_bottom_y), (0, 255, 0), 2)
    cv.rectangle(frame, (player_goal_left_x, player_goal_top_y), (player_goal_right_x, player_goal_bottom_y), (255, 0, 255), 2)
    
    cv.imshow('frame', frame)
    cv.imshow('green', green)
    cv.setMouseCallback('frame', onMouse)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()