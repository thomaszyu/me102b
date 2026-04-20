"""
Table / world geometry for the air-hockey robot.

Coordinate system (world / table frame):
    origin (0, 0) = center dot on the table
    +x axis      = along the long (862 mm) dimension, toward the ROBOT side
    +y axis      = along the short (480 mm) dimension, toward the bottom
                   of the rotated camera image (rotate_image(..., -180.6))

All distances are millimetres.
"""

TABLE_LENGTH_MM = 862.0
TABLE_WIDTH_MM = 480.0

HALF_LENGTH_MM = TABLE_LENGTH_MM / 2.0
HALF_WIDTH_MM = TABLE_WIDTH_MM / 2.0

# Blue lines running across the width (constant x in world frame).
# 158 mm from each short edge, 6.5 mm wide.
LINE_THICKNESS_MM = 6.5
LINE_OFFSET_FROM_EDGE_MM = 158.0
LEFT_LINE_X_MM = -(HALF_LENGTH_MM - LINE_OFFSET_FROM_EDGE_MM)   # -273.0
RIGHT_LINE_X_MM = +(HALF_LENGTH_MM - LINE_OFFSET_FROM_EDGE_MM)  # +273.0
CENTER_LINE_X_MM = 0.0

# Center circle / dot.
CENTER_CIRCLE_OUTER_DIAMETER_MM = 141.0
CENTER_CIRCLE_RADIUS_MM = CENTER_CIRCLE_OUTER_DIAMETER_MM / 2.0  # 70.5
CENTER_DOT_DIAMETER_MM = 20.5

# Goal regions (rectangles in world-mm). The puck enters a goal when it
# reaches the short rail within the goal-mouth width. These are a bit
# generous on the x axis so trajectory extrapolation near the rail counts.
GOAL_MOUTH_WIDTH_MM = 160.0          # along y
GOAL_DEPTH_MM = 40.0                 # along x, in front of the rail
_GM_HALF = GOAL_MOUTH_WIDTH_MM / 2.0

# Robot goal is at +x rail.
ROBOT_GOAL_X_MIN = HALF_LENGTH_MM - GOAL_DEPTH_MM
ROBOT_GOAL_X_MAX = HALF_LENGTH_MM + 20.0       # small pad past the rail
ROBOT_GOAL_Y_MIN = -_GM_HALF
ROBOT_GOAL_Y_MAX = +_GM_HALF

# Player goal is at -x rail.
PLAYER_GOAL_X_MIN = -HALF_LENGTH_MM - 20.0
PLAYER_GOAL_X_MAX = -HALF_LENGTH_MM + GOAL_DEPTH_MM
PLAYER_GOAL_Y_MIN = -_GM_HALF
PLAYER_GOAL_Y_MAX = +_GM_HALF

# HSV window for the BLUE calibration features. Tune if needed.
# OpenCV hue range is 0..180; blue is centred around ~105-115.
BLUE_HSV_LOWER = (95, 110, 60)
BLUE_HSV_UPPER = (130, 255, 255)

# Warped (top-down) visualization size.
PX_PER_MM = 1.0                        # 1 px == 1 mm in the warped view
WARPED_W_PX = int(TABLE_LENGTH_MM * PX_PER_MM)
WARPED_H_PX = int(TABLE_WIDTH_MM * PX_PER_MM)

# Path for the saved calibration.
CALIBRATION_FILE = "calibration_data.npz"
