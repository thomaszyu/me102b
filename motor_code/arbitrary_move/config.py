import numpy as np

# stores all global variables for consistency across files

MOTOR_SIGN = {
    1:  -1.0, # winding is positive revs
    2:  1.0, # winding is negative revs
    3:  -1.0, # winding is positive revs
    4:  1.0, # winding is negative revs
}

MOTOR_TORQUE_SCALE = {
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: -1.0,
}

FRAME_OFFSET = np.array([-35.0, -17.0])  # [X, Y] mm — shift all corners to align frame with table

CORNER_POSITIONS = {
    1: np.array([13.8, 291.85]) + FRAME_OFFSET, # front left
    2: np.array([-426.3, 235.3]) + FRAME_OFFSET, # rear left
    3: np.array([-426.3, -235.3]) + FRAME_OFFSET, # rear right
    4: np.array([13.8, -291.85]) + FRAME_OFFSET, # front right
}

CORNERS = np.array([CORNER_POSITIONS[i] for i in [1, 2, 3, 4]])
SIGNS = np.array([MOTOR_SIGN[i] for i in [1, 2, 3, 4]])

SPOOL_DIAM_MM = 75.0
SPOOL_CIRC_MM = 320.0 #tuned
TICK_RATE = 0.01 # sec

ACCEL_LIMIT = 6.0 # rev/s^2
MAX_TORQUE = 4.0 # N m
TENSION_BIAS_REV = 0.015 # rev bias to maintain tension lmao ajdkjhasbdkjashdba
TENSION_TORQUE = 0.05
MOVE_GAIN = 1.0  # scales encoder delta — increase if mallet undershoots, decrease if overshoots