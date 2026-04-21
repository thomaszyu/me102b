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
    4: -2.0,
}

CORNER_POSITIONS = {
    1: np.array([13.8, 291.85]), # front left
    2: np.array([-426.3, 235.3]), # rear left
    3: np.array([-426.3, -235.3]), # rear right
    4: np.array([13.8, -291.85]), # front right
}

CORNERS = np.array([CORNER_POSITIONS[i] for i in [1, 2, 3, 4]])
SIGNS = np.array([MOTOR_SIGN[i] for i in [1, 2, 3, 4]])

SPOOL_DIAM_MM = 75.0
SPOOL_CIRC_MM = np.pi * SPOOL_DIAM_MM

TICK_RATE = 0.05 # sec

ACCEL_LIMIT = 2.0 # rev/s^2
MAX_TORQUE = 0.6 # N m