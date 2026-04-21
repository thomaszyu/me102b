import numpy as np
from config import *

########################
## KINEMATICS HELPERS ##
########################

""" These are all in terms of encoder readings and not lengths, where possible. """

# inverse kinematics
def xy_to_enc(pos): # pos is a 2-vector, this retunrs a 4-vector containing counts
    """
    inverse kinematics: get encoder positions (rev) from mallet xy position (mm)

    after homing, the encoders are zeroed at fully wound positions, 
    so cable length maps directly to encoder readings via spool_circ_mm.

    enc[i] = SIGNS[i] * cable_length_mm[i] / spool_circ_mm
    """
    lengths_mm = np.linalg.norm(CORNERS - pos, axis=1)  # get lengths
    return SIGNS * lengths_mm / SPOOL_CIRC_MM           # map lengths to encoder readings

# forward kinematics
def enc_to_xy(enc): 
    """ least-squares forward kinematics. get mallet xy position (mm) from encoder readings. """
    # enc is vector of encoder readings
    lengths_mm = np.abs(enc * SIGNS) * SPOOL_CIRC_MM   # always positive
 
    # setup and solve least squares
    x0, y0 = CORNERS[0]
    l0 = lengths_mm[0]
    A, b = [], []
    for i in [1, 2, 3]:
        xi, yi = CORNERS[i]
        li = lengths_mm[i]
        A.append([2 * (xi - x0), 2 * (yi - y0)])
        b.append((l0**2 - li**2) + (xi**2 - x0**2) + (yi**2 - y0**2))
    return np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]

# convert xy velocity to encoder velocity
def xy_vel_to_enc_vel(pos_xy, vel_xy):
    """
    Cable-drive Jacobian: Cartesian velocity (mm/s) → encoder velocity (rev/s).
 
        d(length)/dt = unit_vec(puck→corner) · v_puck
        d(enc)/dt    = SIGN * d(length)/dt / spool_circ
 
    Returns shape (4,).
    """
    diff  = CORNERS - pos_xy                                  # compute displacements from corners
    norms = np.linalg.norm(diff, axis=1, keepdims=True)       # compute lengths (vectors to corners)
    unit  = diff / np.maximum(norms, 1e-9)                    # normalize to unit vectors
    dl_dt = unit @ vel_xy                                     # velocity along the cable length
    return SIGNS * dl_dt / SPOOL_CIRC_MM                      # encoder velocities