import numpy as np

####################
## SPLINE HELPERS ##
####################
""" Spline profile is normalized tau = [0, 1] """

# compute end acceleration
def compute_a1(p0, p1, p2, p3, v0, v1, v2, v3, a0, dt):
    """ 3-step lookahead C2 end-acceleration estimate """
    dt2 = dt ** 2
    d1  = (20 / dt2) * (p2 - 2*p1 + p0) - (2 / dt) * (3*v2 + 4*v1 + 3*v0) + 3*a0
    d2  = (20 / dt2) * (p3 - 2*p2 + p1) - (2 / dt) * (3*v3 + 4*v2 + 3*v1)
    return np.clip((4*d1 - d2) / 15, -10000, 10000)
 
# get spline coefficients
def get_quintic_coeffs_norm(p0, v0, a0, p1, v1, a1, dt):
    """ quintic spline coefficients for tau in [0, 1] (nondimensional time) """
    v0_n, v1_n = v0 * dt,    v1 * dt
    a0_n, a1_n = a0 * dt**2, a1 * dt**2
 
    c0 = p0
    c1 = v0_n
    c2 = a0_n / 2.0
 
    b_p = p1 - (c0 + c1 + c2)
    b_v = v1_n - (c1 + 2*c2)
    b_a = a1_n - 2*c2
 
    c3 =  10*b_p - 4*b_v + 0.5*b_a
    c4 = -15*b_p + 7*b_v -     b_a
    c5 =   6*b_p - 3*b_v + 0.5*b_a
 
    return (c0, c1, c2, c3, c4, c5)
 
# spline evaluation helper
def evaluate_spline_norm(coeffs, tau, dt):
    """ evaluate spline at nondimensional time tau; returns (position, velocity, acceleration)."""
    c0, c1, c2, c3, c4, c5 = coeffs
 
    p     = c0 + c1*tau + c2*tau**2 + c3*tau**3 + c4*tau**4 + c5*tau**5
    v_tau = c1 + 2*c2*tau + 3*c3*tau**2 + 4*c4*tau**3 + 5*c5*tau**4
    a_tau = 2*c2 + 6*c3*tau + 12*c4*tau**2 + 20*c5*tau**3
 
    return p, v_tau / dt, a_tau / dt**2