import asyncio
import moteus
import numpy as np

from home_motors import initialize_and_calibrate
from motor_utils_for_arbitrary_move import *
from spline_utils import *
from kinematics_utils import *


########################
## TRAJECTORY PLANNER ##
########################

def plan_trajectory(start_enc: np.ndarray,
                    end_xy:    np.ndarray,
                    duration:  float,
                    dt:        float = TICK_RATE) -> list[dict]:
    """
    plan a straight-line cartesian move and return tick-by-tick motor commands
    all in encoder revolutions — no mm in the output
 
    strategy
    1. Plan a quintic spline in xy space (straight line, zero boundary velocity, acceleration)
    2. sample the spline at every tick to get XY position and velocity
    3. convert xy to encoder position via xy_to_enc (inverse kinematics)
       convert xy velocity to encoder velocity via cable Jacobian (see kinematics_utils)
    4. apply an offset so the planned delta matches the actual motor position (start_enc)
       rather than the geometric start of the spline.
 
    parameters
    start_enc : (4,) current encoder readings in revolutions (from moteus)
    end_xy    : (2,) target puck position in mm
    duration  : seconds to complete the move
    dt        : control tick period in seconds
 
    returns ticks : list of dicts, one per tick, each with:
        'target_enc'  : (4,) absolute encoder target (rev) to pass to set_position
        'feedfwd_vel' : (4,) encoder velocity feedforward (rev/s)
    """
    num_steps = max(1, int(round(duration / dt)))
    start_xy  = enc_to_xy(start_enc)
 
    # step 1. cartesian quintic spline (zero boundary velocities & accelerations)
    zero2  = np.zeros(2)
    p_data = np.array([start_xy, end_xy, end_xy, end_xy, end_xy], dtype=float)
    v_data = np.zeros((5, 2))
 
    coeffs_xy = get_quintic_coeffs_norm(p_data[0], v_data[0], zero2,
                                        p_data[1], v_data[1], zero2, duration)
 
    # step 2. sample at every tick boundary (tau = 0 to 1)
    taus     = np.linspace(0.0, 1.0, num_steps + 1)
    xy_path  = np.array([evaluate_spline_norm(coeffs_xy, t, duration)[0] for t in taus])
    vxy_path = np.array([evaluate_spline_norm(coeffs_xy, t, duration)[1] for t in taus])
 
    # step 3. convert to encoder space
    enc_path     = np.array([xy_to_enc(xy)                       for xy in xy_path])
    enc_vel_path = np.array([xy_vel_to_enc_vel(xy_path[k], vxy_path[k])
                             for k in range(num_steps + 1)])

    # step 3.5. retraction bias — shorten all cables slightly to maintain tension
    enc_path = enc_path - SIGNS * TENSION_BIAS_REV

    # step 4. offset so planned deltas are rooted at the actual encoder reading
    # enc_path[0] is the geometric start; start_enc is where the motor actually is
    # any discrepancy (slack, slip, prior error) is absorbed here once
    enc_offset = start_enc - enc_path[0]   # (4,) rev
 
    ticks = [
        {
            'target_enc':  enc_path[k] + enc_offset,
            'prev_enc':    enc_path[k - 1] + enc_offset,
            'feedfwd_vel': enc_vel_path[k],
        }
        for k in range(1, num_steps + 1)
    ]
 
    return ticks

 
# high-level API
async def move_to(motors:      dict,
                  end_xy:      np.ndarray,
                  duration:    float) -> np.ndarray:
    """
    move the end-effector to end_xy (mm) over duration seconds
 
    reads live encoder positions at call time. no external state tracking
    required between moves. we can call move_to repeatedly.
 
    parameters
    motors      : dict {1..4 : moteus.Controller} (from initialize_and_calibrate())
    end_xy      : (2,) target puck position in mm
    duration    : seconds to complete the move
    max_torque  : maximum motor torque in Nm
    accel_limit : motor acceleration limit in rev/s²
 
    returns final_enc : (4,) actual encoder readings at end of move (rev)
    """
    start_enc = await read_encoders(motors)
    start_xy  = enc_to_xy(start_enc)
    print(f"Move: {np.round(start_xy, 1)} → {np.round(end_xy, 1)}  ({duration:.2f}s)")
 
    ticks     = plan_trajectory(start_enc, end_xy, duration)
    final_enc = await execute_move(motors, ticks)
 
    actual_xy = enc_to_xy(final_enc)
    error_mm  = np.linalg.norm(actual_xy - end_xy)
    print(f"Done. Actual: {np.round(actual_xy, 1)}  Error: {error_mm:.2f} mm")
 
    return final_enc
 

###################
## EXAMPLE USAGE ##
###################
 
async def main():
    motors, _ = await initialize_and_calibrate()
 
    # move_to reads encoders fresh each time — just pass the target
    await move_to(motors, np.array([-200.0,  150.0]), duration=2.0)
    await move_to(motors, np.array([-200.0, -150.0]), duration=2.0)
    await move_to(motors, np.array([  -300.0,   0.0]), duration=2.0)
 
    print("All moves complete.")
 
 
if __name__ == "__main__":
    asyncio.run(main())