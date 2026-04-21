import asyncio
import moteus
import numpy as np
from config import * # global variables

async def read_encoders(motors):
    """
    motors is a dict
    query all four motors and return current encoder positions (rev)
    uses minimal hold torque so the mallet doesn't drift during the query
    """
    states = await asyncio.gather(*[
        motors[mid].set_position(
            position           = np.nan,
            kp_scale           = 0.0,
            kd_scale           = 0.0,
            feedforward_torque = 0.05 * MOTOR_TORQUE_SCALE[mid],
            maximum_torque     = 0.1,
            watchdog_timeout   = np.nan,
            query              = True,
        )
        for mid in [1, 2, 3, 4]
    ])
    return np.array([states[i].values[moteus.Register.POSITION] for i in range(4)])
 
 
async def execute_move(motors:      dict,
                       ticks:       list) -> np.ndarray:
    """
    stream pre-planned tick commands to the four motors
    returns the actual encoder positions (rev) read back after the final tick
    """
    ids = [1, 2, 3, 4]
 
    for tick in ticks:
        target  = tick['target_enc']
        feedfwd = tick['feedfwd_vel']
 
        await asyncio.gather(*[
            motors[mid].set_position(
                position         = target[mid - 1],
                velocity         = feedfwd[mid - 1],
                accel_limit      = ACCEL_LIMIT,
                maximum_torque   = MAX_TORQUE,
                watchdog_timeout = np.nan,
                query            = False,   # skip query mid-move for latency
            )
            for mid in ids
        ])
        await asyncio.sleep(TICK_RATE)
 
    # Hold final position and read back actual encoder values
    final_target = ticks[-1]['target_enc']
    states = await asyncio.gather(*[
        motors[mid].set_position(
            position         = final_target[mid - 1],
            velocity         = 0.0,
            accel_limit      = ACCEL_LIMIT,
            maximum_torque   = MAX_TORQUE,
            watchdog_timeout = np.nan,
            query            = True,        # read back only on the final tick
        )
        for mid in ids
    ])
 
    return np.array([states[i].values[moteus.Register.POSITION] for i in range(4)])