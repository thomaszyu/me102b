import asyncio
import moteus
import numpy as np

# Sign convention: positive dl = cable gets longer = spool pays out.
# If your motors wind cable for positive revolutions, flip the sign here.
MOTOR_SIGN = {
    1:  -1, # winding is positive revs
    2:  1, # winding is negative revs
    3:  -1, # winding is positive revs
    4:  1, # winding is negative revs
}

async def main():
    test = await moteus.Controller(id=1).set_stop(query=True)
    print(test)


async def check():
    c = moteus.Controller(id=1)
    # Just a query, no torque, no movement
    state = await c.set_stop(query=True)
    print(f"State: {state}")

async def check2(id):
    c = moteus.Controller(id=id)
    state = await c.set_position(
        position=np.nan,          # Ignore position tracking
        velocity=np.nan,            # No velocity goal/limit interference
        feedforward_torque= 0.3, # Negative for retract, Positive for extend
        maximum_torque=0.5,       # Safety cap
        watchdog_timeout=np.nan   # Keeps it moving without constant packets
    )
    await asyncio.sleep(0.5)

asyncio.run(check2(1))