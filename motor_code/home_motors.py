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

async def initialize_and_calibrate(ids=[1, 2, 3, 4]):
    motors = {i: moteus.Controller(id=i) for i in ids}

    # Calibration logic
    calibration_data = {}

    async def stall_and_record(target_id):
        print(f"Homing Motor {target_id}...")
        # Clear any old faults/modes and set everything to stop mode
        for i in range(1, 5): await motors[i].set_stop()

        id_list = [1, 2, 3, 4]
        tasks = []
        
        vel = 10 # initialize to nonzero value
        flag = False

        t = 0.1

        while True:
            tasks = []
            for id in id_list:
                if (target_id == id):
                    tasks.append(motors[id].set_position(
                        position=np.nan,          # Ignore position tracking
                        velocity=-2*MOTOR_SIGN[target_id],
                        maximum_torque=0.3,       # Safety cap
                        watchdog_timeout=np.nan,
                        query=True
                    ))
                else:
                    tasks.append(motors[id].set_position(
                        feedforward_torque=-0.01 * MOTOR_SIGN[id],
                        maximum_torque=0.3,       # Safety cap
                        watchdog_timeout=np.nan
                    ))
        
            states = await asyncio.gather(*tasks)
            await asyncio.sleep(t)
            
            pos = states[target_id-1].values[moteus.Register.POSITION]
            vel = states[target_id-1].values[moteus.Register.VELOCITY]
            torque = states[target_id-1].values[moteus.Register.TORQUE]
            print(pos, vel, torque)
            
            # Only exit if we are actually stalled after trying to move
            if abs(vel) < 0.05:
                if flag == False:
                    flag = True
                    continue
                if flag == True:
                    return await motors[target_id].set_stop(query=True)
            else: flag = False

    # Motor 1: Wind back, set 0, record Motor 3
    await stall_and_record(1)
    await motors[1].set_rezero(0.0)
    m3_start = (await motors[3].set_stop(query=True)).values[moteus.Register.POSITION]
    await asyncio.sleep(0.1)

    # Motor 3: Wind back, record displacement
    m3_end = (await stall_and_record(3)).values[moteus.Register.POSITION]
    length = abs(m3_start - m3_end)
    await motors[3].set_rezero(0.0)
    await asyncio.sleep(0.1)
    
    # Motor 2: Wind back, set 0, record Motor 4
    await stall_and_record(2)
    await motors[2].set_rezero(0.0)
    m4_start = (await motors[4].set_stop(query=True)).values[moteus.Register.POSITION]
    await asyncio.sleep(0.1)
    
    # Motor 4: Wind back, set 0, record Motor 2
    await stall_and_record(4)
    await motors[4].set_rezero(0.0)
    calibration_data['m2_max'] = (await motors[2].set_stop(query=True)).values[moteus.Register.POSITION]

    # move all motors to middle position simultaneously
    async def move_together_all():
        flag = False
        counter = 0

        while True:
            # commands are sent nearly instantly one after another
            states = await asyncio.gather(
                motors[1].set_position(
                    position=MOTOR_SIGN[1]*length/2,
                    velocity=np.nan,
                    accel_limit=1,
                    maximum_torque=0.2,
                    watchdog_timeout=np.nan,
                    query=True
                ),        
                motors[2].set_position(
                    position=MOTOR_SIGN[2]*length/2,
                    velocity=np.nan,
                    accel_limit=1,
                    maximum_torque=0.2,
                    watchdog_timeout=np.nan,
                    query=True
                ),           
                motors[3].set_position(
                    position=MOTOR_SIGN[3]*length/2,
                    velocity=np.nan,
                    accel_limit=1,
                    maximum_torque=0.2,
                    watchdog_timeout=np.nan,
                    query=True
                ),
                motors[4].set_position(
                    position=MOTOR_SIGN[4]*length/2,
                    velocity=np.nan,
                    accel_limit=1,
                    maximum_torque=0.2,
                    watchdog_timeout=np.nan,
                    query=True
                )      
            )
            await asyncio.sleep(0.1)

            vel1 = states[0].values[moteus.Register.VELOCITY]
            vel2 = states[1].values[moteus.Register.VELOCITY]
            vel3 = states[2].values[moteus.Register.VELOCITY]
            vel4 = states[2].values[moteus.Register.VELOCITY]
            # print(vel1, vel3)

            counter += 1
            
            if abs(vel1) < 0.05 and abs(vel2) < 0.05 and abs(vel3) < 0.05 and abs(vel4) < 0.05 and (counter >= 10):
                if flag == True: return
                else: 
                    flag = True
                    continue
            else: flag = False

    await move_together_all()


    print(f"Calibration Complete: {calibration_data}")
    return motors, calibration_data





async def main():
    # Correct: This waits for the motors to initialize and calibrate
    motors, calibration_data = await initialize_and_calibrate()
    
    # Now you can use the motors
    print(f"Calibration finished: {calibration_data}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())