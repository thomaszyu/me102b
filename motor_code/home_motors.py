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

MOTOR_TORQUE_SCALE = {
    1: 1,
    2: 1,
    3: 1,
    4: -1,
}

async def initialize_and_calibrate(ids=[1, 2, 3, 4]):
    motors = {i: moteus.Controller(id=i) for i in ids}

    # Calibration logic
    calibration_data = {}

    async def stall_and_record(target_id):
        print(f"Homing Motor {target_id}...")
        # Only stop the target motor to clear faults; others keep tension
        await motors[target_id].set_stop()

        id_list = [1, 2, 3, 4]
        tasks = []

        flag = False
        has_moved = False

        t = 0.1

        while True:
            tasks = []
            for id in id_list:
                if (target_id == id):
                    tasks.append(motors[id].set_position(
                        position=np.nan,
                        kd_scale=1.0,
                        velocity=-0.75*MOTOR_SIGN[target_id],
                        maximum_torque=0.75,
                        watchdog_timeout=np.nan,
                        query=True
                    ))
                else:
                    tasks.append(motors[id].set_position(
                        position=np.nan,
                        kp_scale = 0.0,
                        kd_scale = 0.0,
                        feedforward_torque= 0.05*MOTOR_TORQUE_SCALE[id],
                        maximum_torque=0.1,
                        watchdog_timeout=np.nan,
                        query=True
                    ))

            states = await asyncio.gather(*tasks)
            await asyncio.sleep(t)

            pos = states[target_id-1].values[moteus.Register.POSITION]
            vel = states[target_id-1].values[moteus.Register.VELOCITY]
            torque = states[target_id-1].values[moteus.Register.TORQUE]
            print(pos, vel, torque)

            # Wait until motor has actually started moving before checking stall
            if abs(vel) > 0.2:
                has_moved = True

            if not has_moved:
                continue

            # Only exit if we are actually stalled after moving
            if abs(vel) < 0.04:
                if flag == False:
                    flag = True
                    continue
                if flag == True:
                    print(f"Motor {target_id} homed.")
                    # Return position query but keep holding tension
                    return await motors[target_id].set_position(
                        position=np.nan,
                        kp_scale=0.0,
                        kd_scale=0.0,
                        feedforward_torque=0.05 * MOTOR_TORQUE_SCALE[target_id],
                        maximum_torque=0.1,
                        watchdog_timeout=np.nan,
                        query=True
                    )
            else: flag = False

    async def hold_tension(id):
        """Command a motor to hold tension without position control."""
        return await motors[id].set_position(
            position=np.nan,
            kp_scale=0.0,
            kd_scale=0.0,
            feedforward_torque=0.05 * MOTOR_TORQUE_SCALE[id],
            maximum_torque=0.1,
            watchdog_timeout=np.nan,
            query=True
        )

    async def hold_all_except(skip_id):
        """Keep tension on all motors except one."""
        tasks = []
        for id in [1, 2, 3, 4]:
            if id != skip_id:
                tasks.append(hold_tension(id))
        await asyncio.gather(*tasks)

    # Motor 4: Wind back, set 0, record Motor 2
    await stall_and_record(4)
    await motors[4].set_rezero(0.0)
    await hold_all_except(4)
    m2_start = (await hold_tension(2)).values[moteus.Register.POSITION]
    await asyncio.sleep(0.1)

    # Motor 2: Wind back, record displacement
    m2_end = (await stall_and_record(2)).values[moteus.Register.POSITION]
    length = abs(m2_start - m2_end)
    await motors[2].set_rezero(0.0)
    await hold_all_except(2)
    await asyncio.sleep(0.1)

    # Motor 1: Wind back, set 0, record Motor 3
    await stall_and_record(1)
    await motors[1].set_rezero(0.0)
    await hold_all_except(1)
    m3_start = (await hold_tension(3)).values[moteus.Register.POSITION]
    await asyncio.sleep(0.1)

    # Motor 3: Wind back, set 0
    await stall_and_record(3)
    await motors[3].set_rezero(0.0)
    await hold_all_except(3)
    calibration_data['m1_max'] = (await hold_tension(1)).values[moteus.Register.POSITION]

    # move all motors to middle position simultaneously
    async def move_together_all():
        flag = False
        counter = 0
        ids = [1, 2, 3, 4]

        while True:
            tasks = []

            for id in ids:
                tasks.append(
                    motors[id].set_position(
                        position=MOTOR_SIGN[id]*length/2,
                        velocity=np.nan,
                        accel_limit=1,
                        maximum_torque=0.6,
                        watchdog_timeout=np.nan,
                        query=True
                    )
                )
            # commands are sent nearly instantly one after another
            states = await asyncio.gather(*tasks)
            await asyncio.sleep(0.1)

            vel1 = states[0].values[moteus.Register.VELOCITY]
            vel2 = states[1].values[moteus.Register.VELOCITY]
            vel3 = states[2].values[moteus.Register.VELOCITY]
            vel4 = states[3].values[moteus.Register.VELOCITY]

            counter += 1
            
            if abs(vel1) < 0.05 and abs(vel2) < 0.05 and abs(vel3) < 0.05 and abs(vel4) < 0.05 and (counter >= 30):
                if flag == True: return
                else: 
                    flag = True
                    continue
            else: flag = False

    # await move_together_all()


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