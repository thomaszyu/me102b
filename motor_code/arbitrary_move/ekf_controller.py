"""
EKF-based closed-loop mallet controller.

Fuses camera measurements (primary, +/-2mm) with encoder-based predictions.
Accepts target positions in the same mm coordinate frame as the vision system.

Usage:
    from ekf_controller import EKFController

    controller = EKFController(motors, vision_system)
    await controller.move_to(target_xy, duration)
"""

import asyncio
import sys
import os
import numpy as np
import time
import moteus

# Add vision_code to path so we can import VisionSystem
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'vision_code'))

from config import *
from kinematics_utils import xy_to_enc, enc_to_xy, xy_vel_to_enc_vel
from motor_utils_for_arbitrary_move import read_encoders, execute_move
from spline_utils import get_quintic_coeffs_norm, evaluate_spline_norm


################
## EKF        ##
################

class MalletEKF:
    """
    Extended Kalman Filter for mallet position tracking.
    State: [x, y, vx, vy]  (mm, mm, mm/s, mm/s)

    Prediction: constant-velocity model
    Measurement: camera (x, y) in mm
    """

    def __init__(self,
                 process_noise_pos=5.0,     # mm — how much the mallet jerks between ticks
                 process_noise_vel=50.0,    # mm/s — velocity uncertainty growth per tick
                 camera_noise=2.0):         # mm — camera measurement noise (+/-2mm)
        # State [x, y, vx, vy]
        self.x = np.zeros(4)
        # Covariance
        self.P = np.eye(4) * 1000.0  # start uncertain

        # Process noise
        self.Q = np.diag([
            process_noise_pos**2,
            process_noise_pos**2,
            process_noise_vel**2,
            process_noise_vel**2,
        ])

        # Measurement noise (camera measures x, y)
        self.R = np.eye(2) * camera_noise**2

        # Measurement matrix: we observe [x, y] from state [x, y, vx, vy]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

    def initialize(self, x, y):
        """Set initial position (e.g., from first camera reading)."""
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([4.0, 4.0, 100.0, 100.0])  # reasonably confident in position

    def predict(self, dt):
        """Predict state forward by dt seconds using constant-velocity model."""
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q * dt  # scale process noise by dt

    def update(self, z):
        """Update state with camera measurement z = [x_mm, y_mm]."""
        z = np.asarray(z, dtype=float)
        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.R      # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def position(self):
        return self.x[:2].copy()

    @property
    def velocity(self):
        return self.x[2:].copy()


######################
## CONTROLLER       ##
######################

class EKFController:
    """
    Closed-loop mallet controller using EKF + camera feedback.

    Coordinates are in the vision system's mm frame (origin at table center).
    The controller:
    1. Plans a quintic trajectory from current position to target
    2. Each tick: reads camera, runs EKF, computes correction, sends to motors
    """

    def __init__(self, motors, vision_system, kp=0.5):
        """
        motors: dict {1..4: moteus.Controller}
        vision_system: VisionSystem instance (already started)
        kp: proportional correction gain (0 = open loop, 1 = full correction)
        """
        self.motors = motors
        self.vision = vision_system
        self.kp = kp
        self.ekf = MalletEKF()
        self._initialized = False

    async def initialize_ekf(self):
        """Initialize EKF from current camera reading. Call once after homing."""
        for _ in range(10):  # wait for a valid camera frame
            mallet, _, _ = self.vision.get_positions()
            if mallet[2]:  # valid
                self.ekf.initialize(mallet[0], mallet[1])
                self._initialized = True
                print(f"EKF initialized at ({mallet[0]:.1f}, {mallet[1]:.1f}) mm")
                return
            await asyncio.sleep(0.05)
        # Fallback: use encoder FK
        enc = await read_encoders(self.motors)
        xy = enc_to_xy(enc)
        self.ekf.initialize(xy[0], xy[1])
        self._initialized = True
        print(f"EKF initialized from encoders at ({xy[0]:.1f}, {xy[1]:.1f}) mm")

    async def move_to(self, target_xy, duration):
        """
        Move mallet to target_xy (mm) over duration seconds with closed-loop correction.

        target_xy: np.array([x, y]) in vision mm coordinates
        duration: seconds
        """
        if not self._initialized:
            await self.initialize_ekf()

        target_xy = np.asarray(target_xy, dtype=float)
        dt = TICK_RATE
        num_steps = max(1, int(round(duration / dt)))

        # Get current state from EKF + encoders
        start_xy = self.ekf.position.copy()
        start_enc = await read_encoders(self.motors)

        print(f"EKF Move: ({start_xy[0]:.1f}, {start_xy[1]:.1f}) → "
              f"({target_xy[0]:.1f}, {target_xy[1]:.1f})  ({duration:.2f}s)")

        # Plan quintic trajectory in XY space
        zero2 = np.zeros(2)
        coeffs_xy = get_quintic_coeffs_norm(
            start_xy, zero2, zero2,
            target_xy, zero2, zero2, duration)

        taus = np.linspace(0.0, 1.0, num_steps + 1)
        xy_path = np.array([evaluate_spline_norm(coeffs_xy, t, duration)[0] for t in taus])
        vxy_path = np.array([evaluate_spline_norm(coeffs_xy, t, duration)[1] for t in taus])

        # Control loop
        ids = [1, 2, 3, 4]
        last_time = time.time()
        slack_ff = np.zeros(4)
        smooth_offset = None  # smoothed encoder offset
        OFFSET_ALPHA = 0.15   # smoothing factor (0 = frozen, 1 = no smoothing)

        for k in range(1, num_steps + 1):
            now = time.time()
            actual_dt = now - last_time
            last_time = now

            # --- EKF predict ---
            self.ekf.predict(actual_dt)

            # --- EKF update (camera) ---
            mallet, _, _ = self.vision.get_positions()
            if mallet[2]:  # valid detection
                self.ekf.update([mallet[0], mallet[1]])

            # --- Read current encoders ---
            current_enc = await read_encoders(self.motors)

            # --- Smoothed offset from camera ---
            camera_xy = self.ekf.position
            expected_enc = xy_to_enc(camera_xy)
            raw_offset = current_enc - expected_enc
            if smooth_offset is None:
                smooth_offset = raw_offset.copy()
            else:
                smooth_offset = (1 - OFFSET_ALPHA) * smooth_offset + OFFSET_ALPHA * raw_offset

            # --- Target for this tick ---
            planned_xy = xy_path[k]
            enc_delta = xy_to_enc(planned_xy) - expected_enc
            target_enc = current_enc + MOVE_GAIN * enc_delta - SIGNS * TENSION_BIAS_REV

            # Velocity feedforward from planned trajectory
            feedfwd_enc = xy_vel_to_enc_vel(planned_xy, vxy_path[k])

            # --- Send to motors ---
            states = await asyncio.gather(*[
                self.motors[mid].set_position(
                    position           = target_enc[mid - 1],
                    velocity           = feedfwd_enc[mid - 1],
                    feedforward_torque = slack_ff[mid - 1],
                    maximum_torque     = MAX_TORQUE * abs(MOTOR_TORQUE_SCALE[mid]),
                    watchdog_timeout   = np.nan,
                    query              = True,
                )
                for mid in ids
            ])
            await asyncio.sleep(dt)

            # Slack detection for next tick
            for mid in ids:
                i = mid - 1
                torque = states[i].values[moteus.Register.TORQUE]
                if abs(torque) < 0.1:
                    slack_ff[i] = TENSION_TORQUE * MOTOR_TORQUE_SCALE[mid]
                else:
                    slack_ff[i] = 0.0

        # Final: hold target position
        camera_xy = self.ekf.position
        final_enc = await read_encoders(self.motors)
        expected_enc = xy_to_enc(camera_xy)
        enc_delta = xy_to_enc(target_xy) - expected_enc
        hold_enc = final_enc + MOVE_GAIN * enc_delta - SIGNS * TENSION_BIAS_REV
        await asyncio.gather(*[
            self.motors[mid].set_position(
                position         = hold_enc[mid - 1],
                velocity         = 0.0,
                maximum_torque   = MAX_TORQUE,
                watchdog_timeout = np.nan,
                query            = True,
            )
            for mid in ids
        ])

        # Report using EKF estimate
        final_estimated = self.ekf.position
        error_mm = np.linalg.norm(final_estimated - target_xy)
        print(f"Done. EKF pos: ({final_estimated[0]:.1f}, {final_estimated[1]:.1f})  "
              f"Error: {error_mm:.1f}mm")

        return final_estimated

    async def follow_path(self, xy_positions, xy_velocities, dt=TICK_RATE):
        """
        Stream a continuous path tick-by-tick. No stopping between points.

        xy_positions: (N, 2) array of XY waypoints in mm, one per tick
        xy_velocities: (N, 2) array of XY velocities in mm/s, one per tick
        dt: time between ticks (defaults to TICK_RATE)
        """
        if not self._initialized:
            await self.initialize_ekf()

        ids = [1, 2, 3, 4]
        slack_ff = np.zeros(4)
        smooth_offset = None
        OFFSET_ALPHA = 0.2

        # Seed current_enc before the loop
        current_enc = await read_encoders(self.motors)

        print(f"Following path: {len(xy_positions)} ticks, {len(xy_positions)*dt:.1f}s")

        for k in range(len(xy_positions)):
            # --- EKF predict + update ---
            self.ekf.predict(dt)
            mallet, _, _ = self.vision.get_positions()
            if mallet[2]:
                self.ekf.update([mallet[0], mallet[1]])

            # --- Smoothed offset (uses encoder readings from LAST tick's response) ---
            camera_xy = self.ekf.position
            expected_enc = xy_to_enc(camera_xy)
            raw_offset = current_enc - expected_enc
            if smooth_offset is None:
                smooth_offset = raw_offset.copy()
            else:
                smooth_offset = (1 - OFFSET_ALPHA) * smooth_offset + OFFSET_ALPHA * raw_offset

            # --- Target ---
            target_xy = xy_positions[k]
            enc_delta = xy_to_enc(target_xy) - expected_enc
            target_enc = current_enc + MOVE_GAIN * enc_delta - SIGNS * TENSION_BIAS_REV

            # --- Velocity feedforward ---
            vel_xy = xy_velocities[k]
            feedfwd_enc = xy_vel_to_enc_vel(target_xy, vel_xy)

            # --- Send to motors (single CAN round-trip) ---
            states = await asyncio.gather(*[
                self.motors[mid].set_position(
                    position           = target_enc[mid - 1],
                    velocity           = feedfwd_enc[mid - 1],
                    feedforward_torque = slack_ff[mid - 1],
                    maximum_torque     = MAX_TORQUE * abs(MOTOR_TORQUE_SCALE[mid]),
                    watchdog_timeout   = np.nan,
                    query              = True,
                )
                for mid in ids
            ])
            await asyncio.sleep(dt)

            # Read encoder positions from motor responses (no extra CAN trip)
            current_enc = np.array([
                states[i].values[moteus.Register.POSITION] for i in range(4)
            ])

            # Slack detection
            for mid in ids:
                i = mid - 1
                torque = states[i].values[moteus.Register.TORQUE]
                if abs(torque) < 0.1:
                    slack_ff[i] = TENSION_TORQUE * MOTOR_TORQUE_SCALE[mid]
                else:
                    slack_ff[i] = 0.0

        print(f"Path complete. EKF pos: ({self.ekf.position[0]:.1f}, {self.ekf.position[1]:.1f})")
        return self.ekf.position


def generate_circle_path(center, radius, duration, num_rotations=1.0, dt=TICK_RATE):
    """
    Generate a smooth circle trajectory with S-curve angular velocity.
    Returns (positions, velocities) arrays at tick rate.
    """
    num_steps = int(duration / dt)
    t = np.linspace(0, 1, num_steps)

    # Quintic S-curve for smooth start/stop
    s = 10*t**3 - 15*t**4 + 6*t**5
    theta = s * (2 * np.pi * num_rotations)

    ds_dt = (30*t**2 - 60*t**3 + 30*t**4) / duration
    dtheta_dt = ds_dt * (2 * np.pi * num_rotations)

    cx, cy = center
    positions = np.column_stack([
        cx + radius * np.cos(theta),
        cy + radius * np.sin(theta),
    ])
    velocities = np.column_stack([
        -radius * np.sin(theta) * dtheta_dt,
         radius * np.cos(theta) * dtheta_dt,
    ])

    return positions, velocities


######################
## EXAMPLE USAGE    ##
######################

async def main():
    import moteus
    from home_motors import initialize_and_calibrate
    from vision import VisionSystem

    # Start vision (no display — macOS requires GUI on main thread)
    vis = VisionSystem()
    vis.start(show_display=False)

    # Calibrate vision manually first by running: python vision.py
    # Then start this script. Or pre-set H_matrix/H_display.
    print("Waiting for valid camera detection...")
    await asyncio.sleep(1.0)

    # Home motors
    motors, _ = await initialize_and_calibrate()

    # Create controller
    ctrl = EKFController(motors, vis, kp=0.5)
    await ctrl.initialize_ekf()

    # Move to start of circle
    center = np.array([-200.0, 0.0])
    radius = 100.0
    start = center + np.array([radius, 0.0])
    await ctrl.move_to(start, duration=1.5)

    # Smooth continuous circle
    positions, velocities = generate_circle_path(
        center=center, radius=radius, duration=5.0, num_rotations=1.0)
    await ctrl.follow_path(positions, velocities)

    print("Circle complete.")
    vis.stop()


if __name__ == "__main__":
    asyncio.run(main())
