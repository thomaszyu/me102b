"""
Moteus PID tuning script for cable robot.

Runs step-response tests on individual motors or coordinated XY moves,
logs position/velocity/torque at every tick, and plots the results.
Lets you adjust kp_scale, kd_scale, accel_limit, and velocity_limit
between tests without reflashing firmware.

Usage:
    python tune_pid.py            # interactive menu
    python tune_pid.py --home     # home motors first, then menu
"""

import asyncio
import argparse
import sys
import os
import time
import numpy as np
import moteus

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'vision_code'))

from config import *
from kinematics_utils import xy_to_enc, enc_to_xy, xy_vel_to_enc_vel

# ── defaults (tweak these between runs) ──────────────────────────────
DEFAULT_KP_SCALE = KP_SCALE     # from config.py
DEFAULT_KD_SCALE = KD_SCALE     # from config.py
DEFAULT_ACCEL_LIMIT = ACCEL_LIMIT  # from config.py
DEFAULT_VEL_LIMIT = float('nan')  # rev/s  (nan = no limit)
DEFAULT_STEP_SIZE = 0.15         # rev  (per-motor step response size)
DEFAULT_STEP_DURATION = 1.5      # seconds to record after the step
DEFAULT_XY_STEP = 120.0           # mm  (XY step response size)


class TuneState:
    """Holds the tuning parameters that the user can change interactively."""

    def __init__(self):
        self.kp_scale = DEFAULT_KP_SCALE
        self.kd_scale = DEFAULT_KD_SCALE
        self.accel_limit = DEFAULT_ACCEL_LIMIT
        self.vel_limit = DEFAULT_VEL_LIMIT
        self.step_size = DEFAULT_STEP_SIZE
        self.step_duration = DEFAULT_STEP_DURATION
        self.xy_step = DEFAULT_XY_STEP
        self.max_torque = MAX_TORQUE

    def summary(self):
        al = f"{self.accel_limit:.1f}" if not np.isnan(self.accel_limit) else "none"
        vl = f"{self.vel_limit:.1f}" if not np.isnan(self.vel_limit) else "none"
        return (
            f"  kp_scale      = {self.kp_scale:.3f}\n"
            f"  kd_scale      = {self.kd_scale:.3f}\n"
            f"  accel_limit   = {al} rev/s²\n"
            f"  vel_limit     = {vl} rev/s\n"
            f"  max_torque    = {self.max_torque:.2f} Nm\n"
            f"  step_size     = {self.step_size:.3f} rev  (single-motor)\n"
            f"  xy_step       = {self.xy_step:.0f} mm   (XY move)\n"
            f"  step_duration = {self.step_duration:.1f} s"
        )


# ── data logging ─────────────────────────────────────────────────────

class StepLog:
    """Records time-series data for one step-response test."""

    def __init__(self, label=""):
        self.label = label
        self.t = []           # time (s, relative)
        self.cmd = []         # commanded position (rev or mm)
        self.pos = []         # measured position (rev or mm)
        self.vel = []         # measured velocity (rev/s or mm/s)
        self.torque = []      # measured torque (Nm)

    def append(self, t, cmd, pos, vel, torque):
        self.t.append(t)
        self.cmd.append(cmd)
        self.pos.append(pos)
        self.vel.append(vel)
        self.torque.append(torque)

    def as_arrays(self):
        return (
            np.array(self.t),
            np.array(self.cmd),
            np.array(self.pos),
            np.array(self.vel),
            np.array(self.torque),
        )

    def metrics(self):
        """Compute step-response quality metrics."""
        t, cmd, pos, vel, torque = self.as_arrays()
        if len(t) < 5:
            return {}

        # Settling threshold: 5% of step magnitude, with a minimum floor
        # (2% is too tight for cable robots — steady-state compliance > 0.003 rev)
        if pos.ndim == 2:
            # XY mode — compute magnitude of position error
            err = np.linalg.norm(pos - cmd, axis=1)
            final_err = err[-1]
            step_mag = np.linalg.norm(cmd[-1] - cmd[0])
            thresh = max(0.05 * step_mag, 3.0)  # 5% or 3mm floor
            settled_mask = err < thresh
            torque_peak = np.max(np.abs(torque))
        else:
            step_mag = abs(cmd[-1] - cmd[0])
            err = np.abs(pos - cmd)
            final_err = err[-1]
            thresh = max(0.05 * step_mag, 0.01)  # 5% or 0.01 rev floor (~3mm cable)
            settled_mask = err < thresh
            torque_peak = np.max(np.abs(torque))

        # Find settling time: walk backward from end to find last unsettled point.
        # Only consider the step portion (after cmd changes from its initial value).
        # The step onset is the first index where cmd differs from cmd[0].
        step_onset_idx = 0
        for j in range(len(cmd)):
            if (pos.ndim == 2 and not np.array_equal(cmd[j], cmd[0])) or \
               (pos.ndim != 2 and cmd[j] != cmd[0]):
                step_onset_idx = j
                break

        settling_time = t[-1] - t[step_onset_idx]  # default = full step duration
        for i in range(len(t) - 1, step_onset_idx - 1, -1):
            if not settled_mask[i]:
                if i + 1 < len(t):
                    settling_time = t[i + 1] - t[step_onset_idx]
                break
        else:
            settling_time = 0.0  # always within threshold

        # Overshoot: how far past the final value did we go?
        if pos.ndim == 1 and step_mag > 0.001:
            direction = np.sign(cmd[-1] - cmd[0])
            overshoot_raw = direction * (pos - cmd[-1])
            overshoot_pct = 100.0 * np.max(overshoot_raw) / step_mag
        else:
            overshoot_pct = 0.0

        return {
            'settling_time_s': settling_time,
            'overshoot_pct': max(0.0, overshoot_pct),
            'final_error': final_err,
            'peak_torque_Nm': torque_peak,
            'step_magnitude': step_mag,
        }


# ── motor helpers ────────────────────────────────────────────────────

async def read_encoders(motors):
    """Query all motors for current encoder positions (rev)."""
    states = await asyncio.gather(*[
        motors[mid].set_position(
            position=np.nan, kp_scale=0.0, kd_scale=0.0,
            feedforward_torque=0.05 * MOTOR_TORQUE_SCALE[mid],
            maximum_torque=0.1, watchdog_timeout=np.nan, query=True,
        )
        for mid in [1, 2, 3, 4]
    ])
    return np.array([s.values[moteus.Register.POSITION] for s in states])


async def stop_all(motors):
    """Command all motors to hold current position with zero velocity."""
    enc = await read_encoders(motors)
    await asyncio.gather(*[
        motors[mid].set_position(
            position=enc[mid - 1], velocity=0.0,
            maximum_torque=MAX_TORQUE, watchdog_timeout=np.nan, query=True,
        )
        for mid in [1, 2, 3, 4]
    ])


async def read_firmware_pid(motors):
    """Read firmware PID config from each motor via diagnostic stream."""
    params = [
        "servo.pid_position.kp",
        "servo.pid_position.ki",
        "servo.pid_position.kd",
        "servo.default_velocity_limit",
        "servo.default_accel_limit",
    ]
    print("\n── Firmware PID Configuration ──")
    for mid in [1, 2, 3, 4]:
        print(f"  Motor {mid}:")
        stream = moteus.Stream(motors[mid])
        for param in params:
            try:
                result = await stream.command(f"conf get {param}\n".encode())
                val = result.decode().strip()
                print(f"    {param} = {val}")
            except Exception as e:
                print(f"    {param} = (read failed: {e})")
    print()


# ── single-motor step response ───────────────────────────────────────

async def test_single_motor(motors, motor_id, state):
    """
    Step-response test on one motor.

    1. Record baseline position for 0.3s
    2. Step to (baseline + step_size) and hold for step_duration
    3. Step back to baseline and hold for step_duration
    4. Plot and print metrics
    """
    dt = TICK_RATE
    mid = motor_id
    ids = [1, 2, 3, 4]

    # Read starting positions for all motors
    start_enc = await read_encoders(motors)
    target_pos = start_enc[mid - 1]
    step = state.step_size * MOTOR_SIGN[mid]  # step in winding direction

    log_fwd = StepLog(label=f"Motor {mid} step +{state.step_size:.3f} rev")
    log_back = StepLog(label=f"Motor {mid} step back")

    # Single t0 for the entire test so times are monotonic across phases
    test_t0 = time.time()

    async def hold_and_record(target, log, duration, all_targets):
        """Send position commands and record response."""
        n_ticks = int(duration / dt)
        for _ in range(n_ticks):
            states = await asyncio.gather(*[
                motors[m].set_position(
                    position=all_targets[m - 1],
                    velocity=0.0,
                    kp_scale=state.kp_scale,
                    kd_scale=state.kd_scale,
                    accel_limit=state.accel_limit,
                    velocity_limit=state.vel_limit,
                    maximum_torque=state.max_torque * abs(MOTOR_TORQUE_SCALE[m]),
                    watchdog_timeout=np.nan,
                    query=True,
                )
                for m in ids
            ])
            await asyncio.sleep(dt)

            s = states[mid - 1]
            log.append(
                t=time.time() - test_t0,
                cmd=target,
                pos=s.values[moteus.Register.POSITION],
                vel=s.values.get(moteus.Register.VELOCITY, 0.0),
                torque=s.values.get(moteus.Register.TORQUE, 0.0),
            )

    # Baseline (0.3s)
    all_targets = start_enc.copy()
    print(f"  Baseline hold (0.3s)...")
    await hold_and_record(target_pos, log_fwd, 0.3, all_targets)

    # Forward step
    stepped_pos = target_pos + step
    all_targets[mid - 1] = stepped_pos
    print(f"  Stepping motor {mid}: {target_pos:.4f} → {stepped_pos:.4f} rev")
    await hold_and_record(stepped_pos, log_fwd, state.step_duration, all_targets)

    # Step back (separate log gets its own t0)
    test_t0 = time.time()
    all_targets[mid - 1] = target_pos
    print(f"  Stepping back: {stepped_pos:.4f} → {target_pos:.4f} rev")
    await hold_and_record(target_pos, log_back, state.step_duration, all_targets)

    # Metrics
    for log in [log_fwd, log_back]:
        m = log.metrics()
        print(f"\n  {log.label}:")
        print(f"    Settling time:  {m.get('settling_time_s', '?'):.3f} s")
        print(f"    Overshoot:      {m.get('overshoot_pct', '?'):.1f}%")
        print(f"    Final error:    {m.get('final_error', '?'):.5f} rev")
        print(f"    Peak torque:    {m.get('peak_torque_Nm', '?'):.3f} Nm")

    # Plot
    plot_step_response([log_fwd, log_back], state, f"Motor {mid} Step Response")
    return log_fwd, log_back


# ── accel/decel ramp test ─────────────────────────────────────────────

async def test_ramp(motors, motor_id, state):
    """
    Trapezoidal velocity profile test on one motor.

    Generates a smooth position trajectory with:
      1. Acceleration phase (ramp velocity from 0 to target)
      2. Cruise phase (hold constant velocity)
      3. Deceleration phase (ramp velocity back to 0)

    Records commanded vs actual position, velocity, and torque.
    Useful for tuning accel_limit and velocity_limit.
    """
    dt = TICK_RATE
    mid = motor_id
    ids = [1, 2, 3, 4]

    start_enc = await read_encoders(motors)
    start_pos = start_enc[mid - 1]

    # Trapezoidal profile parameters
    total_dist = state.step_size * 2  # rev (larger than step test for room to ramp)
    cruise_vel = 2.0   # rev/s — target cruise velocity
    accel = 8.0        # rev/s² — acceleration rate

    # Compute phase durations
    ramp_time = cruise_vel / accel
    ramp_dist = 0.5 * accel * ramp_time ** 2
    if 2 * ramp_dist >= total_dist:
        # Not enough room to reach cruise — triangular profile
        ramp_time = np.sqrt(total_dist / accel)
        cruise_time = 0.0
        total_dist = accel * ramp_time ** 2  # actual distance
    else:
        cruise_dist = total_dist - 2 * ramp_dist
        cruise_time = cruise_dist / cruise_vel

    total_time = 2 * ramp_time + cruise_time
    direction = MOTOR_SIGN[mid]  # winding direction

    print(f"\n── Ramp Test: Motor {mid} ──")
    print(f"  Distance: {total_dist:.3f} rev")
    print(f"  Cruise vel: {cruise_vel:.1f} rev/s, Accel: {accel:.1f} rev/s²")
    print(f"  Ramp: {ramp_time:.2f}s, Cruise: {cruise_time:.2f}s, Total: {total_time:.2f}s")

    log = StepLog(label=f"Motor {mid} ramp")
    all_targets = start_enc.copy()

    # Generate trajectory: position and velocity at each tick
    n_ticks = int((total_time + 0.5) / dt)  # +0.5s settle at end
    t0 = time.time()

    for k in range(n_ticks):
        t = k * dt
        # Compute trapezoidal profile at time t
        if t < ramp_time:
            # Accelerating
            vel = accel * t
            pos = 0.5 * accel * t ** 2
        elif t < ramp_time + cruise_time:
            # Cruising
            vel = cruise_vel
            pos = ramp_dist + cruise_vel * (t - ramp_time)
        elif t < total_time:
            # Decelerating
            t_decel = t - ramp_time - cruise_time
            vel = cruise_vel - accel * t_decel
            pos = ramp_dist + cruise_vel * cruise_time + cruise_vel * t_decel - 0.5 * accel * t_decel ** 2
        else:
            # Hold at end
            vel = 0.0
            pos = total_dist

        cmd_pos = start_pos + direction * pos
        cmd_vel = direction * vel
        all_targets[mid - 1] = cmd_pos

        states = await asyncio.gather(*[
            motors[m].set_position(
                position=all_targets[m - 1],
                velocity=cmd_vel if m == mid else 0.0,
                kp_scale=state.kp_scale,
                kd_scale=state.kd_scale,
                accel_limit=state.accel_limit,
                velocity_limit=state.vel_limit,
                maximum_torque=state.max_torque * abs(MOTOR_TORQUE_SCALE[m]),
                watchdog_timeout=np.nan,
                query=True,
            )
            for m in ids
        ])
        await asyncio.sleep(dt)

        s = states[mid - 1]
        log.append(
            t=time.time() - t0,
            cmd=cmd_pos,
            pos=s.values[moteus.Register.POSITION],
            vel=s.values.get(moteus.Register.VELOCITY, 0.0),
            torque=s.values.get(moteus.Register.TORQUE, 0.0),
        )

    # Ramp back
    log_back = StepLog(label=f"Motor {mid} ramp back")
    end_pos = start_pos + direction * total_dist
    t0_back = time.time()
    for k in range(n_ticks):
        t = k * dt
        if t < ramp_time:
            vel = accel * t
            pos = 0.5 * accel * t ** 2
        elif t < ramp_time + cruise_time:
            vel = cruise_vel
            pos = ramp_dist + cruise_vel * (t - ramp_time)
        elif t < total_time:
            t_decel = t - ramp_time - cruise_time
            vel = cruise_vel - accel * t_decel
            pos = ramp_dist + cruise_vel * cruise_time + cruise_vel * t_decel - 0.5 * accel * t_decel ** 2
        else:
            vel = 0.0
            pos = total_dist

        cmd_pos = end_pos - direction * pos  # reverse direction
        cmd_vel = -direction * vel
        all_targets[mid - 1] = cmd_pos

        states = await asyncio.gather(*[
            motors[m].set_position(
                position=all_targets[m - 1],
                velocity=cmd_vel if m == mid else 0.0,
                kp_scale=state.kp_scale,
                kd_scale=state.kd_scale,
                accel_limit=state.accel_limit,
                velocity_limit=state.vel_limit,
                maximum_torque=state.max_torque * abs(MOTOR_TORQUE_SCALE[m]),
                watchdog_timeout=np.nan,
                query=True,
            )
            for m in ids
        ])
        await asyncio.sleep(dt)

        s = states[mid - 1]
        log_back.append(
            t=time.time() - t0_back,
            cmd=cmd_pos,
            pos=s.values[moteus.Register.POSITION],
            vel=s.values.get(moteus.Register.VELOCITY, 0.0),
            torque=s.values.get(moteus.Register.TORQUE, 0.0),
        )

    # Metrics
    for lg in [log, log_back]:
        m = lg.metrics()
        print(f"\n  {lg.label}:")
        print(f"    Final error:    {m.get('final_error', '?'):.5f} rev")
        print(f"    Peak torque:    {m.get('peak_torque_Nm', '?'):.3f} Nm")

    plot_step_response([log, log_back], state, f"Motor {mid} Ramp Profile")
    return log, log_back


# ── XY step response ─────────────────────────────────────────────────

def generate_trapezoid_1d(distance, max_vel, accel, decel, dt):
    """
    Generate a 1D trapezoidal velocity profile.

    Returns arrays of (position, velocity) at each tick, normalized
    along [0, distance]. Handles the triangular case where distance
    is too short to reach max_vel.
    """
    # Time to accelerate/decelerate to max_vel
    t_accel = max_vel / accel
    t_decel = max_vel / decel
    d_accel = 0.5 * accel * t_accel ** 2
    d_decel = 0.5 * decel * t_decel ** 2

    if d_accel + d_decel >= distance:
        # Triangular profile — can't reach max_vel
        # Peak velocity: v_peak² = 2 * distance * accel * decel / (accel + decel)
        v_peak = np.sqrt(2.0 * distance * accel * decel / (accel + decel))
        t_accel = v_peak / accel
        t_decel = v_peak / decel
        t_cruise = 0.0
    else:
        v_peak = max_vel
        d_cruise = distance - d_accel - d_decel
        t_cruise = d_cruise / max_vel

    total_time = t_accel + t_cruise + t_decel

    n = int(np.ceil(total_time / dt)) + 1
    positions = np.zeros(n)
    velocities = np.zeros(n)

    for k in range(n):
        t = k * dt
        if t <= t_accel:
            # Accelerating
            velocities[k] = accel * t
            positions[k] = 0.5 * accel * t ** 2
        elif t <= t_accel + t_cruise:
            # Cruising
            tc = t - t_accel
            velocities[k] = v_peak
            positions[k] = 0.5 * accel * t_accel ** 2 + v_peak * tc
        elif t <= total_time:
            # Decelerating
            td = t - t_accel - t_cruise
            velocities[k] = v_peak - decel * td
            positions[k] = (0.5 * accel * t_accel ** 2 +
                            v_peak * t_cruise +
                            v_peak * td - 0.5 * decel * td ** 2)
        else:
            velocities[k] = 0.0
            positions[k] = distance

    return positions, velocities, total_time


async def test_xy_step(motors, direction, state, vision=None):
    """
    Profiled XY move with trapezoidal velocity (accel → cruise → decel).

    Uses config XY_MAX_VEL, XY_ACCEL, XY_DECEL for the motion profile.
    EKF closed-loop control with velocity feedforward to the motors.

    direction: 'x', 'y', or 'diag'
    vision: VisionSystem instance (required for EKF mode)
    """
    from ekf_controller import MalletEKF

    dt = TICK_RATE
    ids = [1, 2, 3, 4]

    if vision is None:
        print("  ERROR: XY tests require vision (camera). Run without --no-vision.")
        return None, None

    # Initialize EKF from camera
    ekf = MalletEKF()
    ekf_ready = False
    for _ in range(20):
        mallet, _, _ = vision.get_positions()
        if mallet[2]:
            ekf.initialize(mallet[0], mallet[1])
            ekf_ready = True
            break
        await asyncio.sleep(0.05)

    if not ekf_ready:
        print("  ERROR: Camera cannot see mallet. Cannot run EKF-based XY test.")
        return None, None

    start_xy = ekf.position.copy()
    current_enc = await read_encoders(motors)

    if direction == 'x':
        delta = np.array([state.xy_step, 0.0])
    elif direction == 'y':
        delta = np.array([0.0, state.xy_step])
    else:
        d = state.xy_step / np.sqrt(2)
        delta = np.array([d, d])

    target_xy = start_xy + delta
    move_dist = np.linalg.norm(delta)
    move_dir = delta / move_dist  # unit vector of motion

    # Generate trapezoidal profile along the motion direction
    trap_pos, trap_vel, move_time = generate_trapezoid_1d(
        move_dist, XY_MAX_VEL, XY_ACCEL, XY_DECEL, dt)
    n_move = len(trap_pos)

    # Convert 1D profile to 2D waypoints + velocities
    xy_path = start_xy + np.outer(trap_pos, move_dir)       # (n, 2)
    xy_vel_path = np.outer(trap_vel, move_dir)               # (n, 2) mm/s

    log = StepLog(label=f"XY {direction} +{state.xy_step:.0f}mm")
    log_back = StepLog(label=f"XY {direction} back")

    slack_ff = np.zeros(4)
    loop_time = time.time()

    print(f"  EKF closed-loop XY test with trapezoidal profile")
    print(f"  Start: ({start_xy[0]:.1f}, {start_xy[1]:.1f})  →  "
          f"Target: ({target_xy[0]:.1f}, {target_xy[1]:.1f})")
    print(f"  Profile: accel={XY_ACCEL:.0f} mm/s², cruise={XY_MAX_VEL:.0f} mm/s, "
          f"decel={XY_DECEL:.0f} mm/s²")
    print(f"  Move time: {move_time:.3f}s  ({n_move} ticks)")

    async def run_profile(xy_waypoints, xy_velocities, log, label, settle_time=0.5):
        """Execute a profiled move + settle period."""
        nonlocal current_enc, slack_ff, loop_time
        t0 = time.time()

        for k in range(len(xy_waypoints)):
            now = time.time()
            actual_dt = max(now - loop_time, 0.001)
            loop_time = now

            # EKF predict + camera update
            ekf.predict(actual_dt)
            mallet, _, _ = vision.get_positions()
            if mallet[2]:
                ekf.update([mallet[0], mallet[1]])

            commanded_xy = xy_waypoints[k]
            commanded_vel = xy_velocities[k]

            # Compute motor position target (EKF closed-loop)
            camera_xy = ekf.position
            expected_enc = xy_to_enc(camera_xy)
            enc_delta = xy_to_enc(commanded_xy) - expected_enc
            target_enc = current_enc + MOVE_GAIN * enc_delta - SIGNS * TENSION_BIAS_REV

            # Velocity feedforward: convert XY velocity to encoder velocity
            feedfwd_enc = xy_vel_to_enc_vel(commanded_xy, commanded_vel)

            states = await asyncio.gather(*[
                motors[m].set_position(
                    position=target_enc[m - 1],
                    velocity=feedfwd_enc[m - 1],
                    kp_scale=state.kp_scale,
                    kd_scale=state.kd_scale,
                    accel_limit=state.accel_limit,
                    velocity_limit=state.vel_limit,
                    feedforward_torque=slack_ff[m - 1],
                    maximum_torque=state.max_torque * abs(MOTOR_TORQUE_SCALE[m]),
                    watchdog_timeout=np.nan,
                    query=True,
                )
                for m in ids
            ])
            await asyncio.sleep(dt)

            current_enc = np.array([
                states[i].values[moteus.Register.POSITION] for i in range(4)
            ])

            # Slack detection
            for m in ids:
                i = m - 1
                torque = states[i].values[moteus.Register.TORQUE]
                if abs(torque) < 0.1:
                    slack_ff[i] = TENSION_TORQUE * MOTOR_TORQUE_SCALE[m]
                else:
                    slack_ff[i] = 0.0

            torques = np.array([
                states[i].values.get(moteus.Register.TORQUE, 0.0) for i in range(4)
            ])

            log.append(
                t=time.time() - t0,
                cmd=commanded_xy.copy(),
                pos=ekf.position.copy(),
                vel=np.linalg.norm(commanded_vel),
                torque=np.max(np.abs(torques)),
            )

        # Settle at final position
        final_xy = xy_waypoints[-1]
        n_settle = int(settle_time / dt)
        for _ in range(n_settle):
            now = time.time()
            actual_dt = max(now - loop_time, 0.001)
            loop_time = now

            ekf.predict(actual_dt)
            mallet, _, _ = vision.get_positions()
            if mallet[2]:
                ekf.update([mallet[0], mallet[1]])

            camera_xy = ekf.position
            expected_enc = xy_to_enc(camera_xy)
            enc_delta = xy_to_enc(final_xy) - expected_enc
            target_enc = current_enc + MOVE_GAIN * enc_delta - SIGNS * TENSION_BIAS_REV

            states = await asyncio.gather(*[
                motors[m].set_position(
                    position=target_enc[m - 1],
                    velocity=0.0,
                    kp_scale=state.kp_scale,
                    kd_scale=state.kd_scale,
                    accel_limit=state.accel_limit,
                    velocity_limit=state.vel_limit,
                    feedforward_torque=slack_ff[m - 1],
                    maximum_torque=state.max_torque * abs(MOTOR_TORQUE_SCALE[m]),
                    watchdog_timeout=np.nan,
                    query=True,
                )
                for m in ids
            ])
            await asyncio.sleep(dt)

            current_enc = np.array([
                states[i].values[moteus.Register.POSITION] for i in range(4)
            ])

            for m in ids:
                i = m - 1
                torque = states[i].values[moteus.Register.TORQUE]
                if abs(torque) < 0.1:
                    slack_ff[i] = TENSION_TORQUE * MOTOR_TORQUE_SCALE[m]
                else:
                    slack_ff[i] = 0.0

            torques = np.array([
                states[i].values.get(moteus.Register.TORQUE, 0.0) for i in range(4)
            ])

            log.append(
                t=time.time() - t0,
                cmd=final_xy.copy(),
                pos=ekf.position.copy(),
                vel=0.0,
                torque=np.max(np.abs(torques)),
            )

    # Re-anchor forward profile to current EKF position
    actual_start = ekf.position.copy()
    xy_path = actual_start + np.outer(trap_pos, move_dir)
    xy_vel_path = np.outer(trap_vel, move_dir)

    # Forward move
    print(f"  Moving {direction}...")
    await run_profile(xy_path, xy_vel_path, log, "forward")

    # Re-anchor return profile to where we actually ended up
    actual_end = ekf.position.copy()
    return_delta = actual_end - actual_start
    return_dist = np.linalg.norm(return_delta)
    if return_dist > 1.0:
        return_dir = return_delta / return_dist
    else:
        return_dir = move_dir

    trap_pos_back, trap_vel_back, _ = generate_trapezoid_1d(
        return_dist, XY_MAX_VEL, XY_ACCEL, XY_DECEL, dt)
    xy_path_back = actual_end - np.outer(trap_pos_back, return_dir)
    xy_vel_back = np.outer(-trap_vel_back, return_dir)
    print(f"  Moving back...")
    await run_profile(xy_path_back, xy_vel_back, log_back, "back")

    # Metrics
    for lg in [log, log_back]:
        m = lg.metrics()
        print(f"\n  {lg.label}:")
        print(f"    Settling time:  {m.get('settling_time_s', '?'):.3f} s")
        print(f"    Final error:    {m.get('final_error', '?'):.2f} mm")
        print(f"    Peak torque:    {m.get('peak_torque_Nm', '?'):.3f} Nm")

    plot_xy_step_response([log, log_back], state, f"XY Profiled ({direction})")
    return log, log_back


# ── sweep kp/kd ──────────────────────────────────────────────────────

async def sweep_gains(motors, motor_id, state):
    """
    Sweep kp_scale and kd_scale over a grid and run step responses.
    Plots a summary heatmap of settling time and overshoot.
    """
    kp_values = [0.1, 0.25, 0.5, 0.75, 1.0]
    kd_values = [0.1, 0.25, 0.5, 0.75, 1.0]

    print(f"\n  Sweeping kp_scale x kd_scale on motor {motor_id}")
    print(f"  kp: {kp_values}")
    print(f"  kd: {kd_values}")
    print(f"  {len(kp_values) * len(kd_values)} tests, ~{len(kp_values) * len(kd_values) * (state.step_duration + 0.5):.0f}s total\n")

    results = []
    original_kp = state.kp_scale
    original_kd = state.kd_scale

    for kp in kp_values:
        for kd in kd_values:
            state.kp_scale = kp
            state.kd_scale = kd
            print(f"  Testing kp={kp:.2f}, kd={kd:.2f}...")

            # Quick step response (shorter duration for sweep)
            saved_dur = state.step_duration
            state.step_duration = 1.0
            log_fwd, _ = await test_single_motor_quiet(motors, motor_id, state)
            state.step_duration = saved_dur

            m = log_fwd.metrics()
            results.append({
                'kp': kp,
                'kd': kd,
                'settling': m.get('settling_time_s', float('inf')),
                'overshoot': m.get('overshoot_pct', 0.0),
                'peak_torque': m.get('peak_torque_Nm', 0.0),
            })

    state.kp_scale = original_kp
    state.kd_scale = original_kd

    # Print summary table
    print(f"\n{'─' * 70}")
    print(f"  {'kp':>6}  {'kd':>6}  {'settling(s)':>11}  {'overshoot%':>10}  {'peak τ(Nm)':>10}")
    print(f"{'─' * 70}")
    best = min(results, key=lambda r: r['settling'] + 0.01 * r['overshoot'])
    for r in results:
        marker = " ◀ best" if r is best else ""
        print(f"  {r['kp']:6.2f}  {r['kd']:6.2f}  {r['settling']:11.3f}  "
              f"{r['overshoot']:10.1f}  {r['peak_torque']:10.3f}{marker}")
    print(f"{'─' * 70}")
    print(f"\n  Recommended: kp_scale={best['kp']:.2f}, kd_scale={best['kd']:.2f}")

    plot_sweep_results(results, motor_id)
    return results


async def test_single_motor_quiet(motors, motor_id, state):
    """Same as test_single_motor but without printing/plotting (for sweeps)."""
    dt = TICK_RATE
    mid = motor_id
    ids = [1, 2, 3, 4]

    start_enc = await read_encoders(motors)
    target_pos = start_enc[mid - 1]
    step = state.step_size * MOTOR_SIGN[mid]

    log_fwd = StepLog(label=f"Motor {mid} kp={state.kp_scale:.2f} kd={state.kd_scale:.2f}")

    all_targets = start_enc.copy()
    stepped_pos = target_pos + step
    all_targets[mid - 1] = stepped_pos

    n_ticks = int(state.step_duration / dt)
    t0 = time.time()
    for _ in range(n_ticks):
        states = await asyncio.gather(*[
            motors[m].set_position(
                position=all_targets[m - 1], velocity=0.0,
                kp_scale=state.kp_scale, kd_scale=state.kd_scale,
                accel_limit=state.accel_limit, velocity_limit=state.vel_limit,
                maximum_torque=state.max_torque * abs(MOTOR_TORQUE_SCALE[m]),
                watchdog_timeout=np.nan, query=True,
            )
            for m in ids
        ])
        await asyncio.sleep(dt)

        s = states[mid - 1]
        log_fwd.append(
            t=time.time() - t0,
            cmd=stepped_pos,
            pos=s.values[moteus.Register.POSITION],
            vel=s.values.get(moteus.Register.VELOCITY, 0.0),
            torque=s.values.get(moteus.Register.TORQUE, 0.0),
        )

    # Step back (no logging)
    all_targets[mid - 1] = target_pos
    for _ in range(int(0.5 / dt)):
        await asyncio.gather(*[
            motors[m].set_position(
                position=all_targets[m - 1], velocity=0.0,
                kp_scale=state.kp_scale, kd_scale=state.kd_scale,
                accel_limit=state.accel_limit, velocity_limit=state.vel_limit,
                maximum_torque=state.max_torque * abs(MOTOR_TORQUE_SCALE[m]),
                watchdog_timeout=np.nan, query=True,
            )
            for m in ids
        ])
        await asyncio.sleep(dt)

    return log_fwd, None


# ── plotting ─────────────────────────────────────────────────────────

def plot_step_response(logs, state, title):
    """Plot position, velocity, and torque for step response logs."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping plot)")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{title}\nkp={state.kp_scale:.2f}  kd={state.kd_scale:.2f}  "
                 f"accel_lim={state.accel_limit}  max_torque={state.max_torque:.1f}",
                 fontsize=11)

    for log in logs:
        t, cmd, pos, vel, torque = log.as_arrays()
        axes[0].plot(t, pos, label=f"{log.label} (actual)")
        axes[0].plot(t, cmd, '--', alpha=0.5, label=f"{log.label} (cmd)")
        axes[1].plot(t, vel, label=log.label)
        axes[2].plot(t, torque, label=log.label)

    axes[0].set_ylabel("Position (rev)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Velocity (rev/s)")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Torque (Nm)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"tune_{title.replace(' ', '_').lower()}.png"
    fpath = os.path.join(os.path.dirname(__file__), fname)
    plt.savefig(fpath, dpi=120)
    print(f"  Plot saved: {fpath}")
    plt.close()


def plot_xy_step_response(logs, state, title):
    """Plot XY position tracking and torque."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping plot)")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{title}\nkp={state.kp_scale:.2f}  kd={state.kd_scale:.2f}  "
                 f"accel_lim={state.accel_limit}", fontsize=11)

    for log in logs:
        t, cmd, pos, vel, torque = log.as_arrays()
        pos = np.array(pos)
        cmd = np.array(cmd)
        if pos.ndim == 2:
            err = np.linalg.norm(pos - cmd, axis=1)
            axes[0].plot(t, pos[:, 0], label=f"{log.label} X")
            axes[0].plot(t, cmd[:, 0], '--', alpha=0.5)
            axes[1].plot(t, pos[:, 1], label=f"{log.label} Y")
            axes[1].plot(t, cmd[:, 1], '--', alpha=0.5)
        else:
            err = np.abs(pos - cmd)
            axes[0].plot(t, pos, label=log.label)
            axes[0].plot(t, cmd, '--', alpha=0.5)

        axes[2].plot(t, torque, label=log.label)

    axes[0].set_ylabel("X position (mm)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Y position (mm)")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Peak torque (Nm)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"tune_{title.replace(' ', '_').replace('(', '').replace(')', '').lower()}.png"
    fpath = os.path.join(os.path.dirname(__file__), fname)
    plt.savefig(fpath, dpi=120)
    print(f"  Plot saved: {fpath}")
    plt.close()


def plot_sweep_results(results, motor_id):
    """Heatmap of settling time and overshoot from a kp/kd sweep."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping plot)")
        return

    kps = sorted(set(r['kp'] for r in results))
    kds = sorted(set(r['kd'] for r in results))

    settle_grid = np.full((len(kds), len(kps)), np.nan)
    overshoot_grid = np.full((len(kds), len(kps)), np.nan)
    for r in results:
        i = kds.index(r['kd'])
        j = kps.index(r['kp'])
        settle_grid[i, j] = r['settling']
        overshoot_grid[i, j] = r['overshoot']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Motor {motor_id} — kp/kd Sweep", fontsize=12)

    im1 = ax1.imshow(settle_grid, origin='lower', aspect='auto', cmap='RdYlGn_r')
    ax1.set_xticks(range(len(kps)))
    ax1.set_xticklabels([f"{v:.2f}" for v in kps])
    ax1.set_yticks(range(len(kds)))
    ax1.set_yticklabels([f"{v:.2f}" for v in kds])
    ax1.set_xlabel("kp_scale")
    ax1.set_ylabel("kd_scale")
    ax1.set_title("Settling Time (s)")
    plt.colorbar(im1, ax=ax1)
    for i in range(len(kds)):
        for j in range(len(kps)):
            ax1.text(j, i, f"{settle_grid[i,j]:.2f}", ha='center', va='center', fontsize=7)

    im2 = ax2.imshow(overshoot_grid, origin='lower', aspect='auto', cmap='RdYlGn_r')
    ax2.set_xticks(range(len(kps)))
    ax2.set_xticklabels([f"{v:.2f}" for v in kps])
    ax2.set_yticks(range(len(kds)))
    ax2.set_yticklabels([f"{v:.2f}" for v in kds])
    ax2.set_xlabel("kp_scale")
    ax2.set_ylabel("kd_scale")
    ax2.set_title("Overshoot (%)")
    plt.colorbar(im2, ax=ax2)
    for i in range(len(kds)):
        for j in range(len(kps)):
            ax2.text(j, i, f"{overshoot_grid[i,j]:.0f}", ha='center', va='center', fontsize=7)

    plt.tight_layout()
    fpath = os.path.join(os.path.dirname(__file__), f"tune_sweep_motor{motor_id}.png")
    plt.savefig(fpath, dpi=120)
    print(f"  Sweep plot saved: {fpath}")
    plt.close()


# ── interactive menu ─────────────────────────────────────────────────

def prompt_float(msg, default):
    """Prompt user for a float, accepting 'nan' and empty (=default)."""
    raw = input(f"  {msg} [{default}]: ").strip()
    if not raw:
        return default
    if raw.lower() == 'nan':
        return float('nan')
    try:
        return float(raw)
    except ValueError:
        print(f"  Invalid number, using {default}")
        return default


async def interactive_menu(motors, state, vision=None):
    """Main interactive tuning loop."""

    cv_status = "connected" if vision is not None else "not available"

    while True:
        print(f"\n{'═' * 50}")
        print("  MOTEUS PID TUNING")
        print(f"{'═' * 50}")
        print(f"\nCurrent parameters:")
        print(state.summary())
        print(f"  vision        = {cv_status}")
        print(f"\nCommands:")
        print(f"  1-4    Step response on motor 1-4")
        print(f"  r      Accel/decel ramp test on a motor")
        print(f"  x      XY step in X direction  (camera feedback: {cv_status})")
        print(f"  y      XY step in Y direction  (camera feedback: {cv_status})")
        print(f"  d      XY step diagonal        (camera feedback: {cv_status})")
        print(f"  s      Sweep kp/kd grid on a motor")
        print(f"  p      Change tuning parameters")
        print(f"  f      Read firmware PID config")
        print(f"  q      Quit")

        cmd = input("\n> ").strip().lower()

        if cmd in ('1', '2', '3', '4'):
            mid = int(cmd)
            print(f"\n── Step Response: Motor {mid} ──")
            await test_single_motor(motors, mid, state)

        elif cmd == 'r':
            mid = input("  Motor to ramp (1-4): ").strip()
            if mid in ('1', '2', '3', '4'):
                await test_ramp(motors, int(mid), state)
            else:
                print("  Invalid motor ID.")

        elif cmd == 'x':
            print(f"\n── XY Step Response: X ──")
            await test_xy_step(motors, 'x', state, vision=vision)

        elif cmd == 'y':
            print(f"\n── XY Step Response: Y ──")
            await test_xy_step(motors, 'y', state, vision=vision)

        elif cmd == 'd':
            print(f"\n── XY Step Response: Diagonal ──")
            await test_xy_step(motors, 'diag', state, vision=vision)

        elif cmd == 's':
            mid = input("  Motor to sweep (1-4): ").strip()
            if mid in ('1', '2', '3', '4'):
                await sweep_gains(motors, int(mid), state)
            else:
                print("  Invalid motor ID.")

        elif cmd == 'p':
            print(f"\n── Change Parameters (Enter to keep current) ──")
            state.kp_scale = prompt_float("kp_scale", state.kp_scale)
            state.kd_scale = prompt_float("kd_scale", state.kd_scale)
            state.accel_limit = prompt_float("accel_limit (rev/s², 'nan'=none)", state.accel_limit)
            state.vel_limit = prompt_float("vel_limit (rev/s, 'nan'=none)", state.vel_limit)
            state.max_torque = prompt_float("max_torque (Nm)", state.max_torque)
            state.step_size = prompt_float("step_size (rev)", state.step_size)
            state.xy_step = prompt_float("xy_step (mm)", state.xy_step)
            state.step_duration = prompt_float("step_duration (s)", state.step_duration)

        elif cmd == 'f':
            await read_firmware_pid(motors)

        elif cmd == 'q':
            print("Stopping motors...")
            await stop_all(motors)
            if vision is not None:
                vision.stop()
            break

        else:
            print("  Unknown command.")


# ── main ─────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Moteus PID tuning for cable robot")
    parser.add_argument('--home', action='store_true', help="Home motors before tuning")
    parser.add_argument('--no-vision', action='store_true', help="Skip vision system (encoder FK only)")
    args = parser.parse_args()

    # Start vision system (unless --no-vision)
    vision = None
    if not args.no_vision:
        try:
            from vision import VisionSystem
            vision = VisionSystem()
            vision.start(show_display=False)
            print("Vision system started. Waiting for camera...")
            await asyncio.sleep(1.0)

            mallet, _, _ = vision.get_positions()
            if mallet[2]:
                print(f"  Camera sees mallet at ({mallet[0]:.1f}, {mallet[1]:.1f}) mm")
            else:
                print(f"  Camera running but mallet not detected yet")
        except Exception as e:
            print(f"  Vision system failed to start: {e}")
            print(f"  Continuing with encoder FK only.")
            vision = None
    else:
        print("Vision disabled (--no-vision). Using encoder FK for XY tests.")

    # Connect to motors
    if args.home:
        from home_motors import initialize_and_calibrate
        print("Homing motors...")
        motors, _ = await initialize_and_calibrate()
    else:
        transport = moteus.Fdcanusb()
        motors = {
            mid: moteus.Controller(id=mid, transport=transport)
            for mid in [1, 2, 3, 4]
        }
        print("Connected to motors (no homing). Reading positions...")
        enc = await read_encoders(motors)
        if vision is not None:
            mallet, _, _ = vision.get_positions()
            if mallet[2]:
                print(f"  Camera position: ({mallet[0]:.1f}, {mallet[1]:.1f}) mm")
        xy = enc_to_xy(enc)
        print(f"  Encoder FK position: ({xy[0]:.1f}, {xy[1]:.1f}) mm")

    state = TuneState()
    await interactive_menu(motors, state, vision=vision)


if __name__ == "__main__":
    asyncio.run(main())
