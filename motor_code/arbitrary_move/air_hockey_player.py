"""
Naive MPC air hockey player.

Tracks the puck, predicts its trajectory (with wall bounces),
and decides whether to defend (block incoming shots) or attack
(strike the puck toward the opponent's goal).

Usage:
    python air_hockey_player.py
"""

import asyncio
import sys
import os
import numpy as np
import time
import moteus

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'vision_code'))

from config import *
from kinematics_utils import xy_to_enc, xy_vel_to_enc_vel
from motor_utils_for_arbitrary_move import read_encoders
from ekf_controller import EKFController, MalletEKF
from spline_utils import get_quintic_coeffs_norm, evaluate_spline_norm


########################
## TABLE / GAME CONFIG #
########################

TABLE_CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "table_calibration.json")

# Default table bounds (overridden by calibration JSON if present)
TABLE_X_MIN = -273.0
TABLE_X_MAX =  273.0
TABLE_Y_MIN = -240.0
TABLE_Y_MAX =  240.0
CORNER_RADIUS = 0.0

def load_table_bounds():
    """Load table bounds from calibration JSON. Updates module globals."""
    global TABLE_X_MIN, TABLE_X_MAX, TABLE_Y_MIN, TABLE_Y_MAX, CORNER_RADIUS
    global ROBOT_GOAL_X, DEFEND_X, OPPONENT_GOAL_X, MALLET_Y_MIN, MALLET_Y_MAX
    if os.path.exists(TABLE_CALIBRATION_FILE):
        import json
        with open(TABLE_CALIBRATION_FILE) as f:
            data = json.load(f)
        b = data['bounds']
        TABLE_X_MIN = b['x_min']
        TABLE_X_MAX = b['x_max']
        TABLE_Y_MIN = b['y_min']
        TABLE_Y_MAX = b['y_max']
        CORNER_RADIUS = data.get('corner_radius', 0.0)
        print(f"Loaded table bounds: X[{TABLE_X_MIN:.0f}, {TABLE_X_MAX:.0f}] "
              f"Y[{TABLE_Y_MIN:.0f}, {TABLE_Y_MAX:.0f}] R={CORNER_RADIUS:.0f}mm")
    else:
        print(f"No table calibration found, using defaults.")

    # Recompute derived values
    ROBOT_GOAL_X  = TABLE_X_MIN
    DEFEND_X      = TABLE_X_MIN + 120
    OPPONENT_GOAL_X = TABLE_X_MAX
    MALLET_Y_MIN = TABLE_Y_MIN + 30
    MALLET_Y_MAX = TABLE_Y_MAX - 30

# Load on import
load_table_bounds()

# Object radii (mm)
PUCK_RADIUS = 25.0
MALLET_RADIUS = 30.0

# Wall offset (must match TABLE_BORDER_OFFSET in air_hockey_debug.py)
WALL_OFFSET = 50.0

# Puck center bounce limits (actual wall + offset - puck radius)
PUCK_X_MIN = TABLE_X_MIN - WALL_OFFSET + PUCK_RADIUS
PUCK_X_MAX = TABLE_X_MAX + WALL_OFFSET - PUCK_RADIUS
PUCK_Y_MIN = TABLE_Y_MIN - WALL_OFFSET + PUCK_RADIUS
PUCK_Y_MAX = TABLE_Y_MAX + WALL_OFFSET - PUCK_RADIUS

# Mallet safe workspace (corners inset by this margin — stay away from edge)
WORKSPACE_MARGIN = 80.0
_corner_xs = [c[0] for c in CORNERS]
_corner_ys = [c[1] for c in CORNERS]
MALLET_X_MIN = min(_corner_xs) + WORKSPACE_MARGIN
MALLET_X_MAX = max(_corner_xs) - WORKSPACE_MARGIN
MALLET_Y_MIN = min(_corner_ys) + WORKSPACE_MARGIN
MALLET_Y_MAX = max(_corner_ys) - WORKSPACE_MARGIN

# Robot defends the LEFT side (negative X)
ROBOT_GOAL_X  = TABLE_X_MIN
DEFEND_X      = max(TABLE_X_MIN + 120, MALLET_X_MIN)  # never outside safe workspace
ATTACK_LIMIT_X = 0.0

# Opponent's goal (RIGHT side)
OPPONENT_GOAL_X = TABLE_X_MAX
OPPONENT_GOAL_CENTER_Y = 0.0


def clamp_to_workspace(xy):
    """Clamp an XY target to the safe mallet workspace."""
    return np.array([
        np.clip(xy[0], MALLET_X_MIN, MALLET_X_MAX),
        np.clip(xy[1], MALLET_Y_MIN, MALLET_Y_MAX),
    ])

# Strategy thresholds
PUCK_SLOW_THRESH = 50.0    # mm/s — puck considered "slow" below this
PUCK_COMING_THRESH = -30.0 # mm/s — puck vx below this = coming toward us (negative X)
ATTACK_ZONE_X = 120.0      # puck must be left of this X to attempt a strike
DEFENSE_ONLY = False       # set True to disable attacking


########################
## PUCK TRACKER       ##
########################

class PuckTracker:
    """Tracks puck position and velocity with filtering."""

    def __init__(self, alpha_pos=0.5, alpha_vel=0.3, max_jump=100.0):
        
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.last_raw = None
        self.last_time = None
        self.initialized = False
        self.alpha_pos = alpha_pos
        self.alpha_vel = alpha_vel
        self.max_jump = max_jump
        self.visible = False

    def update(self, raw_pos, valid, now=None):
        """Update with a new camera reading."""
        if now is None:
            now = time.time()

        self.visible = valid
        if not valid:
            return

        raw = np.array(raw_pos[:2], dtype=float)

        if not self.initialized:
            self.pos = raw.copy()
            self.vel = np.zeros(2)
            self.last_raw = raw.copy()
            self.last_time = now
            self.initialized = True
            return

        # Jump rejection
        if np.linalg.norm(raw - self.pos) > self.max_jump:
            return

        # Velocity estimate
        dt = now - self.last_time
        if dt > 0.001:
            raw_vel = (raw - self.last_raw) / dt
            self.vel = (1 - self.alpha_vel) * self.vel + self.alpha_vel * raw_vel

        # Position filter
        self.pos = (1 - self.alpha_pos) * self.pos + self.alpha_pos * raw

        self.last_raw = raw.copy()
        self.last_time = now


########################
## PUCK PREDICTION    ##
########################

def predict_puck_trajectory(pos, vel, dt, num_steps):
    """
    Simulate puck trajectory with wall bounces.
    Returns list of (x, y) positions.
    """
    x, y = float(pos[0]), float(pos[1])
    vx, vy = float(vel[0]), float(vel[1])
    trajectory = []

    for _ in range(num_steps):
        x += vx * dt
        y += vy * dt

        # Reflect off top/bottom walls (puck center bounces at puck radius from wall)
        if y < PUCK_Y_MIN:
            y = 2 * PUCK_Y_MIN - y
            vy = -vy
        elif y > PUCK_Y_MAX:
            y = 2 * PUCK_Y_MAX - y
            vy = -vy

        # Reflect off left/right walls (goals are open, but approximate)
        if x < PUCK_X_MIN:
            x = 2 * PUCK_X_MIN - x
            vx = -vx
        elif x > PUCK_X_MAX:
            x = 2 * PUCK_X_MAX - x
            vx = -vx

        trajectory.append(np.array([x, y]))

    return trajectory


def predict_intercept(puck_pos, puck_vel, target_x):
    """
    Find where and when the puck crosses target_x, with wall bounces.
    Returns (intercept_y, time_seconds) or (None, None).
    """
    x, y = float(puck_pos[0]), float(puck_pos[1])
    vx, vy = float(puck_vel[0]), float(puck_vel[1])

    # Puck must be moving toward target_x
    if abs(vx) < 5.0:
        return None, None
    if (target_x < x and vx > 0) or (target_x > x and vx < 0):
        return None, None

    # Simulate until crossing target_x (max 5 seconds)
    sim_dt = 0.005
    t = 0.0
    for _ in range(1000):
        x += vx * sim_dt
        y += vy * sim_dt
        t += sim_dt

        # Wall bounces (Y only)
        if y < PUCK_Y_MIN:
            y = 2 * PUCK_Y_MIN - y
            vy = -vy
        elif y > PUCK_Y_MAX:
            y = 2 * PUCK_Y_MAX - y
            vy = -vy

        # Check crossing
        if (vx < 0 and x <= target_x) or (vx > 0 and x >= target_x):
            return np.clip(y, MALLET_Y_MIN, MALLET_Y_MAX), t

    return None, None


def predict_intercept_y(puck_pos, puck_vel, target_x):
    """Convenience wrapper — returns just the Y or None."""
    y, _ = predict_intercept(puck_pos, puck_vel, target_x)
    return y


########################
## STRATEGY           ##
########################

# Contact geometry
CONTACT_DIST = PUCK_RADIUS + MALLET_RADIUS  # 55mm

# Attack tuning
STRIKE_THROUGH = 20.0         # mm past contact for follow-through
GOAL_Y_HALF = 80.0            # half-width of goal opening
ATTACK_COOLDOWN_TICKS = 30    # ticks (~0.3s) cooldown after attack
ATTACK_MAX_PUCK_SPEED = 400.0 # mm/s — only attack pucks slower than this
STRIKE_SPEED = 800.0         # mm/s — mallet speed at contact
MIN_APPROACH_TIME = 0.15      # seconds — minimum approach duration
ATTACK_LINE_X = -120.0  # intercept at the attack zone line, not defense line


class AttackState:
    """Tracks a trajectory-based attack."""
    def __init__(self):
        self.phase = None           # None, 'ARMED', 'TRAJECTORY'
        self.traj_pos = None        # (N, 2) XY waypoints
        self.traj_vel = None        # (N, 2) XY velocities mm/s
        self.traj_tick = 0
        self.traj_duration = 0.0    # approach phase duration (seconds)
        self.intercept_y = 0.0      # Y position on the attack line
        self.strike_dir = None      # for debug viz
        self.contact_pos = None     # for debug viz
        self.windup_target = None   # for debug viz
        self.cooldown = 0

    def reset(self):
        self.phase = None
        self.traj_pos = None
        self.traj_vel = None
        self.traj_tick = 0
        self.traj_duration = 0.0
        self.intercept_y = 0.0
        self.strike_dir = None
        self.contact_pos = None
        self.windup_target = None

    def start_cooldown(self):
        self.reset()
        self.cooldown = ATTACK_COOLDOWN_TICKS


def plan_attack(puck, mallet_xy, defend_pos):
    """
    Plan a trajectory to intercept the puck at the attack line,
    hit it toward the opponent's goal, then return to defense.

    Flow:
      1. Predict where/when the puck crosses the attack line
      2. Quintic spline: mallet → intercept point, arriving with
         velocity aimed at the goal
      3. Follow-through past contact
      4. Quintic spline back to defense position

    Returns: (traj_pos, traj_vel, contact_pos, strike_dir, start_pos) or None
    """
    if not puck.initialized or not puck.visible:
        return None

    puck_speed = np.linalg.norm(puck.vel)
    if puck_speed > ATTACK_MAX_PUCK_SPEED:
        return None

    # Don't attack if puck is behind the defense line (too close to our goal)
    if puck.pos[0] < DEFEND_X + 20:
        return None

    attack_x = ATTACK_LINE_X if ATTACK_LINE_X is not None else DEFEND_X

    # Predict where the puck crosses the attack line
    intercept_y, intercept_time = predict_intercept(puck.pos, puck.vel, attack_x)

    if intercept_y is None:
        # Puck isn't heading toward attack line — if slow and nearby, use current Y
        if puck.pos[0] < attack_x + 100 and puck_speed < 100:
            intercept_y = np.clip(puck.pos[1], MALLET_Y_MIN, MALLET_Y_MAX)
            intercept_time = None
        else:
            return None

    # The mallet intercepts AT the attack line X
    # Contact point = (attack_x, intercept_y) — mallet center is here at contact
    contact_pos = np.array([attack_x, intercept_y])

    # Aim at opponent goal center
    goal_pos = np.array([OPPONENT_GOAL_X, OPPONENT_GOAL_CENTER_Y])
    hit_dir = goal_pos - contact_pos
    hit_dist = np.linalg.norm(hit_dir)
    if hit_dist < 1.0:
        return None
    hit_dir = hit_dir / hit_dist

    contact_clamped = clamp_to_workspace(contact_pos)
    if np.linalg.norm(contact_clamped - contact_pos) > 15:
        return None

    # Compute approach duration
    approach_dist = np.linalg.norm(mallet_xy - contact_clamped)
    if approach_dist < 10:
        return None

    # Duration: match puck timing if available, otherwise distance-based
    avg_speed = max(0.7 * STRIKE_SPEED, 300.0)
    dist_duration = max(approach_dist / avg_speed, MIN_APPROACH_TIME)
    if intercept_time is not None and intercept_time > MIN_APPROACH_TIME:
        # Use puck timing but don't go slower than we need to
        duration = min(intercept_time, dist_duration * 0.8)
    else:
        duration = dist_duration
    duration = min(duration, 0.5)  # cap at 0.5s

    # --- Build full trajectory (vectorized, no Python loops) ---
    zero2 = np.zeros(2)
    end_vel = hit_dir * STRIKE_SPEED

    def eval_spline_batch(coeffs, n, dur):
        """Evaluate quintic spline at n points, returns (n,2) pos and vel."""
        c0, c1, c2, c3, c4, c5 = coeffs
        t = np.linspace(0.0, 1.0, n).reshape(-1, 1)  # (n, 1)
        t2, t3, t4, t5 = t**2, t**3, t**4, t**5
        pos = c0 + c1*t + c2*t2 + c3*t3 + c4*t4 + c5*t5
        vel = (c1 + 2*c2*t + 3*c3*t2 + 4*c4*t3 + 5*c5*t4) / dur
        return pos, vel

    # Phase 1: Approach spline
    coeffs = get_quintic_coeffs_norm(
        mallet_xy.copy(), zero2, zero2,
        contact_clamped, end_vel, zero2, duration)
    n_approach = max(2, int(duration / TICK_RATE))
    approach_pos, approach_vel = eval_spline_batch(coeffs, n_approach, duration)

    # Phase 2: Follow-through
    n_ft = max(1, int(STRIKE_THROUGH / (STRIKE_SPEED * TICK_RATE)))
    ft_k = np.arange(1, n_ft + 1).reshape(-1, 1)
    ft_pos = contact_clamped + hit_dir * STRIKE_SPEED * TICK_RATE * ft_k
    ft_pos = np.clip(ft_pos, [MALLET_X_MIN, MALLET_Y_MIN], [MALLET_X_MAX, MALLET_Y_MAX])
    ft_vel = np.tile(end_vel, (n_ft, 1))

    # Phase 3: Return spline
    return_start = ft_pos[-1].copy()
    return_dist = np.linalg.norm(return_start - defend_pos)
    return_dur = max(return_dist / 300.0, 0.3)
    coeffs_ret = get_quintic_coeffs_norm(
        return_start, end_vel, zero2,
        defend_pos.copy(), zero2, zero2, return_dur)
    n_return = max(2, int(return_dur / TICK_RATE))
    return_pos, return_vel = eval_spline_batch(coeffs_ret, n_return, return_dur)
    return_pos = np.clip(return_pos, [MALLET_X_MIN, MALLET_Y_MIN], [MALLET_X_MAX, MALLET_Y_MAX])

    # Concatenate all phases
    traj_pos = np.vstack([approach_pos, ft_pos, return_pos])
    traj_vel = np.vstack([approach_vel, ft_vel, return_vel])

    # Reject if any point is outside workspace
    if (np.any(traj_pos[:, 0] < MALLET_X_MIN) or np.any(traj_pos[:, 0] > MALLET_X_MAX) or
            np.any(traj_pos[:, 1] < MALLET_Y_MIN) or np.any(traj_pos[:, 1] > MALLET_Y_MAX)):
        return None

    return (traj_pos, traj_vel,
            contact_clamped, hit_dir.copy(), mallet_xy.copy())


_attack = AttackState()
_last_strategy = 'IDLE'


def decide_strategy(puck, mallet_xy):
    """
    Decide what to do based on puck state.
    Returns: ('DEFEND', target_y, None) or ('ATTACK', target_xy, target_vel)
             or ('IDLE', None, None)
    target_vel is XY velocity in mm/s (or None for position-only modes).
    """
    global _attack, _last_strategy

    if not puck.initialized or not puck.visible:
        _attack.reset()
        return 'IDLE', None, None

    px, py = puck.pos
    vx = puck.vel[0]
    puck_on_our_side = px < 0

    # Tick down cooldown
    if _attack.cooldown > 0:
        _attack.cooldown -= 1

    # --- DEFEND: puck coming toward our goal fast ---
    defend_thresh = PUCK_COMING_THRESH
    if _last_strategy == 'DEFEND':
        defend_thresh = PUCK_COMING_THRESH + 20

    # During trajectory execution, only abort for very fast incoming
    if _attack.phase == 'TRAJECTORY':
        defend_thresh = -400.0

    # Skip DEFEND if puck is attackable (ARMED or planning)
    puck_speed = np.linalg.norm(puck.vel)
    if puck_speed < ATTACK_MAX_PUCK_SPEED and puck_on_our_side and not DEFENSE_ONLY:
        defend_thresh = -400.0

    if vx < defend_thresh:
        if _attack.phase is not None:
            _attack.start_cooldown()
        else:
            _attack.reset()
        _last_strategy = 'DEFEND'
        intercept_y = predict_intercept_y(puck.pos, puck.vel, DEFEND_X)
        if intercept_y is not None:
            return 'DEFEND', intercept_y, None
        return 'DEFEND', np.clip(py, MALLET_Y_MIN, MALLET_Y_MAX), None

    # --- Executing trajectory: advance one tick ---
    if _attack.phase == 'TRAJECTORY' and _attack.traj_pos is not None:
        k = _attack.traj_tick
        if k < len(_attack.traj_pos):
            pos = _attack.traj_pos[k]
            vel = _attack.traj_vel[k]
            _attack.traj_tick += 1
            _last_strategy = 'STRIKE'
            return 'STRIKE', pos, vel
        else:
            _attack.start_cooldown()
            _last_strategy = 'DEFEND'
            return 'DEFEND', np.clip(py, MALLET_Y_MIN, MALLET_Y_MAX), None

    # --- Try to plan and execute attack immediately ---
    if not DEFENSE_ONLY and puck_on_our_side and _attack.cooldown == 0:
        defend_pos = clamp_to_workspace(np.array([DEFEND_X, 0.0]))
        plan = plan_attack(puck, mallet_xy, defend_pos)
        if plan is not None:
            traj_pos, traj_vel, contact, strike_dir, windup = plan
            _attack.phase = 'TRAJECTORY'
            _attack.traj_pos = traj_pos
            _attack.traj_vel = traj_vel
            _attack.traj_tick = 1
            _attack.contact_pos = contact
            _attack.strike_dir = strike_dir
            _attack.windup_target = windup
            _last_strategy = 'STRIKE'
            return 'STRIKE', traj_pos[0], traj_vel[0]

    # --- DEFEND (passive) ---
    if puck_on_our_side:
        _attack.reset()
        _last_strategy = 'DEFEND'
        return 'DEFEND', np.clip(py, MALLET_Y_MIN, MALLET_Y_MAX), None

    # --- IDLE ---
    _attack.reset()
    _last_strategy = 'IDLE'
    return 'IDLE', None, None


########################
## MAIN PLAYER LOOP   ##
########################

async def play_air_hockey(ctrl, duration=120.0):
    """
    Main air hockey loop. Same motor control pattern as puck_tracker.
    """
    if not ctrl._initialized:
        await ctrl.initialize_ekf()

    puck = PuckTracker(alpha_pos=0.5, alpha_vel=0.3, max_jump=100.0)

    ids = [1, 2, 3, 4]
    slack_ff = np.zeros(4)
    smooth_offset = None
    OFFSET_ALPHA = 0.2
    MAX_MALLET_LOST = 10
    mallet_lost_count = 0

    # Smoothed desired + ramped commanded position
    desired_xy = ctrl.ekf.position.copy()
    commanded_xy = ctrl.ekf.position.copy()
    smooth_target_enc = None   # smoothed motor commands (filters EKF noise)
    CMD_ALPHA = 0.3            # motor command smoothing (0=frozen, 1=no filter)
    MAX_SPEED_NORMAL = 400.0   # mm/s for defend/idle
    loop_time = time.time()

    # Seed encoders
    current_enc = await read_encoders(ctrl.motors)

    last_strategy = 'IDLE'

    print(f"Air hockey player started. Defending X={DEFEND_X:.0f}mm for {duration:.0f}s.")
    print(f"  Workspace: X[{MALLET_X_MIN:.0f}, {MALLET_X_MAX:.0f}] Y[{MALLET_Y_MIN:.0f}, {MALLET_Y_MAX:.0f}]")

    start_time = time.time()
    tick = 0
    while time.time() - start_time < duration:
        # --- Measure real dt ---
        now = time.time()
        dt = max(now - loop_time, 0.001)
        loop_time = now

        # --- EKF predict + update (mallet) ---
        ctrl.ekf.predict(dt)
        mallet_reading, puck_reading, _ = ctrl.vision.get_positions()

        if mallet_reading[2]:
            ctrl.ekf.update([mallet_reading[0], mallet_reading[1]])
            mallet_lost_count = 0
        else:
            mallet_lost_count += 1

        # --- Update puck tracker ---
        puck.update(puck_reading, puck_reading[2])

        # --- Safety freeze ---
        if mallet_lost_count >= MAX_MALLET_LOST:
            states = await asyncio.gather(*[
                ctrl.motors[mid].set_position(
                    position=current_enc[mid - 1], velocity=0.0,
                    maximum_torque=MAX_TORQUE * abs(MOTOR_TORQUE_SCALE[mid]),
                    watchdog_timeout=np.nan, query=True,
                ) for mid in ids
            ])
            await asyncio.sleep(dt)
            current_enc = np.array([
                states[i].values[moteus.Register.POSITION] for i in range(4)
            ])
            continue

        # --- Decide strategy ---
        mallet_xy = ctrl.ekf.position
        strategy, target_data, target_vel = decide_strategy(puck, mallet_xy)

        # --- Compute commanded position + velocity ---
        prev_commanded = commanded_xy.copy()
        commanded_vel = np.zeros(2)

        if strategy == 'STRIKE' and target_data is not None:
            # During attack trajectory: follow the precomputed path directly
            commanded_xy = clamp_to_workspace(target_data)
            if target_vel is not None:
                commanded_vel = target_vel
        else:
            # DEFEND / IDLE: smooth + ramp
            if strategy == 'DEFEND' and target_data is not None:
                raw_desired = clamp_to_workspace(np.array([DEFEND_X, target_data]))
            else:
                raw_desired = clamp_to_workspace(np.array([DEFEND_X, 0.0]))

            desired_xy = (1 - 0.15) * desired_xy + 0.15 * raw_desired

            max_step = MAX_SPEED_NORMAL * dt
            direction = desired_xy - commanded_xy
            dist = np.linalg.norm(direction)
            if dist > max_step:
                commanded_xy = commanded_xy + direction * (max_step / dist)
            else:
                commanded_xy = desired_xy.copy()

            # Velocity feedforward from the ramp (was missing — caused oscillation)
            commanded_vel = (commanded_xy - prev_commanded) / dt

        if strategy != last_strategy or tick % 40 == 0:
            print(f"  [{time.time() - start_time:.1f}s] {strategy}  "
                  f"cmd=({commanded_xy[0]:.0f}, {commanded_xy[1]:.0f})  "
                  f"mallet=({mallet_xy[0]:.0f}, {mallet_xy[1]:.0f})  "
                  f"puck=({puck.pos[0]:.0f}, {puck.pos[1]:.0f}) v=({puck.vel[0]:.0f}, {puck.vel[1]:.0f})")
            last_strategy = strategy

        # --- Compute motor target ---
        camera_xy = ctrl.ekf.position
        expected_enc = xy_to_enc(camera_xy)
        raw_offset = current_enc - expected_enc
        if smooth_offset is None:
            smooth_offset = raw_offset.copy()
        else:
            smooth_offset = (1 - OFFSET_ALPHA) * smooth_offset + OFFSET_ALPHA * raw_offset

        enc_delta = xy_to_enc(commanded_xy) - expected_enc
        target_enc = current_enc + MOVE_GAIN * enc_delta - SIGNS * TENSION_BIAS_REV

        # Smooth motor commands to filter EKF noise (defense only).
        # STRIKE bypasses — the spline trajectory is already smooth.
        if strategy == 'STRIKE':
            smooth_target_enc = target_enc.copy()
        elif smooth_target_enc is None:
            smooth_target_enc = target_enc.copy()
        else:
            smooth_target_enc = (1 - CMD_ALPHA) * smooth_target_enc + CMD_ALPHA * target_enc
            target_enc = smooth_target_enc

        # Safety clamp: limit how far any motor can move per tick.
        MAX_ENC_STEP = 0.2  # rev per tick
        enc_step = target_enc - current_enc
        clamped = np.clip(enc_step, -MAX_ENC_STEP, MAX_ENC_STEP)
        if not np.allclose(enc_step, clamped):
            print(f"  !! CLAMPED motor cmd: max delta was {np.max(np.abs(enc_step)):.3f} rev")
        target_enc = current_enc + clamped

        # Velocity feedforward: convert XY velocity to encoder velocity
        enc_vel = xy_vel_to_enc_vel(commanded_xy, commanded_vel)

        # --- Send to motors ---
        states = await asyncio.gather(*[
            ctrl.motors[mid].set_position(
                position           = target_enc[mid - 1],
                velocity           = enc_vel[mid - 1],
                kp_scale           = KP_SCALE,
                kd_scale           = KD_SCALE,
                velocity_limit     = VEL_LIMIT,
                feedforward_torque = slack_ff[mid - 1],
                maximum_torque     = MAX_TORQUE * abs(MOTOR_TORQUE_SCALE[mid]),
                watchdog_timeout   = np.nan,
                query              = True,
            )
            for mid in ids
        ])
        await asyncio.sleep(TICK_RATE)

        # Read encoders from response
        current_enc = np.array([
            states[i].values[moteus.Register.POSITION] for i in range(4)
        ])

        # Slack detection (torque-based)
        for mid in ids:
            i = mid - 1
            torque = states[i].values[moteus.Register.TORQUE]
            if abs(torque) < 0.1:
                slack_ff[i] = TENSION_TORQUE * MOTOR_TORQUE_SCALE[mid]
            else:
                slack_ff[i] = 0.0

        tick += 1

    elapsed = time.time() - start_time
    avg_hz = tick / elapsed if elapsed > 0 else 0
    print(f"Game over. {tick} ticks in {elapsed:.1f}s = {avg_hz:.0f} Hz")


########################
## MAIN               ##
########################

async def main():
    from home_motors import initialize_and_calibrate
    from vision import VisionSystem

    # Start vision
    vis = VisionSystem()
    vis.start(show_display=False)
    print("Waiting for camera...")
    await asyncio.sleep(1.0)

    # Home motors
    motors, _ = await initialize_and_calibrate()

    # Create controller + initialize EKF
    ctrl = EKFController(motors, vis)
    await ctrl.initialize_ekf()

    # Move to starting defense position
    await ctrl.move_to(np.array([DEFEND_X, 0.0]), duration=0.5)

    # Play!
    await play_air_hockey(ctrl, duration=120.0)

    vis.stop()


if __name__ == "__main__":
    asyncio.run(main())
