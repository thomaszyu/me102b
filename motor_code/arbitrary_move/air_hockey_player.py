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
from kinematics_utils import xy_to_enc
from motor_utils_for_arbitrary_move import read_encoders
from ekf_controller import EKFController, MalletEKF


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

# Puck center bounce limits (wall + puck radius)
PUCK_X_MIN = TABLE_X_MIN + PUCK_RADIUS
PUCK_X_MAX = TABLE_X_MAX - PUCK_RADIUS
PUCK_Y_MIN = TABLE_Y_MIN + PUCK_RADIUS
PUCK_Y_MAX = TABLE_Y_MAX - PUCK_RADIUS

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
ATTACK_ZONE_X = -80.0      # puck must be left of this X to attempt a strike


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


def predict_intercept_y(puck_pos, puck_vel, target_x):
    """
    Find the Y position where the puck crosses target_x,
    accounting for wall bounces. Returns None if puck won't reach target_x.
    """
    x, y = float(puck_pos[0]), float(puck_pos[1])
    vx, vy = float(puck_vel[0]), float(puck_vel[1])

    # Puck must be moving toward target_x
    if abs(vx) < 5.0:
        return None
    if (target_x < x and vx > 0) or (target_x > x and vx < 0):
        return None

    # Simulate until crossing target_x (max 5 seconds)
    dt = 0.005
    for _ in range(1000):
        x += vx * dt
        y += vy * dt

        # Wall bounces (Y only — X walls are goals)
        if y < PUCK_Y_MIN:
            y = 2 * PUCK_Y_MIN - y
            vy = -vy
        elif y > PUCK_Y_MAX:
            y = 2 * PUCK_Y_MAX - y
            vy = -vy

        # Check crossing
        if (vx < 0 and x <= target_x) or (vx > 0 and x >= target_x):
            return np.clip(y, MALLET_Y_MIN, MALLET_Y_MAX)

    return None


########################
## STRATEGY           ##
########################

# Attack tuning
WINDUP_OFFSET = 40.0       # mm behind puck (shorter = faster setup)
WINDUP_CLOSE_THRESH = 30.0 # mm — close enough to start striking (was 25)
STRIKE_THROUGH = 50.0      # mm past the puck to aim the strike
STRIKE_SPEED_LIMIT = 400.0 # mm/s — max mallet speed during strike

class AttackState:
    """Tracks the phase of an attack sequence."""
    def __init__(self):
        self.phase = None        # None, 'WINDUP', 'STRIKE'
        self.strike_target = None
        self.strike_dir = None

    def reset(self):
        self.phase = None
        self.strike_target = None
        self.strike_dir = None


_attack = AttackState()
_last_strategy = 'IDLE'


def decide_strategy(puck, mallet_xy):
    """
    Decide what to do based on puck state.
    Returns: ('DEFEND', target_y) or ('WINDUP', target_xy)
             or ('STRIKE', target_xy) or ('IDLE', None)
    """
    global _attack, _last_strategy

    if not puck.initialized or not puck.visible:
        _attack.reset()
        return 'IDLE', None

    px, py = puck.pos
    vx, vy = puck.vel
    puck_on_our_side = px < 0

    # --- DEFEND: puck coming toward our goal (overrides everything) ---
    # Hysteresis: easier to enter DEFEND than to leave it
    defend_thresh = PUCK_COMING_THRESH
    if _last_strategy == 'DEFEND':
        defend_thresh = PUCK_COMING_THRESH + 40

    if vx < defend_thresh:
        _attack.reset()
        _last_strategy = 'DEFEND'
        intercept_y = predict_intercept_y(puck.pos, puck.vel, DEFEND_X)
        if intercept_y is not None:
            return 'DEFEND', intercept_y
        return 'DEFEND', np.clip(py, MALLET_Y_MIN, MALLET_Y_MAX)

    # --- If mid-STRIKE, commit — give target just ahead of mallet in strike dir ---
    if _attack.phase == 'STRIKE':
        mallet_to_puck = puck.pos - mallet_xy
        if np.dot(mallet_to_puck, _attack.strike_dir) < -20:
            # Past the puck — done
            _attack.reset()
            _last_strategy = 'DEFEND'
            return 'DEFEND', np.clip(py, MALLET_Y_MIN, MALLET_Y_MAX)
        # Target = mallet position + small step in strike direction
        # commanded_xy ramp does the speed limiting
        step_target = mallet_xy + _attack.strike_dir * STRIKE_THROUGH
        step_target = clamp_to_workspace(step_target)
        _last_strategy = 'STRIKE'
        return 'STRIKE', step_target

    # --- ATTACK: puck on our side, slow, and Y is aligned ---
    # No WINDUP — stay on defend line, match Y, then charge forward
    if px < ATTACK_ZONE_X and np.linalg.norm(puck.vel) < PUCK_SLOW_THRESH:
        y_diff = abs(mallet_xy[1] - py)
        if y_diff < WINDUP_CLOSE_THRESH:
            # Aligned — initiate strike
            strike_dir = np.array([1.0, (OPPONENT_GOAL_CENTER_Y - py) / (OPPONENT_GOAL_X - px + 1e-9)])
            strike_dir = strike_dir / (np.linalg.norm(strike_dir) + 1e-9)
            _attack.phase = 'STRIKE'
            _attack.strike_dir = strike_dir.copy()
            # First target = small step ahead, not the final destination
            step_target = mallet_xy + strike_dir * STRIKE_THROUGH
            _attack.strike_target = clamp_to_workspace(step_target)
            _last_strategy = 'STRIKE'
            return 'STRIKE', _attack.strike_target
        else:
            _last_strategy = 'DEFEND'
            return 'DEFEND', np.clip(py, MALLET_Y_MIN, MALLET_Y_MAX)

    # --- DEFEND (passive): puck on our side but not slow enough to attack ---
    if puck_on_our_side:
        _attack.reset()
        _last_strategy = 'DEFEND'
        return 'DEFEND', np.clip(py, MALLET_Y_MIN, MALLET_Y_MAX)

    # --- IDLE: puck clearly on opponent's side ---
    _attack.reset()
    _last_strategy = 'IDLE'
    return 'IDLE', None


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

    # Commanded position — ramps toward target at max_speed, never jumps
    commanded_xy = ctrl.ekf.position.copy()
    MAX_SPEED_NORMAL = 400.0   # mm/s for defend/windup/idle
    MAX_SPEED_STRIKE = 400.0   # mm/s for strikes
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
        strategy, target_data = decide_strategy(puck, mallet_xy)

        if strategy == 'DEFEND' and target_data is not None:
            desired_xy = clamp_to_workspace(np.array([DEFEND_X, target_data]))
        elif strategy in ('WINDUP', 'STRIKE') and target_data is not None:
            desired_xy = clamp_to_workspace(target_data)
        else:
            desired_xy = clamp_to_workspace(np.array([DEFEND_X, 0.0]))

        # --- Ramp commanded_xy toward desired_xy at max speed ---
        max_speed = MAX_SPEED_STRIKE if strategy == 'STRIKE' else MAX_SPEED_NORMAL
        max_step = max_speed * dt
        direction = desired_xy - commanded_xy
        dist = np.linalg.norm(direction)
        if dist > max_step:
            commanded_xy = commanded_xy + direction * (max_step / dist)
        else:
            commanded_xy = desired_xy.copy()

        if strategy != last_strategy or tick % 40 == 0:
            print(f"  [{time.time() - start_time:.1f}s] {strategy}  "
                  f"cmd=({commanded_xy[0]:.0f}, {commanded_xy[1]:.0f})  "
                  f"mallet=({mallet_xy[0]:.0f}, {mallet_xy[1]:.0f})  "
                  f"puck=({puck.pos[0]:.0f}, {puck.pos[1]:.0f}) v=({puck.vel[0]:.0f}, {puck.vel[1]:.0f})")
            last_strategy = strategy

        # --- Compute motor target (same as puck_tracker) ---
        camera_xy = ctrl.ekf.position
        expected_enc = xy_to_enc(camera_xy)
        raw_offset = current_enc - expected_enc
        if smooth_offset is None:
            smooth_offset = raw_offset.copy()
        else:
            smooth_offset = (1 - OFFSET_ALPHA) * smooth_offset + OFFSET_ALPHA * raw_offset

        enc_delta = xy_to_enc(commanded_xy) - expected_enc
        target_enc = current_enc + MOVE_GAIN * enc_delta - SIGNS * TENSION_BIAS_REV

        # --- Send to motors ---
        states = await asyncio.gather(*[
            ctrl.motors[mid].set_position(
                position           = target_enc[mid - 1],
                velocity           = 0.0,
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

        # Slack detection
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
    await ctrl.move_to(np.array([DEFEND_X, 0.0]), duration=1.5)

    # Play!
    await play_air_hockey(ctrl, duration=120.0)

    vis.stop()


if __name__ == "__main__":
    asyncio.run(main())
