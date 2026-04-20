"""
Cable robot motor controller for 4 mjbots moteus motors.

Motor IDs and cable mapping:
    Motor 1 = FL (Front Left)  -> corner_positions index 2 (top left)
    Motor 2 = BL (Back Left)   -> corner_positions index 0 (bottom left)
    Motor 3 = BR (Back Right)  -> corner_positions index 1 (bottom right)
    Motor 4 = FR (Front Right) -> corner_positions index 3 (top right)

All 4 motors are commanded simultaneously via a single transport.cycle()
call per tick, so CAN bus delays are shared across motors and they move
in lockstep.
"""

import asyncio
import math
import numpy as np

import moteus

from multi_move_no_substepping import (
    corner_positions,
    move_interp_accel_final,
    generate_smooth_circle,
    draw_line,
    draw_arc,
    get_quintic_coeffs_norm,
    evaluate_spline_norm,
    forward_kinematics,
)

# ── Hardware config ──────────────────────────────────────────────────────────

SPOOL_DIAMETER_MM = 75.0  # mm — adjust to your spool
SPOOL_CIRCUMFERENCE_MM = math.pi * SPOOL_DIAMETER_MM  # mm per revolution

# Motor IDs on the CAN bus.
MOTOR_FL = 1
MOTOR_BL = 2
MOTOR_BR = 3
MOTOR_FR = 4

# Map motor ID → index in the motor_history / corner_positions arrays.
# corner_positions order: 0=BL, 1=BR, 2=TL(FL), 3=TR(FR)
MOTOR_TO_CORNER = {
    MOTOR_FL: 2,  # front left  -> top left corner
    MOTOR_BL: 0,  # back left   -> bottom left corner
    MOTOR_BR: 1,  # back right  -> bottom right corner
    MOTOR_FR: 3,  # front right -> top right corner
}

# Motor servo parameters — tune these for your setup.
VELOCITY_LIMIT = 4.0       # rev/s
MAX_TORQUE = 0.5           # N·m
KD_SCALE = 2.0
WATCHDOG_TIMEOUT = 0.5     # seconds — kill motor if no command for this long
                           # set to 0.0 to disable watchdog

# Sign convention: positive dl = cable gets longer = spool pays out.
# If your motors wind cable for positive revolutions, flip the sign here.
MOTOR_SIGN = {
    MOTOR_FL:  1,
    MOTOR_BL:  1,
    MOTOR_BR:  1,
    MOTOR_FR:  1,
}


# ── Conversion helpers ───────────────────────────────────────────────────────

def mm_to_revs(mm: float) -> float:
    """Convert cable length change in mm to motor revolutions."""
    return mm / SPOOL_CIRCUMFERENCE_MM


def revs_to_mm(revs: float) -> float:
    """Convert motor revolutions to cable length change in mm."""
    return revs * SPOOL_CIRCUMFERENCE_MM


# ── Core controller ──────────────────────────────────────────────────────────

class CableRobotController:
    """Drive 4 mjbots moteus motors simultaneously along a pre-computed
    trajectory of cable-length deltas."""

    def __init__(self):
        self.transport = None
        self.controllers = {}  # motor_id -> moteus.Controller
        self.start_positions = {}  # motor_id -> starting position in revs

    async def connect(self):
        """Open CAN transport and create controller objects."""
        self.transport = moteus.Fdcanusb()
        motor_ids = [MOTOR_FL, MOTOR_BL, MOTOR_BR, MOTOR_FR]
        self.controllers = {
            mid: moteus.Controller(id=mid, transport=self.transport)
            for mid in motor_ids
        }

    async def stop_all(self):
        """Send stop command to all motors."""
        await self.transport.cycle([
            c.make_stop() for c in self.controllers.values()
        ])

    async def initialize(self):
        """Stop motors, recapture position, and record starting positions."""
        controllers = list(self.controllers.values())

        # Stop and recapture.
        await self.transport.cycle([c.make_stop() for c in controllers])
        await self.transport.cycle([
            c.make_recapture_position_velocity() for c in controllers
        ])

        # Read current positions.
        results = await self.transport.cycle([
            c.make_position(position=math.nan, query=True)
            for c in controllers
        ])
        self.start_positions = {
            r.id: r.values[moteus.Register.POSITION] for r in results
        }
        print("Motor starting positions (revs):")
        for mid in sorted(self.start_positions):
            print(f"  Motor {mid}: {self.start_positions[mid]:.4f}")

    async def run_trajectory(self, motor_history: np.ndarray, tick_rate: float):
        """Execute a trajectory defined by per-tick cable length deltas.

        Args:
            motor_history: (N, 4) array of cable length deltas in mm per tick.
                           Column order matches corner_positions: [BL, BR, FL, FR].
            tick_rate: ticks per second (e.g. 100 for 10ms per tick).
        """
        dt = 1.0 / tick_rate
        n_ticks = len(motor_history)

        # Accumulate absolute cable deltas (mm) then convert to cumulative
        # motor revolutions, respecting the motor-to-corner mapping.
        cumulative_mm = np.zeros(4)  # per corner index

        print(f"Running trajectory: {n_ticks} ticks at {tick_rate} Hz "
              f"({n_ticks / tick_rate:.2f}s)")

        loop = asyncio.get_event_loop()
        t0 = loop.time()

        for tick in range(n_ticks):
            dl = motor_history[tick]  # [BL, BR, FL, FR] in mm
            cumulative_mm += dl

            # Build commands for all 4 motors in one list — they all go
            # into a single transport.cycle() call so the CAN frames are
            # sent back-to-back with minimal inter-motor delay.
            commands = []
            for motor_id, controller in self.controllers.items():
                corner_idx = MOTOR_TO_CORNER[motor_id]
                target_revs = (
                    self.start_positions[motor_id]
                    + MOTOR_SIGN[motor_id] * mm_to_revs(cumulative_mm[corner_idx])
                )
                commands.append(
                    controller.make_position(
                        position=target_revs,
                        velocity_limit=VELOCITY_LIMIT,
                        maximum_torque=MAX_TORQUE,
                        kd_scale=KD_SCALE,
                        watchdog_timeout=WATCHDOG_TIMEOUT,
                        query=True,
                    )
                )

            results = await self.transport.cycle(commands)

            # Optional: print progress every 0.5 seconds.
            if tick % max(1, int(tick_rate * 0.5)) == 0:
                status = []
                for r in results:
                    pos = r.values[moteus.Register.POSITION]
                    delta_mm = revs_to_mm(pos - self.start_positions[r.id])
                    status.append(f"M{r.id}:{delta_mm:+.1f}mm")
                print(f"  tick {tick}/{n_ticks}  {' | '.join(status)}")

            # Sleep until the next tick's wall-clock target to prevent drift.
            target_time = t0 + (tick + 1) * dt
            now = loop.time()
            await asyncio.sleep(max(0, target_time - now))

        print("Trajectory complete.")

    async def run_trajectory_from_waypoints(self, data: np.ndarray,
                                            steps_per_sec: float):
        """Compute cable deltas from a waypoint array and execute.

        Args:
            data: (N, 4) array with columns [px, py, vx, vy] in mm and mm/s.
            steps_per_sec: trajectory tick rate.
        """
        _, motor_history = move_interp_accel_final(data, steps_per_sec)
        await self.run_trajectory(motor_history, steps_per_sec)

    async def shutdown(self):
        """Stop all motors cleanly."""
        await self.stop_all()
        print("Motors stopped.")


# ── Example trajectories ─────────────────────────────────────────────────────

def make_circle_trajectory(center=(250, 250), radius=100, duration=5.0,
                           steps_per_sec=100):
    """Generate a smooth circle trajectory and cable deltas."""
    data = generate_smooth_circle(center=center, radius=radius,
                                  duration=duration, dt=1 / steps_per_sec)
    pos_history, motor_history = move_interp_accel_final(data, steps_per_sec)
    return pos_history, motor_history


def make_line_trajectory(start, end, duration=2.0, steps_per_sec=100):
    """Generate a straight line trajectory with zero start/end velocity."""
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    zero = np.array([0.0, 0.0])
    data = draw_line(start, end, zero, zero, duration, dt=1 / steps_per_sec)
    pos_history, motor_history = move_interp_accel_final(data, steps_per_sec)
    return pos_history, motor_history


def make_arc_trajectory(center, start_pos, radians, duration=2.0,
                        steps_per_sec=100):
    """Generate an arc trajectory with zero start/end velocity."""
    center = np.array(center, dtype=float)
    start_pos = np.array(start_pos, dtype=float)
    zero = np.array([0.0, 0.0])
    data = draw_arc(center, start_pos, radians, zero, zero, duration,
                    dt=1 / steps_per_sec)
    pos_history, motor_history = move_interp_accel_final(data, steps_per_sec)
    return pos_history, motor_history


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    steps_per_sec = 100

    # Generate trajectory.
    _, motor_history = make_circle_trajectory(
        center=(250, 250), radius=100, duration=5.0,
        steps_per_sec=steps_per_sec,
    )

    # Connect and run.
    robot = CableRobotController()
    await robot.connect()
    await robot.initialize()

    try:
        await robot.run_trajectory(motor_history, tick_rate=steps_per_sec)
    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        await robot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
