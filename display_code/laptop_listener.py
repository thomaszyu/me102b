"""
UART link between the laptop and the ESP32 display.

Exposes a single transport, SerialLink, with this API:
    wait_for_start(timeout=None) -> "easy" | "medium" | "hard" | None
    check_command()              -> "pause" | "stop" | "start" | None
    send_state(...)              -> push puck/mallet/strategy/score
    send_score(score_us, score_them)

Usage (standalone simulator):
    python laptop_listener.py --serial /dev/cu.usbserial-XXXX

Integration with your robot code:
    from display_code.laptop_listener import SerialLink

    link = SerialLink("/dev/cu.usbserial-XXXX")
    difficulty = link.wait_for_start()

    link.send_state(
        puck_x, puck_y, puck_vx, puck_vy, puck_valid,
        mallet_x, mallet_y, mallet_valid,
        strategy="DEFEND", score_us=0, score_them=0,
    )

Requires pyserial:  pip install pyserial
"""

import json
import threading
import time
import math
import queue

try:
    import serial as _pyserial  # pyserial
except ImportError:
    _pyserial = None


SERIAL_BAUD = 115200


def _build_state_msg(puck_x=0, puck_y=0, puck_vx=0, puck_vy=0, puck_valid=False,
                     mallet_x=0, mallet_y=0, mallet_valid=False,
                     strategy="IDLE", score_us=0, score_them=0,
                     trajectory=None, contact_idx=-1):
    """Build the state JSON dict shared by all transports."""
    msg = {
        "type": "state",
        "px": round(puck_x, 1),
        "py": round(puck_y, 1),
        "pvx": round(puck_vx, 1),
        "pvy": round(puck_vy, 1),
        "pv": 1 if puck_valid else 0,
        "mx": round(mallet_x, 1),
        "my": round(mallet_y, 1),
        "mv": 1 if mallet_valid else 0,
        "strategy": strategy,
        "su": score_us,
        "st": score_them,
    }

    if trajectory and len(trajectory) >= 2:
        max_pts = 20
        raw = trajectory
        if len(raw) > max_pts:
            step = len(raw) / max_pts
            sampled = [raw[int(i * step)] for i in range(max_pts)]
            if contact_idx >= 0:
                contact_idx = int(contact_idx / step)
        else:
            sampled = raw
        flat = []
        for pt in sampled:
            flat.append(round(float(pt[0]), 1))
            flat.append(round(float(pt[1]), 1))
        msg["traj"] = flat
        msg["tc"] = contact_idx

    return msg


class SerialLink:
    """Two-way UART link with the ESP32 display.

    Newline-delimited JSON is exchanged over the same USB serial port the
    Arduino IDE Serial Monitor uses; debug prints from the firmware are
    printed locally with a "[esp32]" prefix and skipped by the parser.

    Args:
        port:   serial device path, e.g. "/dev/cu.usbserial-0001" (mac),
                "/dev/ttyUSB0" (Linux), "COM5" (Windows).
        baud:   must match SERIAL_BAUD on the ESP32 (default 115200).
    """

    def __init__(self, port, baud=SERIAL_BAUD):
        if _pyserial is None:
            raise ImportError(
                "pyserial is not installed. Run: pip install pyserial"
            )
        self.port = port
        self.baud = baud
        # Short read timeout so the reader thread polls frequently without
        # busy-spinning when no data is available.
        self.ser = _pyserial.Serial(port, baud, timeout=0.05)

        self._inbox: "queue.Queue[dict]" = queue.Queue()
        self._tx_lock = threading.Lock()
        self._stop = threading.Event()

        # TX side runs on a dedicated thread so that game-loop callers never
        # block on serial I/O. Two channels:
        #   - _tx_queue: small FIFO for control messages (start/stop/pause/
        #     score). Bounded with drop-oldest to bound memory.
        #   - _latest_state / _state_event: single-slot buffer for "state"
        #     messages. Producers overwrite; the writer always sends the
        #     newest one. This naturally throttles state updates to whatever
        #     the link can carry, without ever stalling the producer.
        self._tx_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=16)
        self._latest_state: "bytes | None" = None
        self._state_lock = threading.Lock()
        self._tx_event = threading.Event()

        self._reader = threading.Thread(
            target=self._reader_loop, name="SerialLinkReader", daemon=True
        )
        self._writer = threading.Thread(
            target=self._writer_loop, name="SerialLinkWriter", daemon=True
        )
        self._reader.start()
        self._writer.start()

        # Most ESP32 dev boards reset when the USB serial port opens (DTR/RTS
        # toggle). Give the firmware a moment to come up before we expect
        # messages.
        time.sleep(1.5)
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass

    def _reader_loop(self):
        buf = bytearray()
        while not self._stop.is_set():
            try:
                chunk = self.ser.read(512)
            except Exception:
                time.sleep(0.05)
                continue
            if not chunk:
                continue
            buf.extend(chunk)
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                line = bytes(buf[:nl]).strip()
                del buf[: nl + 1]
                if not line:
                    continue
                if line.startswith(b"{"):
                    try:
                        msg = json.loads(line.decode("utf-8", "replace"))
                    except json.JSONDecodeError:
                        continue
                    if isinstance(msg, dict):
                        self._inbox.put(msg)
                else:
                    # Firmware debug output — surface it but don't try to parse.
                    try:
                        print(f"[esp32] {line.decode('utf-8', 'replace')}")
                    except Exception:
                        pass

    def wait_for_start(self, timeout=None):
        """Block until the ESP32 sends a start command. Returns difficulty string."""
        deadline = None if timeout is None else time.time() + timeout
        while True:
            try:
                wait = None if deadline is None else max(
                    0.0, deadline - time.time())
                msg = self._inbox.get(timeout=wait)
            except queue.Empty:
                return None
            if msg.get("type") == "start":
                diff = msg.get("difficulty", "medium")
                print(f"[display] Game start — difficulty={diff} (serial)")
                return diff

    def check_command(self):
        """Non-blocking check for commands from the ESP32 (pause/stop/start).
        Returns 'pause', 'stop', 'start', or None."""
        try:
            msg = self._inbox.get_nowait()
        except queue.Empty:
            return None
        return msg.get("type")

    def _writer_loop(self):
        """Background thread that performs all serial writes."""
        while not self._stop.is_set():
            # 1) Drain control messages first (start/stop/pause/score).
            try:
                line = self._tx_queue.get_nowait()
                self._do_write(line)
                continue
            except queue.Empty:
                pass

            # 2) If a state update is pending, send the latest one.
            with self._state_lock:
                line = self._latest_state
                self._latest_state = None
            if line is not None:
                self._do_write(line)
                continue

            # 3) Nothing to do — sleep until something is queued.
            self._tx_event.wait(timeout=0.05)
            self._tx_event.clear()

    def _do_write(self, line: bytes):
        """Write a single line to the serial port; tolerate disconnects."""
        with self._tx_lock:
            try:
                self.ser.write(line)
            except Exception:
                # Port may have been pulled / closed; ignore and let the
                # next iteration retry or surface the error elsewhere.
                pass

    def send(self, msg_dict):
        """Queue a JSON message for transmission. Non-blocking.

        State messages overwrite each other (latest-only) so the producer is
        never throttled by the serial link. Other messages go through a
        bounded FIFO; if it fills, the oldest pending message is dropped to
        make room.
        """
        line = (json.dumps(msg_dict, separators=(",", ":")) + "\n").encode()

        if msg_dict.get("type") == "state":
            with self._state_lock:
                self._latest_state = line
            self._tx_event.set()
            return

        try:
            self._tx_queue.put_nowait(line)
        except queue.Full:
            try:
                self._tx_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._tx_queue.put_nowait(line)
            except queue.Full:
                pass
        self._tx_event.set()

    def send_state(self, **kwargs):
        """Send current game state to the ESP32 display. See _build_state_msg.

        Non-blocking. If the link can't keep up, intermediate state messages
        are dropped and only the most recent one is delivered.
        """
        self.send(_build_state_msg(**kwargs))

    def send_score(self, score_us, score_them):
        """Send just a score update."""
        self.send({"type": "score", "su": score_us, "st": score_them})

    def close(self):
        self._stop.set()
        self._tx_event.set()
        try:
            self.ser.close()
        except Exception:
            pass


# --- Standalone test: puck physics simulation with mallet attacks ---
if __name__ == "__main__":
    import random
    import argparse

    parser = argparse.ArgumentParser(
        description="Air hockey display link tester (UART).",
    )
    parser.add_argument(
        "--serial",
        metavar="PORT",
        required=True,
        help="Serial port the ESP32 is connected to "
             "(e.g. /dev/cu.usbserial-0001, /dev/ttyUSB0, COM5).",
    )
    parser.add_argument(
        "--baud", type=int, default=SERIAL_BAUD,
        help=f"Serial baud rate (default {SERIAL_BAUD}).",
    )
    args = parser.parse_args()

    # Table constants (matches air_hockey_player.py)
    TBL_X_MIN, TBL_X_MAX = -273.0, 273.0
    TBL_Y_MIN, TBL_Y_MAX = -240.0, 240.0
    PUCK_RADIUS = 25.0
    MALLET_RADIUS = 30.0
    CONTACT_DIST = PUCK_RADIUS + MALLET_RADIUS
    WALL_X_MIN = TBL_X_MIN + PUCK_RADIUS
    WALL_X_MAX = TBL_X_MAX - PUCK_RADIUS
    WALL_Y_MIN = TBL_Y_MIN + PUCK_RADIUS
    WALL_Y_MAX = TBL_Y_MAX - PUCK_RADIUS
    DEFEND_X = TBL_X_MIN + 120.0   # -153
    MALLET_Y_MIN = TBL_Y_MIN + 30
    MALLET_Y_MAX = TBL_Y_MAX - 30
    GOAL_HALF = 80.0
    FRICTION = 0.993          # faster decay so attacks trigger sooner
    MALLET_DEFEND_SPEED = 600.0  # mm/s while defending
    MALLET_ATTACK_SPEED = 400.0  # mm/s while attacking (slow enough to see)

    print(
        f"Listening for display commands on serial {args.serial} @ {args.baud}...")
    link = SerialLink(args.serial, args.baud)

    difficulty = link.wait_for_start()
    print(f">>> Game started with difficulty: {difficulty}")
    print("Simulating puck physics (Ctrl+C to stop)...")

    # Puck state
    px, py = 100.0, 0.0
    pvx = -200.0 + random.uniform(-50, 50)
    pvy = 150.0 + random.uniform(-100, 100)

    # Mallet state (list so nested functions can mutate)
    mallet = [DEFEND_X, 0.0]
    score_us, score_them = 0, 0

    # Attack state machine
    attack_traj = None       # list of (x,y) waypoints
    attack_tick = 0          # current position along trajectory
    attack_contact_idx = -1  # which waypoint is the contact point
    strat = "DEFEND"

    dt = 0.02  # 50 Hz

    def respawn_puck():
        return (0.0, 0.0,
                random.choice([-1, 1]) * random.uniform(150, 300),
                random.uniform(-150, 150))

    def clamp_mallet(x, y):
        return (max(TBL_X_MIN + 30, min(TBL_X_MAX - 30, x)),
                max(MALLET_Y_MIN, min(MALLET_Y_MAX, y)))

    def plan_attack(mx, my, px, py):
        """Build a trajectory: approach -> contact -> follow-through -> return."""
        goal_y = random.uniform(-GOAL_HALF * 0.6, GOAL_HALF * 0.6)
        pts = []
        n_approach = 20
        n_strike = 10
        n_return = 15

        # Approach: mallet to just behind puck
        for i in range(n_approach):
            t = i / (n_approach - 1)
            pts.append(clamp_mallet(
                mx + (px - mx) * t,
                my + (py - my) * t
            ))

        contact_idx = len(pts) - 1

        # Follow-through: push past puck toward goal
        strike_dx = TBL_X_MAX - px
        strike_dy = goal_y - py
        mag = math.sqrt(strike_dx**2 + strike_dy**2)
        if mag > 0:
            strike_dx /= mag
            strike_dy /= mag
        for i in range(1, n_strike + 1):
            t = i / n_strike
            pts.append(clamp_mallet(
                px + strike_dx * 80 * t,
                py + strike_dy * 80 * t
            ))

        # Return to defend
        ret_x, ret_y = pts[-1]
        for i in range(1, n_return + 1):
            t = i / n_return
            pts.append(clamp_mallet(
                ret_x + (DEFEND_X - ret_x) * t,
                ret_y + (0 - ret_y) * t
            ))

        return pts, contact_idx

    def move_mallet_toward(tx, ty, speed):
        """Move mallet toward target at given speed, return True if arrived."""
        ddx = tx - mallet[0]
        ddy = ty - mallet[1]
        dist = math.sqrt(ddx**2 + ddy**2)
        max_step = speed * dt
        if dist <= max_step:
            mallet[0], mallet[1] = tx, ty
            return True
        else:
            mallet[0] += ddx / dist * max_step
            mallet[1] += ddy / dist * max_step
            return False

    try:
        while True:
            # --- Puck physics ---
            px += pvx * dt
            py += pvy * dt
            pvx *= FRICTION
            pvy *= FRICTION

            # Wall bounces
            if py <= WALL_Y_MIN:
                py = WALL_Y_MIN
                pvy = abs(pvy)
            elif py >= WALL_Y_MAX:
                py = WALL_Y_MAX
                pvy = -abs(pvy)

            # Goals
            scored = False
            if px <= WALL_X_MIN:
                if abs(py) < GOAL_HALF:
                    score_them += 1
                    print(f"  GOAL! Them: {score_us}-{score_them}")
                    scored = True
                else:
                    px = WALL_X_MIN
                    pvx = abs(pvx)
            elif px >= WALL_X_MAX:
                if abs(py) < GOAL_HALF:
                    score_us += 1
                    print(f"  GOAL! Us: {score_us}-{score_them}")
                    scored = True
                else:
                    px = WALL_X_MAX
                    pvx = -abs(pvx)

            if scored:
                px, py, pvx, pvy = respawn_puck()
                attack_traj = None
                strat = "DEFEND"

            # --- Mallet-puck collision ---
            dist = math.sqrt((px - mallet[0])**2 + (py - mallet[1])**2)
            if dist < CONTACT_DIST:
                nx = (px - mallet[0]) / max(dist, 0.1)
                ny = (py - mallet[1]) / max(dist, 0.1)
                # Reflect puck velocity + add mallet momentum
                pvx = abs(pvx) * nx + 200 * nx
                pvy = abs(pvy) * ny + 100 * ny
                px = mallet[0] + nx * (CONTACT_DIST + 1)
                py = mallet[1] + ny * (CONTACT_DIST + 1)

            # --- Mallet AI state machine ---
            speed = math.sqrt(pvx**2 + pvy**2)
            puck_in_our_half = px < 0

            if attack_traj is not None:
                # Following attack trajectory
                if attack_tick < len(attack_traj):
                    tx, ty = attack_traj[attack_tick]
                    arrived = move_mallet_toward(tx, ty, MALLET_ATTACK_SPEED)
                    if arrived:
                        attack_tick += 1
                    if attack_tick <= attack_contact_idx:
                        strat = "WINDUP"
                    else:
                        strat = "STRIKE"
                else:
                    # Attack done
                    attack_traj = None
                    strat = "DEFEND"
                    print("  Attack complete, returning to defend")
            else:
                # Defend: track puck Y on defend line
                target_y = max(MALLET_Y_MIN, min(MALLET_Y_MAX, py * 0.7))
                move_mallet_toward(DEFEND_X, target_y, MALLET_DEFEND_SPEED)
                strat = "DEFEND"

                # Start attack if puck is slow-ish and in our half
                if puck_in_our_half and speed < 350:
                    attack_traj, attack_contact_idx = plan_attack(
                        mallet[0], mallet[1], px, py)
                    attack_tick = 0
                    strat = "WINDUP"
                    print(
                        f"  Attack! puck=({px:.0f},{py:.0f}) speed={speed:.0f}")

            # --- Send to display ---
            link.send_state(
                puck_x=px, puck_y=py, puck_vx=pvx, puck_vy=pvy, puck_valid=True,
                mallet_x=mallet[0], mallet_y=mallet[1], mallet_valid=True,
                strategy=strat, score_us=score_us, score_them=score_them,
                trajectory=attack_traj, contact_idx=attack_contact_idx,
            )
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nStopped.")
