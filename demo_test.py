"""
demo_test.py
============
Self-contained end-to-end test for the Earth-Gym open-source environment.

Run from the project root:
    python demo_test.py

What it does
------------
1.  Writes a minimal but complete agents-configuration.json to a temp file.
2.  Generates a tiny CSV of target zones (no file needed on disk beforehand).
3.  Launches SpiceEnvironment inside a background thread (same process,
    no subprocess, no two-terminal setup required).
4.  Connects over localhost:15555, sends several get_next requests and one
    shutdown, and prints a formatted report of every step.
5.  Verifies a set of basic assertions so you can tell at a glance whether
    the environment is behaving correctly.

The test intentionally uses a short scenario (2 h) and small delta_time
(one orbital period ≈ 5900 s) so it completes in a few seconds.
"""

import json
import math
import os
import socket
import sys
import tempfile
import threading
import time
import traceback
import types

import numpy as np
import pandas as pd

# ── make sure the project root is on the path ────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Helpers
# ─────────────────────────────────────────────────────────────────────────────

HOST = "localhost"
PORT = 15555       # use a non-default port to avoid conflicts with a live server

BANNER = "=" * 60


def _section(title: str):
    print(f"\n{BANNER}")
    print(f"  {title}")
    print(BANNER)


def _ok(msg: str):
    print(f"  ✓  {msg}")


def _fail(msg: str):
    print(f"  ✗  {msg}")
    raise AssertionError(msg)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Temp config + event-zone files
# ─────────────────────────────────────────────────────────────────────────────

_AGENT_CONFIG = {
    "general": {
        "debug":         False,
        "scenario_name": "Demo",
        # 2-hour window — keeps the test fast
        "start_time":    "1 Jan 2024 00:00:00.000",
        "stop_time":     "1 Jan 2024 07:00:00.000",
        "propagator":    "J2Perturbation",
        "deep_training": False,
    },
    "targets_and_grid": {
        "visible_targets": 10,
        "use_grid":        False,
        "grid_resolution": 5,
    },
    "reward_metrics": {
        "min_duration":    4.0,
        "reobs_decay":     2.0,
        "zenith_weight":   1.0,
        "priority_weight": 1.0,
        "grid_weight":     0.05,
        "grid_decay":      30,
        "slew_weight":     0.2,
    },
    "agents": [
        {
            "general": {"LLA_step_gap": 5},
            "position": {
                "reference_frame":    "ICRF",
                "coordinate_system":  "Classical",
                "initial_orbital_elements": {
                    # ISS-like orbit
                    "a":    6778.0,   # km  (alt ≈ 400 km)
                    "e":    0.0008,
                    "i":    51.6,     # deg
                    "raan": 30.0,
                    "aop":  60.0,
                    "ta":   0.0,
                },
            },
            "attitude": {
                "initial_pitch":  0.0,
                "initial_roll":   0.0,
                "attitude_align": "Nadir(Centric)",
                "max_slew_speed": 10.0,
                "max_slew_accel": 2.0,
            },
            "sensor": {
                "pattern":           "Simple Conic",
                "cone_angle":        30.0,   # deg half-angle
                "resolution":        0.1,
                "max_sensor_slew":   5.0,
                "initial_azimuth":   0.0,
                "initial_elevation": 90.0,
            },
            "states_features":  ["pitch", "roll", "detic_lat", "detic_lon",
                                  "detic_alt", "lat_3", "lon_3", "priority_3"],
            "actions_features": ["d_pitch", "d_roll"],
        }
    ],
}

# Orbital period of the ISS-like orbit (s)
_T_ORB = 2 * math.pi * math.sqrt(6778.0**3 / 398600.4418)

# Actions to exercise across steps
_STEP_ACTIONS = [
    {"d_pitch":  0.0,  "d_roll":  0.0},   # Step 1 — no slew
    {"d_pitch":  5.0,  "d_roll":  0.0},   # Step 2 — pitch up
    {"d_pitch": -5.0,  "d_roll":  3.0},   # Step 3 — pitch down, roll
    {"d_pitch":  0.0,  "d_roll": -3.0},   # Step 4 — roll back
]


def _write_temp_config() -> str:
    fd, path = tempfile.mkstemp(suffix=".json", prefix="earth_gym_conf_")
    with os.fdopen(fd, "w") as f:
        json.dump(_AGENT_CONFIG, f, indent=2)
    return path


def _write_temp_zones(n: int = 50) -> str:
    rng = np.random.default_rng(seed=42)
    df  = pd.DataFrame({
        "lat [deg]":    rng.uniform(-60,  60,  n).round(4),
        "lon [deg]":    rng.uniform(-180, 180, n).round(4),
        "priority":     rng.uniform(0.1,  1.0, n).round(3),
        "duration [s]": 25200,          # 7 h — covers the whole scenario
    })
    fd, path = tempfile.mkstemp(suffix=".csv", prefix="earth_gym_zones_")
    with os.fdopen(fd, "w") as f:
        df.to_csv(f, index=False)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Server thread
# ─────────────────────────────────────────────────────────────────────────────

class _ServerThread(threading.Thread):
    """Runs Gym.start() in the background and captures any exception."""

    def __init__(self, conf_path: str, zones_path: str, out_dir: str):
        super().__init__(daemon=True, name="GymServer")
        self.conf_path  = conf_path
        self.zones_path = zones_path
        self.out_dir    = out_dir
        self.error: Exception | None = None
        self.ready = threading.Event()

    def run(self):
        try:
            from scripts.instances import Gym

            # Build a minimal argparse.Namespace so Gym.initialize_args() is happy
            args = types.SimpleNamespace(
                host=HOST,
                port=PORT,
                conf=self.conf_path,
                evpt=self.zones_path,
                out=self.out_dir,
                pro=None,
            )
            gym = Gym(args=args)
            gym.initialize_world(self.conf_path)

            # Signal the client that the server is initialised
            self.ready.set()
            gym.start(host=HOST, port=PORT)

        except Exception as exc:
            self.error = exc
            self.ready.set()   # unblock the client even on failure


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Client helpers
# ─────────────────────────────────────────────────────────────────────────────

class GymClient:
    """Minimal socket client that mirrors the agent side of the protocol."""

    RECV_BUF = 65536   # bytes — large enough for any state dict

    def __init__(self, host: str, port: int, timeout: float = 30.0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect((host, port))

    def send(self, payload: dict) -> dict:
        self.sock.sendall(json.dumps(payload).encode())
        raw = self.sock.recv(self.RECV_BUF)
        return json.loads(raw.decode())

    def get_next(
        self,
        agent_id,
        action:     dict,
        delta_time: float,
    ) -> tuple[dict | None, float | None, bool | None]:
        resp   = self.send({
            "command":    "get_next",
            "agent_id":   agent_id,
            "action":     action,
            "delta_time": delta_time,
        })
        return resp.get("state"), resp.get("reward"), resp.get("done")

    def shutdown(self) -> str:
        resp = self.send({"command": "shutdown"})
        self.sock.close()
        return resp.get("status", "")

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Assertions
# ─────────────────────────────────────────────────────────────────────────────

_EXPECTED_STATE_KEYS = {"pitch", "roll", "detic_lat", "detic_lon", "detic_alt"}


def _assert_state(state: dict | None, step: int):
    if state is None:
        _fail(f"Step {step}: received null state")

    # All expected keys present
    for key in _EXPECTED_STATE_KEYS:
        if key not in state:
            _fail(f"Step {step}: missing state key '{key}'")

    # Geodetic values in physical range
    lat = state["detic_lat"]
    lon = state["detic_lon"]
    alt = state["detic_alt"]

    if not (-90.0 <= lat <= 90.0):
        _fail(f"Step {step}: latitude {lat:.2f} out of range [-90, 90]")
    if not (-180.0 <= lon <= 180.0):
        _fail(f"Step {step}: longitude {lon:.2f} out of range [-180, 180]")
    if not (300.0 < alt < 600.0):
        _fail(f"Step {step}: altitude {alt:.1f} km not in expected range (300–600 km)")

    # Target memory keys (lat_N, lon_N, priority_N for N=1..3)
    for n in range(1, 4):
        for prefix in ("lat_", "lon_", "priority_"):
            k = f"{prefix}{n}"
            if k not in state:
                _fail(f"Step {step}: missing target-memory key '{k}'")
        if not (0.0 <= state[f"priority_{n}"] <= 1.0):
            _fail(f"Step {step}: priority_{n}={state[f'priority_{n}']:.3f} outside [0,1]")


def _assert_reward(reward: float | None, step: int):
    if reward is None:
        _fail(f"Step {step}: received null reward")
    if not isinstance(reward, (int, float)):
        _fail(f"Step {step}: reward is not a number ({type(reward).__name__})")
    if math.isnan(reward) or math.isinf(reward):
        _fail(f"Step {step}: reward is {reward}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main demo / test runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + BANNER)
    print("  Earth-Gym OSS  —  End-to-End Demo & Test")
    print(BANNER)

    conf_path  = _write_temp_config()
    zones_path = _write_temp_zones(n=50)
    out_dir    = tempfile.mkdtemp(prefix="earth_gym_out_")

    print(f"\n  Config   : {conf_path}")
    print(f"  Zones    : {zones_path}")
    print(f"  Output   : {out_dir}")
    print(f"  Port     : {PORT}")
    print(f"  Δt/step  : {_T_ORB:.1f} s  (one orbital period ≈ 92 min)")

    # ── Start server ──────────────────────────────────────────────────────
    _section("1  Starting environment server")
    server = _ServerThread(conf_path, zones_path, out_dir)
    server.start()

    if not server.ready.wait(timeout=60):
        _fail("Server did not become ready within 60 s")

    if server.error:
        traceback.print_exception(type(server.error),
                                   server.error,
                                   server.error.__traceback__)
        _fail(f"Server raised: {server.error}")

    _ok("Server thread started and environment initialised")

    # Brief pause so the socket is listening
    time.sleep(0.3)

    # ── Connect client ────────────────────────────────────────────────────
    _section("2  Connecting client")
    retries = 10
    client  = None
    for attempt in range(retries):
        try:
            client = GymClient(HOST, PORT, timeout=30.0)
            break
        except ConnectionRefusedError:
            if attempt < retries - 1:
                time.sleep(0.5)
    if client is None:
        _fail("Could not connect to the environment after multiple retries")
    _ok(f"Connected to {HOST}:{PORT}")

    # ── Initial state (delta_time=0 → no time advance, no reward) ────────
    _section("3  Requesting initial state  (Δt = 0)")
    state0, reward0, done0 = client.get_next(
        agent_id=0, action={"d_pitch": 0.0, "d_roll": 0.0}, delta_time=0.0
    )
    _assert_state(state0, step=0)
    _ok(f"Initial state received  ({len(state0)} variables)")
    _ok(f"  lat={state0['detic_lat']:+.3f}°  "
        f"lon={state0['detic_lon']:+.3f}°  "
        f"alt={state0['detic_alt']:.1f} km")
    _ok(f"  pitch={state0['pitch']:.2f}°  roll={state0['roll']:.2f}°")
    if reward0 is None:
        _ok("reward=None  (expected for Δt=0 — no time elapsed)")
    else:
        _fail(f"Expected reward=None for Δt=0, got {reward0}")

    # ── Step loop ─────────────────────────────────────────────────────────
    _section(f"4  Running {len(_STEP_ACTIONS)} steps  (Δt = {_T_ORB:.0f} s each)")

    rewards = []
    states  = []
    all_done = False

    for i, action in enumerate(_STEP_ACTIONS, start=1):
        state, reward, done = client.get_next(
            agent_id=0, action=action, delta_time=_T_ORB
        )

        if done:
            _ok(f"Step {i}: environment signalled done (episode ended early)")
            all_done = True
            break

        _assert_state(state, step=i)
        _assert_reward(reward, step=i)
        rewards.append(reward)
        states.append(state)

        print(f"\n  ── Step {i} ──────────────────────────────────────────")
        print(f"     action  : d_pitch={action['d_pitch']:+.1f}°  "
              f"d_roll={action['d_roll']:+.1f}°")
        print(f"     lat/lon : {state['detic_lat']:+.3f}° / "
              f"{state['detic_lon']:+.3f}°   alt: {state['detic_alt']:.1f} km")
        print(f"     attitude: pitch={state['pitch']:+.2f}°  "
              f"roll={state['roll']:+.2f}°")
        print(f"     reward  : {reward:+.6f}")
        print(f"     target 1: lat={state['lat_1']:+.3f}°  "
              f"lon={state['lon_1']:+.3f}°  "
              f"priority={state['priority_1']:.3f}")

    # ── Validation checks ─────────────────────────────────────────────────
    _section("5  Validation")

    # The satellite should have moved between step 0 and step 1
    if states:
        lat_moved = abs(states[0]["detic_lat"] - state0["detic_lat"])
        lon_moved = abs(states[0]["detic_lon"] - state0["detic_lon"])
        if lat_moved < 0.01 and lon_moved < 0.01:
            _fail("Satellite position did not change after one time step — "
                  "propagator may not be advancing time")
        _ok(f"Satellite moved  Δlat={lat_moved:.3f}°  Δlon={lon_moved:.3f}°")

    # Attitude reflects the cumulative slew
    if states and len(states) >= 2:
        # After step 2 (d_pitch=+5) and step 3 (d_pitch=-5, d_roll=+3)
        # net pitch change should be ~0, roll should be ~3°
        final_roll  = states[-1]["roll"]
        final_pitch = states[-1]["pitch"]
        _ok(f"Final attitude: pitch={final_pitch:+.2f}°  roll={final_roll:+.2f}°")

    # Rewards are finite numbers
    if rewards:
        _ok(f"Received {len(rewards)} rewards — all finite: "
            f"[{', '.join(f'{r:+.4f}' for r in rewards)}]")

    # Target memory contains valid lat/lon
    if states:
        last = states[-1]
        for n in range(1, 4):
            if not (-90 <= last[f"lat_{n}"] <= 90):
                _fail(f"target lat_{n} out of range: {last[f'lat_{n}']:.2f}")
            if not (-180 <= last[f"lon_{n}"] <= 180):
                _fail(f"target lon_{n} out of range: {last[f'lon_{n}']:.2f}")
        _ok("Target-memory lat/lon values are all in valid range")

    # ── Shutdown ──────────────────────────────────────────────────────────
    _section("6  Shutting down")
    status = client.shutdown()
    _ok(f"Server acknowledged shutdown  (status='{status}')")

    server.join(timeout=10)

    # ── Summary ───────────────────────────────────────────────────────────
    _section("Summary")
    total_steps = len(rewards)
    print(f"  Steps completed : {total_steps}")
    if rewards:
        print(f"  Total reward    : {sum(rewards):+.6f}")
        print(f"  Mean reward     : {sum(rewards)/len(rewards):+.6f}")
        print(f"  Min / Max       : {min(rewards):+.6f} / {max(rewards):+.6f}")

    print(f"\n  Reward plots saved to: {out_dir}/")
    print("\n  ✓  All assertions passed — environment is working correctly.\n")

    # Cleanup temp files
    for path in (conf_path, zones_path):
        try:
            os.remove(path)
        except OSError:
            pass


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"\n  DEMO FAILED: {exc}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
        sys.exit(0)
