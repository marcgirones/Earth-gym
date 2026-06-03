"""
earth_gym_env.py
================
Gymnasium-compatible wrapper around the Earth-Gym socket environment.

This module acts as the bridge between the PPO training code (main.py /
ppo_earth_gym.py) and the Earth-Gym server (scripts/instances.py).

The wrapper translates between:
  • Gymnasium's  reset() / step(action)  interface (what the PPO expects)
  • Earth-Gym's  JSON socket protocol     (what the server speaks)

Usage
-----
    from earth_gym_env import EarthGymEnv, launch_server, ServerProcess

    # Option A — server already running (two-terminal mode)
    env = EarthGymEnv(host="localhost", port=5555)

    # Option B — launch server in-process (single-terminal / demo mode)
    proc = launch_server(conf, evpt, out)
    env  = EarthGymEnv()
    ...
    proc.stop()

Socket protocol recap
---------------------
Client → Server
    {"command": "get_next",
     "agent_id": 0,
     "action": {"d_pitch": <float>, "d_roll": <float>},   # or d_az/d_el
     "delta_time": <float>}                               # seconds

Server → Client
    {"state": {<feature>: <float>, ...}, "reward": <float|null>, "done": <bool|null>}

    done=True  → episode ended (scenario stop-time reached)
    reward=None, done=None → Δt=0 request (initial state only)
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import types
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ─────────────────────────────────────────────────────────────────────────────
# Default bounds (used when the config is not parsed at wrapper-init time)
# ─────────────────────────────────────────────────────────────────────────────

# Observation feature order must match agents-configuration.json "states_features".
# Edit this list when you change the config (see NOTE-1 in ppo_earth_gym.py).
DEFAULT_OBS_FEATURES = [
    "pitch", "roll",
    "detic_lat", "detic_lon", "detic_alt",
    "lat_1", "lon_1", "priority_1",
    "lat_2", "lon_2", "priority_2",
    "lat_3", "lon_3", "priority_3",
    "lat_4", "lon_4", "priority_4",
    "lat_5", "lon_5", "priority_5",
]

# Per-feature bounds [low, high] in the native unit of each feature.
# Inf-bounded features are clipped by ObservationNorm anyway, but keeping
# them finite helps with numerical stability from the first iteration.
_FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "pitch":      (-90.0,   90.0),
    "roll":       (-180.0, 180.0),
    "detic_lat":  (-90.0,   90.0),
    "detic_lon":  (-180.0, 180.0),
    "detic_alt":  (0.0,    2000.0),  # km
    "a":          (6378.0, 42164.0),
    "e":          (0.0,    0.9),
    "i":          (0.0,    180.0),
    "raan":       (0.0,    360.0),
    "aop":        (0.0,    360.0),
    "ta":         (0.0,    360.0),
    "az":         (0.0,    360.0),
    "el":         (-90.0,   90.0),
}
_LAT_BOUNDS     = (-90.0,   90.0)
_LON_BOUNDS     = (-180.0, 180.0)
_PRIORITY_BOUNDS = (0.0,    1.0)

# Action bounds — match max_slew_speed / max_sensor_slew in your config
DEFAULT_ACTION_FEATURES = ["d_pitch", "d_roll"]
DEFAULT_ACTION_LIMIT    = 10.0  # deg per step (max_slew_speed in config)


# ─────────────────────────────────────────────────────────────────────────────
# EarthGymEnv
# ─────────────────────────────────────────────────────────────────────────────

class EarthGymEnv(gym.Env):
    """
    Gymnasium wrapper for the Earth-Gym socket server.

    Parameters
    ----------
    host            : server hostname (default "localhost")
    port            : server port     (default 5555)
    agent_id        : index of the satellite agent to control (default 0)
    delta_time      : seconds per environment step.  One orbital period works
                      well for a ~400 km orbit (~5553 s).  Shorter values give
                      finer temporal resolution at the cost of more steps per
                      episode.
    obs_features    : ordered list of state-feature names that the server will
                      return. Must match "states_features" in the agent config.
    action_features : list of action keys.  Must match "actions_features".
    action_limit    : symmetric bound applied to every action dimension (deg).
    recv_buf        : socket receive buffer in bytes.
    timeout         : socket timeout in seconds.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        host:            str   = "localhost",
        port:            int   = 5555,
        agent_id:        int   = 0,
        # delta_time: simulated seconds per RL step.
        # Previous value 5553.5s ≈ 0.78 orbital periods — each step advanced
        # ~281° of true anomaly, making consecutive ground-track points appear
        # on opposite sides of the orbit and the lat/lon sequence look random.
        # New value: T/8 = 890.1s → 8 agent decisions per orbit.
        # 400-step episode = 50 complete orbits ≈ 4.1 simulated days.
        delta_time:      float = 50,  # T/100 ≈ 71.26 s → 100 steps per orbit
        obs_features:    list[str] | None = None,
        action_features: list[str] | None = None,
        action_limit:    float = DEFAULT_ACTION_LIMIT,
        recv_buf:        int   = 65536,
        timeout:         float = 120.0,
    ):
        super().__init__()

        self.host            = host
        self.port            = port
        self.agent_id        = agent_id
        self.delta_time      = delta_time
        self.obs_features    = obs_features or DEFAULT_OBS_FEATURES
        self.action_features = action_features or DEFAULT_ACTION_FEATURES
        self.action_limit    = action_limit
        self.recv_buf        = recv_buf
        self.timeout         = timeout

        # ── Spaces ────────────────────────────────────────────────────────
        n_obs = len(self.obs_features)
        lows, highs = self._build_obs_bounds()
        self.observation_space = spaces.Box(
            low=lows, high=highs,
            dtype=np.float32,
        )

        n_act = len(self.action_features)
        self.action_space = spaces.Box(
            low=-action_limit,
            high=action_limit,
            shape=(n_act,),
            dtype=np.float32,
        )

        # ── Internal state ─────────────────────────────────────────────────
        self._sock:      socket.socket | None = None
        self._last_obs:  np.ndarray | None    = None
        self._step_count: int                 = 0
        self._total_reward: float             = 0.0
        # Raw (un-normalised) state dict from the last server response.
        # Read by ppo_earth_gym._obs_to_dict for accurate telemetry logging.
        self.last_raw_state: dict             = {}
        # True after the very first reset() — distinguishes the initial
        # state fetch (delta_time=0) from subsequent episode resets (cmd=reset)
        self._episode_started: bool           = False

    # ── Bound helpers ─────────────────────────────────────────────────────────

    def _build_obs_bounds(self):
        lows, highs = [], []
        for feat in self.obs_features:
            if feat.startswith("lat_"):
                lows.append(_LAT_BOUNDS[0]);      highs.append(_LAT_BOUNDS[1])
            elif feat.startswith("lon_"):
                lows.append(_LON_BOUNDS[0]);      highs.append(_LON_BOUNDS[1])
            elif feat.startswith("priority_"):
                lows.append(_PRIORITY_BOUNDS[0]); highs.append(_PRIORITY_BOUNDS[1])
            else:
                b = _FEATURE_BOUNDS.get(feat, (-1e6, 1e6))
                lows.append(b[0]); highs.append(b[1])
        return np.array(lows, dtype=np.float32), np.array(highs, dtype=np.float32)

    # ── Socket helpers ────────────────────────────────────────────────────────

    def _connect(self):
        """Open (or re-open) the socket connection to the server."""
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        for attempt in range(20):
            try:
                self._sock.connect((self.host, self.port))
                return
            except (ConnectionRefusedError, OSError):
                if attempt < 19:
                    time.sleep(0.5)
        raise ConnectionRefusedError(
            f"Could not connect to Earth-Gym at {self.host}:{self.port} "
            "after 20 attempts. Is the server running?"
        )

    def _send(self, payload: dict) -> dict:
        raw = json.dumps(payload).encode()
        self._sock.sendall(raw)
        response = self._sock.recv(self.recv_buf)
        return json.loads(response.decode())

    # ── State conversion ──────────────────────────────────────────────────────

    def _state_to_obs(self, state: dict) -> np.ndarray:
        """
        Convert the server's state dict to a flat float32 numpy array.
        Features not present in the dict are filled with 0.
        """
        obs = np.zeros(len(self.obs_features), dtype=np.float32)
        for i, feat in enumerate(self.obs_features):
            if feat in state:
                obs[i] = float(state[feat])
        return obs

    def _action_to_dict(self, action: np.ndarray) -> dict:
        """
        Convert a flat numpy action to the Earth-Gym action dict.
        """
        return {
            key: float(action[i])
            for i, key in enumerate(self.action_features)
        }

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment for a new episode.

        Protocol:
        - First call ever: server has just initialised at t=start.
          Send delta_time=0 to fetch the initial state without advancing time.
        - All subsequent calls (after done=True OR mid-episode TorchRL warm-up):
          Send {"command": "reset"} so the server rewinds all simulation state
          (clocks, attitude, target zones) back to the scenario start cleanly.
          The server re-initialises and returns the new initial state.
        """
        super().reset(seed=seed)

        if self._sock is None:
            self._connect()

        self._step_count    = 0
        self._total_reward  = 0.0
        self.last_raw_state = {}

        if not self._episode_started:
            # First ever call — server is already at t=start
            resp = self._send({
                "command":    "get_next",
                "agent_id":   self.agent_id,
                "action":     {k: 0.0 for k in self.action_features},
                "delta_time": 0.0,
            })
            self._episode_started = True
        else:
            # Episode boundary or warm-up reset — rewind server to t=start
            resp = self._send({"command": "reset"})

        state = resp.get("state") or {}
        self.last_raw_state = state
        obs   = self._state_to_obs(state)
        self._last_obs = obs

        info = {"step": self._step_count, "reward": None}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Send one action to the environment and return the next transition.

        Returns
        -------
        obs         : (n_obs,) float32 observation
        reward      : scalar reward
        terminated  : True when the scenario stop-time is reached
        truncated   : always False (no external time limit imposed here)
        info        : dict with step diagnostics
        """
        action_dict = self._action_to_dict(np.asarray(action, dtype=np.float32))

        resp = self._send({
            "command":    "get_next",
            "agent_id":   self.agent_id,
            "action":     action_dict,
            "delta_time": self.delta_time,
        })

        state  = resp.get("state")  or {}
        reward = resp.get("reward") or 0.0
        done   = bool(resp.get("done", False))

        # Keep the raw server state for telemetry (bypasses ObservationNorm)
        self.last_raw_state = state

        obs = self._state_to_obs(state) if state else (
            self._last_obs if self._last_obs is not None
            else np.zeros(len(self.obs_features), dtype=np.float32)
        )
        self._last_obs      = obs
        self._step_count   += 1
        self._total_reward += float(reward)

        info = {
            "step":          self._step_count,
            "total_reward":  self._total_reward,
            "action_sent":   action_dict,
            "raw_state":     state,
        }

        return obs, float(reward), done, False, info

    def close(self):
        """Send shutdown to the server and close the socket."""
        if self._sock is not None:
            try:
                self._send({"command": "shutdown"})
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def __del__(self):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# In-process server launcher (single-terminal / Jupyter mode)
# ─────────────────────────────────────────────────────────────────────────────

class ServerProcess:
    """
    Runs the Earth-Gym server in a daemon thread so that the PPO training
    loop and the environment server can share the same process.

    Usage
    -----
        proc = ServerProcess(conf_path, evpt_path, out_path, port=5555)
        proc.start()
        env = EarthGymEnv(port=5555)
        ...
        proc.stop()
    """

    def __init__(
        self,
        conf_path:  str,
        evpt_path:  str,
        out_path:   str,
        host:       str = "localhost",
        port:       int = 5555,
    ):
        # Make sure earth-gym-oss is importable
        ROOT = os.path.dirname(os.path.abspath(__file__))
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)

        self.conf_path = conf_path
        self.evpt_path = evpt_path
        self.out_path  = out_path
        self.host      = host
        self.port      = port

        self._thread: threading.Thread | None = None
        self._error:  Exception | None        = None
        self.ready    = threading.Event()

    def start(self, timeout: float = 60.0):
        """Start the server thread and block until it is ready."""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="EarthGymServer"
        )
        self._thread.start()

        if not self.ready.wait(timeout=timeout):
            raise TimeoutError(
                "Earth-Gym server did not start within "
                f"{timeout:.0f} s. Check conf/evpt paths."
            )
        if self._error:
            raise RuntimeError(
                f"Earth-Gym server failed to start: {self._error}"
            ) from self._error

    def _run(self):
        try:
            from scripts.instances import Gym

            args = types.SimpleNamespace(
                host=self.host, port=self.port,
                conf=self.conf_path,
                evpt=self.evpt_path,
                out=self.out_path,
                pro=None,
            )
            gym_env = Gym(args=args)
            gym_env.initialize_world(self.conf_path)
            self.ready.set()
            gym_env.start(host=self.host, port=self.port)

        except Exception as exc:
            self._error = exc
            self.ready.set()

    def stop(self):
        """
        The server thread is a daemon and will be killed when the main
        process exits.  Call this explicitly only if you want a clean
        shutdown log message.
        """
        if self._thread and self._thread.is_alive():
            # The server exits cleanly once the client sends "shutdown",
            # which EarthGymEnv.close() does automatically.
            self._thread.join(timeout=5.0)
