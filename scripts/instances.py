"""
scripts/instances.py
====================
Drop-in replacement for the original STK-based instances.py.

STKEngine / STKEnvironment  →  SpiceEnvironment
All STK COM-object calls replaced with:
  - OrbitalPropagator  (scripts/propagator.py)  for orbit propagation
  - AccessChecker      (scripts/utils.py)        for sensor visibility
  - coordinates.py     for frame transforms
  - spiceypy           for time conversion / geodetic transforms
    (transparent analytic fallback when SPICE kernels are absent)
"""

from __future__ import annotations

import json
import os
import socket
import struct
import numpy as np
import psutil
import pandas as pd
from time import perf_counter
from typing import Optional

from scripts.propagator_base import make_propagator
from scripts.coordinates import (
    stk_date_to_et,
    eci_to_ecef,
    ecef_to_geodetic,
    load_spice_kernels,
    download_kernels,
)
from scripts.utils import (
    DataFromJSON,
    DateManager,
    AttitudeManager,
    SensorManager,
    FeaturesManager,
    TargetManager,
    AccessChecker,
    GridManager,
    Rewarder,
    Plotter,
)

# ── Earth radius (km) ─────────────────────────────────────────────────────────
RT = 6371.0


# ─────────────────────────────────────────────────────────────────────────────
# Gym  (socket-based entry point — interface unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class Gym:
    """
    Socket-based RL environment server.

    The public API is identical to the original STK-based Gym; only the
    internal environment class has changed (SpiceEnvironment instead of
    STKEnvironment).
    """

    def __init__(self, args):
        self.initialize_args(args)
        self.running           = True
        self.shutdown_complete = False

    def initialize_args(self, args):
        self.host = args.host
        self.port = args.port

        if args.conf is None:
            raise ValueError("Configuration file not specified in launch.json.")
        self.conf_file_path = args.conf

        if args.evpt is None:
            raise ValueError("Events zones file not specified in launch.json.")
        self.evpt_file_path = args.evpt

        if args.out is None:
            raise ValueError("Output folder not specified in launch.json.")
        self.out_folder_path = args.out

    def initialize_world(self, file_path: str):
        with open(file_path, "r") as f:
            agents_config = json.load(f)

        if not agents_config:
            raise ValueError("Agent configuration is empty.")

        self.env = SpiceEnvironment(
            DataFromJSON(agents_config, "configuration").get_dict(),
            self.evpt_file_path,
            self.out_folder_path,
        )

    def get_next_state_and_reward(self, agent_id, action, delta_time):
        return self.env.step(agent_id, action, delta_time)

    def shutdown(self):
        self.generate_output()
        process      = psutil.Process(os.getpid())
        memory_used  = process.memory_info().rss
        print(f"Memory used: {memory_used / (1024 ** 2):.2f} MB")
        self.shutdown_complete = True

    def generate_output(self):
        self.env.plotter.plot_all()

    def handle_request(self, request: str) -> str:
        request_data = json.loads(request)
        print(f"Received request: {request_data}")

        cmd = request_data["command"]
        if cmd == "get_next":
            state, reward, done = self.get_next_state_and_reward(
                request_data["agent_id"],
                request_data["action"],
                request_data["delta_time"],
            )
            return json.dumps({
                "state":         state,
                "reward":        reward,
                "done":          done,
                "access_events": getattr(self.env, "_last_access_events", []),
            })
        elif cmd == "get_targets":
            # Return the full active target pool so the dashboard can
            # display all 100 targets on the globe, not just the 5 FoR slots.
            rows = []
            for _, row in self.env.target_mg.df.iterrows():
                rows.append({
                    "name":     str(row["name"]),
                    "lat":      float(row["lat [deg]"]),
                    "lon":      float(row["lon [deg]"]),
                    "priority": float(row["priority"]),
                    "n_obs":    int(row["n_obs"]),
                })
            return json.dumps({"targets_all": rows})
        elif cmd == "reset":
            # Re-initialise the environment from scratch and return initial state.
            # Called by EarthGymEnv.reset() after an episode ends (done=True).
            state = self.env.reset()
            return json.dumps({"state": state, "reward": None, "done": False})
        elif cmd == "shutdown":
            self.shutdown()
            self.running = False
            return json.dumps({"status": "shutdown_complete"})
        else:
            raise ValueError(
                f"Invalid command '{cmd}'. Use 'get_next', 'reset', or 'shutdown'."
            )

    def start(self, host: str = "localhost", port: int = 5555):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(1)
        print("Gym environment started. Waiting for connections…")

        conn, addr = server_socket.accept()
        print(f"Connected to: {addr}")

        self.initialize_world(self.conf_file_path)

        while self.running:
            # ── Length-prefix framing ──────────────────────────────────────
            # Read exactly 4 bytes to get the message length, then read
            # exactly that many bytes for the JSON body.
            # The old code used conn.recv(1024) which has two failure modes:
            #   1. Back-to-back PPO requests arrive concatenated → json.loads
            #      sees two JSON objects and raises JSONDecodeError.
            #   2. A request larger than 1024 bytes is truncated → same error.
            # The matching length-prefix send on the response side ensures the
            # client's _recv_exactly loop always gets the complete message.
            header = self._recv_exactly(conn, 4)
            if not header:
                break
            length = struct.unpack(">I", header)[0]
            data = self._recv_exactly(conn, length).decode()
            if not data:
                break
            response = self.handle_request(data)
            resp_bytes = response.encode()
            conn.sendall(struct.pack(">I", len(resp_bytes)) + resp_bytes)

        conn.close()
        server_socket.close()

    @staticmethod
    def _recv_exactly(conn, n: int) -> bytes:
        """Read exactly n bytes from a connected socket."""
        buf = bytearray()
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return bytes(buf)   # connection closed — caller checks empty
            buf.extend(chunk)
        return bytes(buf)

    def is_shutdown(self) -> bool:
        return self.shutdown_complete


# ─────────────────────────────────────────────────────────────────────────────
# _SatelliteTuple  — named container replacing the plain tuple
# ─────────────────────────────────────────────────────────────────────────────

class _SatelliteTuple:
    """Holds all per-satellite objects in one place."""

    def __init__(
        self,
        name:        str,
        propagator:  "BasePropagator",
        sensor_mg:   SensorManager,
        features_mg: FeaturesManager,
        date_mg:     DateManager,
        attitude_mg: AttitudeManager,
    ):
        self.name        = name
        self.propagator  = propagator
        self.sensor_mg   = sensor_mg
        self.features_mg = features_mg
        self.date_mg     = date_mg
        self.attitude_mg = attitude_mg

    # Keep compatibility with the original tuple-unpacking style
    def __iter__(self):
        return iter((
            self.name,
            self.propagator,
            self.sensor_mg,
            self.features_mg,
            self.date_mg,
            self.attitude_mg,
        ))


# ─────────────────────────────────────────────────────────────────────────────
# SpiceEnvironment  (replaces STKEnvironment entirely)
# ─────────────────────────────────────────────────────────────────────────────

class SpiceEnvironment:
    """
    Pure-Python RL environment powered by SPICE + analytical propagation.

    Responsibilities that were previously handled by the STK engine:
    ──────────────────────────────────────────────────────────────────
    Orbit propagation      make_propagator()  (TwoBody / J2 / J4 / SGP4)
    Frame transforms       coordinates.py     (ECI ↔ ECEF ↔ geodetic)
    Sensor access          AccessChecker      (geometric cone check)
    Coverage grid          GridManager        (vectorised lat/lon sweep)
    Reward calculation     Rewarder           (unchanged logic)
    """

    def __init__(
        self,
        agents_config:   dict,
        evpt_file_path:  str,
        out_folder_path: str,
    ):
        self.agents_config   = agents_config
        self.evpt_file_path  = evpt_file_path
        self.out_folder_path = out_folder_path

        # ── Optional: load SPICE kernels (graceful fallback if absent) ───
        try:
            download_kernels()
            load_spice_kernels()
        except Exception as exc:
            print(f"[SPICE] Kernel load skipped — using analytic fallback. ({exc})")

        # ── Scenario time bounds ─────────────────────────────────────────
        self.start_time = agents_config["start_time"]
        self.stop_time  = agents_config["stop_time"]

        # ── Shared managers ──────────────────────────────────────────────
        self.date_mg   = DateManager(self.start_time, self.stop_time)
        self.target_mg = TargetManager(
            self.start_time, agents_config["visible_targets"]
        )
        self.rewarder  = Rewarder(agents_config, self.target_mg)
        self.plotter   = Plotter(out_folder_path)

        # ── Access checker (stateless, shared) ───────────────────────────
        self._access_checker = AccessChecker()

        # ── Load event zones ─────────────────────────────────────────────
        self.all_event_zones = pd.read_csv(evpt_file_path)
        self._draw_n_zones(
            agents_config["visible_targets"],
            self.all_event_zones,
            self.start_time,
            first_id=0,
        )

        # ── Build per-satellite objects ───────────────────────────────────
        self.satellites: list[_SatelliteTuple] = []
        for i, agent_raw in enumerate(agents_config["agents"]):
            agent = DataFromJSON(agent_raw, "agent").get_dict()
            sat   = self._build_satellite(agent, i)
            self.satellites.append(sat)

        # ── Optional coverage grid ────────────────────────────────────────
        self.grid_mg: Optional[GridManager] = None
        if agents_config.get("use_grid", False):
            self.grid_mg = GridManager(
                resolution_deg=agents_config.get("grid_resolution", 5.0)
            )
            self.grid_mg.build_global_grid()

    # ── Scenario / satellite construction ────────────────────────────────

    def _build_satellite(
        self, agent: dict, idx: int
    ) -> _SatelliteTuple:
        """
        Construct all managers for one satellite.
        Replaces STKEnvironment.build_satellite().
        """
        name    = agent.get("name", f"MySatellite{idx}")
        epoch_et = stk_date_to_et(self.start_time)

        propagator  = make_propagator(
            agent, self.agents_config["propagator"], epoch_et
        )
        sensor_mg   = SensorManager(agent)
        date_mg     = DateManager(self.start_time, self.stop_time)
        attitude_mg = AttitudeManager(agent)
        features_mg = FeaturesManager(agent, self.rewarder)

        # ── Initial propagation to get r/v at epoch ───────────────────
        r_eci, v_eci, elems = propagator.propagate(epoch_et)

        # ── Update aux state (LLA) ────────────────────────────────────
        features_mg.update_entire_aux_state(r_eci, epoch_et)

        # ── Populate initial state vector ─────────────────────────────
        checked: list[str] = []
        for var in features_mg.states_features:
            if var in checked:
                continue
            if var in ("a", "e", "i", "raan", "aop", "ta",
                       "x", "y", "z", "vx", "vy", "vz"):
                features_mg.update_orbital_elements(elems)
                checked += ["a", "e", "i", "raan", "aop", "ta",
                            "x", "y", "z", "vx", "vy", "vz"]
            elif var in ("pitch", "roll"):
                features_mg.update_attitude_state(
                    attitude_mg.current_pitch, attitude_mg.current_roll
                )
                checked += ["pitch", "roll"]
            elif var in ("az", "el"):
                features_mg.update_sensor_state(
                    sensor_mg.current_azimuth, sensor_mg.current_elevation
                )
                checked += ["az", "el"]
            elif var in ("detic_lat", "detic_lon", "detic_alt"):
                features_mg.update_detic_state()
                checked += ["detic_lat", "detic_lon", "detic_alt"]
            elif any(var.startswith(p) for p in ("lat_", "lon_", "priority_")):
                features_mg.update_target_memory(
                    self.target_mg.get_FoR_window_df(date_mg, features_mg,
                                                     margin_pct=0),
                    self.target_mg.df,
                )
                n = int(var.split("_")[1])
                checked += [f"lat_{n}", f"lon_{n}", f"priority_{n}"]
            else:
                raise ValueError(
                    f"State feature '{var}' not recognised. Use orbital "
                    "features, 'az', 'el', 'pitch', 'roll', 'detic_lat', "
                    "'detic_lon', 'detic_alt', or 'lat_N'/'lon_N'/'priority_N'."
                )

        return _SatelliteTuple(name, propagator, sensor_mg,
                               features_mg, date_mg, attitude_mg)

    # ── Zone management (no STK Children.New calls) ───────────────────────

    def _draw_n_zones(
        self,
        n:          int,
        given_zones: pd.DataFrame,
        start_date: str,
        first_id:   int = 0,
    ):
        if n > given_zones.shape[0]:
            raise ValueError(
                f"Requested {n} zones but only "
                f"{given_zones.shape[0]} available in file."
            )
        if n == 0:
            return

        zones = given_zones.sample(n, ignore_index=True)

        for i in range(zones.shape[0]):
            end_date = self.date_mg.get_date_after(
                float(zones.loc[i, "duration [s]"]), start_date
            )
            if "lat [deg]" in zones.columns and "lon [deg]" in zones.columns:
                lat      = float(zones.loc[i, "lat [deg]"])
                lon      = float(zones.loc[i, "lon [deg]"])
                priority = float(zones.loc[i, "priority"])
                alt      = float(zones.loc[i, "alt [m]"]) / 1000.0 \
                           if "alt [m]" in zones.columns else 0.0
                self.target_mg.append_zone(
                    f"target{i + first_id}", {}, "Point",
                    lat, lon, priority, start_date, end_date,
                )
            elif "lat 1 [deg]" in zones.columns:
                n_pts = sum(1 for c in zones.columns if c.startswith("lat "))
                lats  = [float(zones.loc[i, f"lat {j} [deg]"]) for j in range(1, n_pts + 1)]
                lons  = [float(zones.loc[i, f"lon {j} [deg]"]) for j in range(1, n_pts + 1)]
                # Filter None-like values
                pairs = [(la, lo) for la, lo in zip(lats, lons)
                         if la is not None and lo is not None]
                if len(pairs) < 3:
                    raise ValueError("Area target must have at least 3 points.")
                lat0 = pairs[0][0]
                lon0 = pairs[0][1]
                prio = float(zones.loc[i, "priority"])
                self.target_mg.append_zone(
                    f"target{i + first_id}", {}, "Area",
                    lat0, lon0, prio, start_date, end_date,
                )
            else:
                raise ValueError(
                    "Unrecognised event-zone column format. "
                    "Use 'lat [deg]'/'lon [deg]' or 'lat 1 [deg]'/'lon 1 [deg]' …"
                )

    def _update_target_zones(self, sat: _SatelliteTuple):
        n = self.target_mg.n_of_zones_to_add(sat.date_mg.current_date)
        if n > 0:
            self._draw_n_zones(
                n, self.all_event_zones,
                sat.date_mg.current_date,
                first_id=self.target_mg.max_id,
            )
        self._unload_expired_zones()

    def _unload_expired_zones(self):
        lowest_et = min(
            s.date_mg.num_of_date(
                s.date_mg.simplify_date(s.date_mg.current_date)
            )
            for s in self.satellites
        )
        self.target_mg.unload_zones_before(lowest_et)

    # ── Step / update logic ───────────────────────────────────────────────

    def step(self, agent_id, action: dict, delta_time: float):
        """
        Forward step.  Mirrors STKEnvironment.step() exactly.
        """
        debug = self.agents_config.get("debug", False)

        if debug:
            t0 = perf_counter()

        if 0.0 < delta_time < 0.5:
            raise ValueError("delta_time must be 0 or ≥ 0.5 s.")

        # ── Update agent state ────────────────────────────────────────
        if debug:
            before = perf_counter()
        done = self._update_agent(agent_id, action, delta_time)
        if debug:
            print(f"Time to update agent: {perf_counter() - before:.3f} s")

        if done:
            print(f"Episode for agent {agent_id} is done.")
            return None, None, True

        # ── Get state ────────────────────────────────────────────────
        if debug:
            before = perf_counter()
        state = self.get_state(agent_id, as_dict=True)
        if debug:
            print(f"Time to get state: {perf_counter() - before:.3f} s")

        if delta_time == 0.0:
            return state, None, None

        # ── Get reward ────────────────────────────────────────────────
        if debug:
            before = perf_counter()
        reward = self._get_reward(agent_id, delta_time)
        if debug:
            print(f"Time to get reward: {perf_counter() - before:.3f} s")

        self.plotter.store_reward(reward)

        # ── Update target zones ───────────────────────────────────────
        sat = self._get_satellite(agent_id)
        self._update_target_zones(sat)

        if debug:
            print(f"Total step time: {perf_counter() - t0:.3f} s")

        return state, reward, False

    def _update_agent(
        self, agent_id, action: dict, delta_time: float
    ) -> bool:
        """
        Apply action, advance clock, propagate orbit, refresh state.
        Replaces STKEnvironment.update_agent().
        """
        sat = self._get_satellite(agent_id)
        (name, propagator, sensor_mg,
         features_mg, date_mg, attitude_mg) = sat

        if delta_time != 0.0:
            # ── Sanitise action values before any downstream use ──────────
            # The PPO network can output NaN or Inf values, especially during
            # early training when weights are still unstable.  A single NaN
            # that reaches features_mg.update_action() poisons
            # slew_constraint() (via abs(NaN) = NaN), which then returns a NaN
            # reward.  That NaN flows through GAE → PPO loss → backward() →
            # optim.step(), turning ALL subsequent network weights to NaN and
            # making the training run unrecoverable.
            # Clamping here — before any manager sees the values — is the
            # single correct place to break this cascade.
            action = {k: (float(v) if np.isfinite(float(v)) else 0.0)
                      for k, v in action.items()}

            # BUG FIX 1: attitude was updated once per key in the loop,
            # so a {d_pitch, d_roll} action applied update_roll_pitch twice,
            # doubling the rotation.  Collect attitude deltas separately and
            # apply them in a single call after the loop.
            _attitude_updated = False
            for key, value in action.items():
                features_mg.update_action(key, value)

                if key == "d_az":
                    sensor_mg.update_azimuth(value)
                elif key == "d_el":
                    sensor_mg.update_elevation(value)
                elif key in ("d_pitch", "d_roll"):
                    pass  # handled once after loop (see below)
                else:
                    raise ValueError(
                        f"Invalid action key '{key}'. "
                        "Use 'd_az', 'd_el', 'd_pitch', or 'd_roll'."
                    )

            # Apply attitude update exactly once.
            # Guard against NaN / Inf coming from the PPO network (e.g. during
            # early training when weights have not yet stabilised, or after a
            # reward spike causes a large gradient step).  A non-finite delta
            # produces a NaN quaternion whose norm is zero, which triggers
            # scipy's "Found zero norm quaternions" ValueError.
            if any(k in action for k in ("d_pitch", "d_roll")):
                d_pitch = float(action.get("d_pitch", 0.0))
                d_roll  = float(action.get("d_roll",  0.0))
                if not np.isfinite(d_pitch):
                    d_pitch = 0.0
                if not np.isfinite(d_roll):
                    d_roll = 0.0
                attitude_mg.update_roll_pitch(d_pitch, d_roll)

            date_mg.update_date_after(delta_time)

            if self.agents_config.get("debug", False):
                print(f"Current date: {date_mg.current_date}")

        # ── Check done ────────────────────────────────────────────────
        if date_mg.time_ended():
            return True

        # ── Propagate orbit to new time ───────────────────────────────
        current_et        = stk_date_to_et(date_mg.current_date)
        r_eci, v_eci, elems = propagator.propagate(current_et)

        # ── Refresh aux state (LLA) ───────────────────────────────────
        features_mg.update_entire_aux_state(r_eci, current_et)

        # ── Refresh RL state vector ───────────────────────────────────
        checked: list[str] = []
        for var in features_mg.state.keys():
            if var in checked:
                continue
            if var in ("a", "e", "i", "raan", "aop", "ta",
                       "x", "y", "z", "vx", "vy", "vz"):
                features_mg.update_orbital_elements(elems)
                checked += ["a", "e", "i", "raan", "aop", "ta",
                            "x", "y", "z", "vx", "vy", "vz"]
            elif var in ("pitch", "roll"):
                features_mg.update_attitude_state(
                    attitude_mg.current_pitch, attitude_mg.current_roll
                )
                checked += ["pitch", "roll"]
            elif var in ("az", "el"):
                features_mg.update_sensor_state(
                    sensor_mg.current_azimuth, sensor_mg.current_elevation
                )
                checked += ["az", "el"]
            elif var in ("detic_lat", "detic_lon", "detic_alt"):
                features_mg.update_detic_state()
                checked += ["detic_lat", "detic_lon", "detic_alt"]
            elif any(var.startswith(p) for p in ("lat_", "lon_", "priority_")):
                features_mg.update_target_memory(
                    self.target_mg.get_FoR_window_df(date_mg, features_mg,
                                                     margin_pct=0),
                    self.target_mg.df,
                )
                n = int(var.split("_")[1])
                checked += [f"lat_{n}", f"lon_{n}", f"priority_{n}"]
            else:
                raise ValueError(
                    f"State feature '{var}' not recognised."
                )

        return False

    # ── Reward computation ────────────────────────────────────────────────

    def _get_reward(self, agent_id, delta_time: float) -> float:
        """
        Compute reward for the current step.
        Replaces STKEnvironment.get_reward().

        STK data-provider access is replaced by AccessChecker geometric
        visibility testing.
        """
        debug = self.agents_config.get("debug", False)
        sat   = self._get_satellite(agent_id)
        (name, propagator, sensor_mg,
         features_mg, date_mg, attitude_mg) = sat

        # ── Time window ───────────────────────────────────────────────
        last_et    = stk_date_to_et(date_mg.last_date)
        current_et = stk_date_to_et(date_mg.current_date)
        min_dur    = self.agents_config.get("min_duration", 0.0)
        adj_et     = current_et + min_dur

        # ── FoR window + max footprint radius ─────────────────────────
        if debug:
            before = perf_counter()
        FoR_df, D_FoR = self.target_mg.get_FoR_window_df(
            date_mg, features_mg, return_D_FoR=True
        )
        if debug:
            print(f"    FoR window ({FoR_df.shape[0]} targets): "
                  f"{perf_counter() - before:.3f} s")

        # ── Compute access events over the step interval ──────────────
        #
        # We sample the interval at N_SAMPLES evenly-spaced times,
        # run the geometric check at each sample, and merge consecutive
        # visible samples into access intervals.  This mirrors what
        # STK's access engine would report but is purely geometric.
        #
        N_SAMPLES    = max(10, int(delta_time / 10))
        sample_times = [last_et + (adj_et - last_et) * k / (N_SAMPLES - 1)
                        for k in range(N_SAMPLES)]

        if debug:
            before = perf_counter()

        access_events: list[dict] = []

        for _, zone in FoR_df.iterrows():
            from scripts.coordinates import geodetic_to_eci

            # BUG FIX 2: target ECI position was computed once at last_et and
            # reused for all samples.  Earth rotates ~23° over a 5553 s step,
            # so the target drifts significantly in ECI space.  Recompute at
            # every sample time so the geometry is always consistent.
            tgt_lat = zone["lat [deg]"]
            tgt_lon = zone["lon [deg]"]

            visible_times:  list[float] = []
            best_elevation: float       = -999.0

            for t_et in sample_times:
                r_eci, v_eci, _ = propagator.propagate(t_et)
                r_tgt_eci = geodetic_to_eci(tgt_lat, tgt_lon, 0.0, t_et)
                has_access, elev = self._access_checker.check_access(
                    r_eci, v_eci, r_tgt_eci,
                    attitude_mg, sensor_mg,
                )
                if has_access:
                    visible_times.append(t_et)
                    best_elevation = max(best_elevation, elev)

            if not visible_times:
                continue

            # Merge into contiguous intervals (gap > 2× sample step → new event)
            dt_sample = (adj_et - last_et) / max(N_SAMPLES - 1, 1)
            gap_limit = 2.5 * dt_sample
            intervals: list[tuple[float, float]] = []
            seg_start = visible_times[0]
            seg_end   = visible_times[0]
            for t in visible_times[1:]:
                if t - seg_end <= gap_limit:
                    seg_end = t
                else:
                    intervals.append((seg_start, seg_end))
                    seg_start = seg_end = t
            intervals.append((seg_start, seg_end))

            from scripts.coordinates import et_to_stk_date
            for iv_start, iv_end in intervals:
                access_events.append({
                    "name":       zone["name"],
                    "start_time": et_to_stk_date(iv_start),
                    "stop_time":  et_to_stk_date(iv_end),
                    "elevation":  best_elevation,
                })

        if debug:
            print(f"    Access computation ({len(access_events)} events): "
                  f"{perf_counter() - before:.3f} s")

        # ── Coverage grid ─────────────────────────────────────────────
        if debug:
            before = perf_counter()

        grid_points_seen: Optional[list[tuple[float, float]]] = None
        if self.agents_config.get("use_grid", False) and self.grid_mg is not None:
            mid_et   = (last_et + current_et) / 2.0
            r_mid, v_mid, _ = propagator.propagate(mid_et)
            grid_points_seen = self.grid_mg.get_seen_points(
                r_mid, v_mid, mid_et, attitude_mg, sensor_mg
            )

        if debug:
            print(f"    Grid points: {perf_counter() - before:.3f} s")

        # ── Call rewarder ─────────────────────────────────────────────
        reward = self.rewarder.calculate_reward(
            access_events,
            delta_time,
            grid_points_seen,
            FoR_df,
            D_FoR,
            date_mg,
            sensor_mg,
            features_mg,
            attitude_mg.angle_domains,
            attitude_max_slew=attitude_mg.max_slew,
        )
        # Cache for telemetry — step() reads this immediately after _get_reward
        self._last_access_events = access_events
        return reward

    # ── State getter ──────────────────────────────────────────────────────

    def get_state(self, agent_id, as_dict: bool = False):
        sat   = self._get_satellite(agent_id)
        state = sat.features_mg.get_state()

        # ── Boresight ground intercept (telemetry only, not an RL feature) ──
        # Compute where the sensor boresight intersects the Earth surface.
        # This lets the visualiser draw the footprint at the correct location
        # instead of always at the sub-satellite point.
        try:
            current_et = stk_date_to_et(sat.date_mg.current_date)
            r_eci, v_eci, _ = sat.propagator.propagate(current_et)
            M = sat.attitude_mg.body_to_eci_matrix(r_eci, v_eci)
            boresight_eci = M @ sat.sensor_mg.boresight_body()

            # Ray–sphere intersection: find where boresight hits Earth (RT km)
            a_coef = 1.0
            b_coef = 2.0 * float(np.dot(r_eci, boresight_eci))
            c_coef = float(np.dot(r_eci, r_eci)) - RT ** 2
            disc   = b_coef ** 2 - 4.0 * a_coef * c_coef
            if disc >= 0:
                t_hit  = (-b_coef - np.sqrt(disc)) / (2.0 * a_coef)
                if t_hit > 0:
                    hit_eci  = r_eci + t_hit * boresight_eci
                    hit_ecef = eci_to_ecef(hit_eci, current_et)
                    bs_lat, bs_lon, _ = ecef_to_geodetic(hit_ecef)
                    state = dict(state)   # copy so we don't mutate features_mg.state
                    state["boresight_lat"] = float(bs_lat)
                    state["boresight_lon"] = float(bs_lon)
        except Exception:
            pass  # graceful — boresight fields simply absent if geometry fails

        # Simulation time — ISO-style string so consumers don't need to know
        # the STK epoch format.  Added here rather than in features_mg so it
        # never pollutes the RL observation vector (it is only used by telemetry).
        try:
            state = dict(state)
            state["sim_time"] = str(sat.date_mg.current_date)
        except Exception:
            pass

        return state if as_dict else list(state.values())

    def reset(self) -> dict:
        """
        Re-initialise the environment to the scenario start time.

        Called by the Gym server when it receives a {"command": "reset"} message.
        Rebuilds every per-satellite manager (date clock, propagator, attitude,
        target zones) so the next episode starts clean without restarting the
        server process.

        Returns the initial state dict for agent 0 (the primary agent).
        """
        # Re-seed the scenario clock
        self.date_mg = DateManager(self.start_time, self.stop_time)

        # Clear and re-draw target zones
        self.target_mg = TargetManager(
            self.start_time, self.agents_config["visible_targets"]
        )
        self.rewarder = Rewarder(self.agents_config, self.target_mg)

        self._draw_n_zones(
            self.agents_config["visible_targets"],
            self.all_event_zones,
            self.start_time,
            first_id=0,
        )

        # Rebuild every satellite from scratch (resets clocks, attitude, LLA log)
        self.satellites = []
        for i, agent_raw in enumerate(self.agents_config["agents"]):
            agent = DataFromJSON(agent_raw, "agent").get_dict()
            sat   = self._build_satellite(agent, i)
            self.satellites.append(sat)

        # Return the initial state of agent 0
        return self.get_state(0, as_dict=True)

    # ── Satellite lookup ──────────────────────────────────────────────────

    def _get_satellite(self, agent_id) -> _SatelliteTuple:
        if isinstance(agent_id, str):
            for s in self.satellites:
                if s.name == agent_id:
                    return s
        elif isinstance(agent_id, int):
            for s in self.satellites:
                if s.name == f"MySatellite{agent_id}":
                    return s
        else:
            raise ValueError(
                "agent_id must be a string (satellite name) or int (index)."
            )
        raise ValueError(f"Satellite with ID '{agent_id}' not found.")
