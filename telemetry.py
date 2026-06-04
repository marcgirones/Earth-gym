"""
telemetry.py
============
Thread-safe telemetry writer for Earth-Gym training runs.

Every training step appends one JSON line to  output/telemetry.jsonl.
Both the matplotlib visualizer and the CesiumJS dashboard consume this file.

Record schema
-------------
{
  "step":          int,        batch index
  "frame":         int,        absolute frame count
  "sat_lat":       float,      satellite geodetic latitude  (deg)  ← from propagator
  "sat_lon":       float,      satellite geodetic longitude (deg)  ← from propagator
  "sat_alt":       float,      satellite altitude above WGS-84 ellipsoid (km)
  "pitch":         float,      attitude pitch (deg)
  "roll":          float,      attitude roll  (deg)
  "boresight_lat": float,      sensor boresight ground intercept latitude  (deg)
  "boresight_lon": float,      sensor boresight ground intercept longitude (deg)
  "reward":        float,      step reward
  "eval_reward":   float|null, evaluation reward (set only at eval steps)
  "targets":       [[lat, lon, priority], ...],  FoR target memory slots
  "ts":            float,      wall-clock UNIX timestamp

NOTE on field naming
--------------------
"sat_lat" / "sat_lon" / "sat_alt" always refer to the satellite's own position
as computed by the orbit propagator (ECI → ECEF → geodetic).  They are written
from raw_state["detic_lat/lon/alt"] — the server state dict — never from the
normalised RL observation vector.

"targets" contains the field-of-regard (FoR) target lat/lon pairs that the RL
agent sees in its observation.  They are NOT satellite positions and must never
be plotted on the ground track.
}
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path


class TelemetryLogger:
    """
    Appends one JSON record per call to  <out_dir>/telemetry.jsonl.

    Thread-safe: a single lock guards the file handle so that the
    training thread and any background eval thread can both write.
    """

    def __init__(self, out_dir: str):
        self._path   = Path(out_dir) / "telemetry.jsonl"
        self._lock   = threading.Lock()
        # Open in WRITE mode ("w"), not append ("a").
        # Stale records from a previous training run (possibly written with
        # buggy code) would otherwise mix with new correct records and corrupt
        # the visualiser's ground track and coverage heatmap.
        # Each new training run starts with a clean telemetry file.
        self._handle = open(self._path, "w", buffering=1)   # line-buffered
        self._frame  = 0

        # Per-step evaluation telemetry — written to a separate file so it can
        # be consumed by the dashboard without polluting the training stream.
        self._eval_path   = Path(out_dir) / "eval_telemetry.jsonl"
        self._eval_handle = open(self._eval_path, "w", buffering=1)
        self._eval_episode = 0   # incremented at the start of each eval run
        self._eval_frame   = 0   # global step counter across all eval runs

    # ── Writers ───────────────────────────────────────────────────────────

    def log_step(
        self,
        step:        int,
        obs:         dict,
        reward:      float,
        raw_state:   dict | None  = None,
        eval_reward: float | None = None,
    ) -> None:
        """
        Append one training-step record.

        Parameters
        ----------
        obs        : feature dict from _obs_to_dict (normalisation-free, for
                     RL features like pitch, roll, target lats)
        raw_state  : full server state dict from _gym_env.last_raw_state.
                     Used for sat position (detic_lat/lon/alt) and boresight so
                     we always log the TRUE values from the propagator, never
                     a normalisation-inverted approximation.
        reward     : step reward
        eval_reward: optional eval reward for checkpoint steps
        """
        # Satellite position — read from raw_state (propagator truth) when
        # available, fall back to obs (normalisation inversion) otherwise.
        rs = raw_state or obs
        sat_lat = float(rs.get("detic_lat", obs.get("detic_lat", 0.0)))
        sat_lon = float(rs.get("detic_lon", obs.get("detic_lon", 0.0)))
        sat_alt = float(rs.get("detic_alt", obs.get("detic_alt", 0.0)))

        # Boresight ground intercept — only in raw_state (server-computed).
        # Falls back to sub-satellite point when not present.
        boresight_lat = float(rs.get("boresight_lat", sat_lat))
        boresight_lon = float(rs.get("boresight_lon", sat_lon))

        # Target memory slots (from RL obs vector)
        targets = []
        for n in range(1, 20):
            if f"lat_{n}" not in obs:
                break
            targets.append([
                round(obs[f"lat_{n}"],             4),
                round(obs[f"lon_{n}"],             4),
                round(obs.get(f"priority_{n}", 0.0), 4),
            ])

        record = {
            "step":          step,
            "frame":         self._frame,
            # ── Satellite position (propagator truth, NOT normalised obs) ──
            # Named sat_lat/sat_lon/sat_alt to make it impossible to confuse
            # with the FoR target slots lat_1..N / lon_1..N in the obs vector.
            "sat_lat":       round(sat_lat,        5),
            "sat_lon":       round(sat_lon,        5),
            "sat_alt":       round(sat_alt,        3),
            # ── Attitude ──────────────────────────────────────────────────
            "pitch":         round(obs.get("pitch", 0.0), 3),
            "roll":          round(obs.get("roll",  0.0), 3),
            # ── Sensor boresight ground intercept ─────────────────────────
            "boresight_lat": round(boresight_lat,  5),
            "boresight_lon": round(boresight_lon,  5),
            # ── Rewards and FoR targets ───────────────────────────────────
            "reward":        round(float(reward),  6),
            "eval_reward":   round(float(eval_reward), 6) if eval_reward is not None else None,
            "targets":       targets,   # FoR lat/lon — NOT satellite positions
            "ts":            round(time.time(),    3),
        }
        self._write(record)
        self._frame += 1

    def begin_eval_episode(self) -> int:
        """
        Call once at the start of each evaluation rollout.
        Returns the episode index so the caller can pass it to log_eval_step.
        """
        self._eval_episode += 1
        return self._eval_episode

    def log_eval_step(
        self,
        eval_episode:   int,
        eval_step:      int,
        train_batch:    int,
        raw_state:      dict,
        obs:            dict,
        reward:         float,
        access_events:  list | None = None,
    ) -> None:
        """
        Append one per-step record to eval_telemetry.jsonl.

        Record schema
        -------------
        {
          "eval_episode":  int,    which evaluation run this belongs to
          "eval_step":     int,    step index within the episode (0-based)
          "eval_frame":    int,    global step counter across all eval runs
          "train_batch":   int,    training batch index that triggered this eval
          "sim_time":      str,    simulation clock string from the server
          "sat_lat":       float,
          "sat_lon":       float,
          "sat_alt":       float,
          "pitch":         float,
          "roll":          float,
          "boresight_lat": float,
          "boresight_lon": float,
          "reward":        float,
          "targets":       [[lat, lon, priority], ...],
          "access_events": [{"name": str, "start_time": str,
                             "stop_time": str, "elevation": float}, ...],
          "ts":            float,  wall-clock UNIX timestamp
        }
        """
        rs = raw_state or {}

        sat_lat = float(rs.get("detic_lat",      obs.get("detic_lat",      0.0)))
        sat_lon = float(rs.get("detic_lon",      obs.get("detic_lon",      0.0)))
        sat_alt = float(rs.get("detic_alt",      obs.get("detic_alt",      0.0)))
        boresight_lat = float(rs.get("boresight_lat", sat_lat))
        boresight_lon = float(rs.get("boresight_lon", sat_lon))

        targets = []
        for n in range(1, 20):
            if f"lat_{n}" not in obs:
                break
            targets.append([
                round(obs[f"lat_{n}"],              4),
                round(obs[f"lon_{n}"],              4),
                round(obs.get(f"priority_{n}", 0.0), 4),
            ])

        record = {
            "eval_episode":  eval_episode,
            "eval_step":     eval_step,
            "eval_frame":    self._eval_frame,
            "train_batch":   train_batch,
            "sim_time":      rs.get("sim_time", ""),
            "sat_lat":       round(sat_lat,        5),
            "sat_lon":       round(sat_lon,        5),
            "sat_alt":       round(sat_alt,        3),
            "pitch":         round(obs.get("pitch", 0.0), 3),
            "roll":          round(obs.get("roll",  0.0), 3),
            "boresight_lat": round(boresight_lat,  5),
            "boresight_lon": round(boresight_lon,  5),
            "reward":        round(float(reward),  6),
            "targets":       targets,
            "access_events": access_events or [],
            "ts":            round(time.time(),    3),
        }
        self._write_eval(record)
        self._eval_frame += 1

    def _write(self, record: dict) -> None:
        with self._lock:
            self._handle.write(json.dumps(record) + "\n")

    def _write_eval(self, record: dict) -> None:
        with self._lock:
            self._eval_handle.write(json.dumps(record) + "\n")

    def close(self) -> None:
        with self._lock:
            self._handle.flush()
            self._handle.close()
            self._eval_handle.flush()
            self._eval_handle.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ── Readers (used by visualizer + dashboard) ──────────────────────────

    @staticmethod
    def load(path: str | Path, last_n: int | None = None) -> list[dict]:
        """
        Read telemetry records from a JSONL file.

        Parameters
        ----------
        path   : path to telemetry.jsonl
        last_n : if given, return only the last N records

        Returns
        -------
        List of record dicts in chronological order.
        """
        path = Path(path)
        if not path.exists():
            return []

        records = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass   # skip malformed lines

        if last_n is not None:
            records = records[-last_n:]
        return records
