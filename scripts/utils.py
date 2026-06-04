"""
scripts/utils.py
================
All utility / manager classes for the Earth-Gym RL environment.

STK has been fully removed.  Geometry is done with numpy + the local
coordinates.py / propagator.py modules.  spiceypy is used for frame
transforms when kernels have been loaded (transparent fallback otherwise).

Classes
-------
DataFromJSON        — JSON config flattener (unchanged)
DateManager         — simulation clock, replaced STK time strings with datetime
AttitudeManager     — satellite attitude (nadir-pointing, pitch/roll slew)
SensorManager       — sensor pointing (az/el cone), no STK sensor object
FeaturesManager     — RL state/action bookkeeping + LLA from propagator
TargetManager       — ground-target / event-zone catalogue
AccessChecker       — geometric sensor-to-target visibility (replaces STK access)
GridManager         — simple lat/lon coverage grid
Rewarder            — reward calculation (same logic, no STK data providers)
Plotter             — matplotlib reward curves (unchanged)
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R

from scripts.coordinates import (
    eci_to_ecef,
    ecef_to_geodetic,
    stk_date_to_et,
    et_to_stk_date,
)

# ── Earth radius (km) ─────────────────────────────────────────────────────────
RT = 6371.0


# ─────────────────────────────────────────────────────────────────────────────
# DataFromJSON  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

class DataFromJSON:
    """Flatten a nested JSON dict into a plain object (leaf nodes only)."""

    def __init__(self, json_dict: dict, data_type: str):
        self.loop(json_dict)
        self.data_type = data_type

    def loop(self, json_dict: dict):
        if not isinstance(json_dict, dict):
            return
        for key, value in json_dict.items():
            if isinstance(value, dict):
                self.loop(value)
            else:
                if hasattr(self, key):
                    raise ValueError(
                        f"Variable '{key}' already exists.  "
                        "Rename the JSON key in your configuration file."
                    )
                setattr(self, key, value)

    def get_dict(self) -> dict:
        return self.__dict__


# ─────────────────────────────────────────────────────────────────────────────
# DateManager
# ─────────────────────────────────────────────────────────────────────────────

class DateManager:
    """
    Simulation clock.

    Internally stores times as *ephemeris time* (float seconds past J2000) and
    converts to/from the STK-style string format for compatibility with the
    existing JSON configuration files.
    """

    def __init__(self, start_date: str, stop_date: str):
        self.class_name   = "Date Manager"
        self.start_date   = start_date          # STK string (kept for compat.)
        self.stop_date    = stop_date
        self.current_date = start_date
        self.last_date    = start_date

        self._start_et   = stk_date_to_et(start_date)
        self._stop_et    = stk_date_to_et(stop_date)
        self._current_et = self._start_et
        self._last_et    = self._start_et

        # Legacy attributes used by TargetManager.num_of_date
        self.current_simplified_date = self._to_simplified(start_date)
        self.last_simplified_date    = self.current_simplified_date

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _to_simplified(stk_date: str) -> str:
        """STK date → 'D N YYYY H M S' (all numbers) for num_of_date()."""
        months = {"Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4,
                  "Jun": 5, "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9,
                  "Nov": 10, "Dec": 11}
        parts        = stk_date.strip().split()
        day          = parts[0]
        month_n      = months[parts[1]]
        year         = parts[2]
        h, m, s      = parts[3].split(":")
        return f"{day} {month_n} {year} {h} {m} {s}"

    # Public month helpers kept for backward compatibility with TargetManager
    @staticmethod
    def month_to_number(month: str) -> int:
        months = {"Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4,
                  "Jun": 5, "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9,
                  "Nov": 10, "Dec": 11}
        if month not in months:
            raise ValueError(f"Unknown month abbreviation: {month}")
        return months[month]

    @staticmethod
    def number_to_month(number: int) -> str:
        months = {0: "Jan", 1: "Feb", 2: "Mar", 3: "Apr", 4: "May",
                  5: "Jun", 6: "Jul", 7: "Aug", 8: "Sep", 9: "Oct",
                  10: "Nov", 11: "Dec"}
        if number not in months:
            raise ValueError(f"Month number must be 0-11, got {number}")
        return months[number]

    def simplify_date(self, date: str) -> str:
        return self._to_simplified(date)

    def fancy_date(self, simplified: str) -> str:
        """'D N YYYY H M S' → STK string."""
        day, month, year, h, m, s = simplified.split()
        return f"{day} {self.number_to_month(int(month))} {year} {h}:{m}:{s}"

    # ── Time queries ──────────────────────────────────────────────────────

    @property
    def current_et(self) -> float:
        return self._current_et

    def get_date_after(
        self, delta_time: float | dict, current_date: str,
        return_simplified: bool = False
    ) -> str:
        et = stk_date_to_et(current_date)
        if isinstance(delta_time, dict):
            dt = (delta_time.get("seconds", 0)
                  + delta_time.get("minutes", 0) * 60
                  + delta_time.get("hours",   0) * 3600
                  + delta_time.get("days",    0) * 86400
                  + delta_time.get("months",  0) * 2_592_000
                  + delta_time.get("years",   0) * 31_536_000)
        elif isinstance(delta_time, (int, float)):
            dt = float(delta_time)
        else:
            raise ValueError("delta_time must be a number or dict.")

        new_date_str = et_to_stk_date(et + dt)
        if return_simplified:
            return self._to_simplified(new_date_str)
        return new_date_str

    def get_current_date_after(
        self, delta_time: float | dict,
        return_simplified: bool = False
    ) -> str:
        return self.get_date_after(delta_time, self.current_date,
                                   return_simplified)

    def update_date_after(self, delta_time: float | dict):
        """Advance the clock by *delta_time* seconds (or dict)."""
        self.last_date              = self.current_date
        self._last_et               = self._current_et
        self.last_simplified_date   = self.current_simplified_date

        new_str                     = self.get_current_date_after(delta_time)
        self.current_date           = new_str
        self._current_et           += (
            delta_time if isinstance(delta_time, (int, float))
            else sum(delta_time.values())        # rough
        )
        self.current_simplified_date = self._to_simplified(new_str)

    def is_in_time_range(self, first: str, last: str, current: str) -> bool:
        return (self.num_of_date(self.simplify_date(first))
                <= self.num_of_date(self.simplify_date(current))
                <= self.num_of_date(self.simplify_date(last)))

    def is_newer_than(self, first: str, second: str) -> bool:
        return (self.num_of_date(self.simplify_date(first))
                > self.num_of_date(self.simplify_date(second)))

    def time_ended(self) -> bool:
        return self._current_et > self._stop_et

    def num_of_date(self, simplified: str) -> float:
        """Simplified date string → total seconds since a fixed epoch (J2000)."""
        # Convert the simplified string back to a proper ET
        parts    = simplified.split()
        day      = int(parts[0])
        month_n  = int(float(parts[1]))
        year     = int(float(parts[2]))
        h        = int(float(parts[3]))
        m        = int(float(parts[4]))
        s        = float(parts[5])
        fancy    = f"{day} {self.number_to_month(month_n)} {year} {h:02d}:{m:02d}:{s:09.6f}"
        return stk_date_to_et(fancy)

    # ── Compatibility shim (used by TargetManager) ────────────────────────
    @staticmethod
    def number_of_days_in_month(month: str, year: int) -> int:
        if month in ["Jan","Mar","May","Jul","Aug","Oct","Dec"]:
            return 31
        if month in ["Apr","Jun","Sep","Nov"]:
            return 30
        # Feb
        return 29 if (year % 4 == 0 and
                      (year % 100 != 0 or year % 400 == 0)) else 28


# ─────────────────────────────────────────────────────────────────────────────
# AttitudeManager  (unchanged logic, no STK command strings in public API)
# ─────────────────────────────────────────────────────────────────────────────

class AttitudeManager:
    """
    Tracks satellite attitude (pitch / roll) for a nadir-pointing satellite.

    The STK command strings are kept for reference but are never executed;
    the geometric calculation in AccessChecker uses current_pitch / current_roll
    directly.
    """

    def __init__(self, agent: dict):
        self.class_name      = "Attitude Manager"
        self.current_pitch   = float(agent["initial_pitch"])
        self.current_roll    = float(agent["initial_roll"])
        self.max_slew        = float(agent["max_slew_speed"])
        self.max_accel       = float(agent["max_slew_accel"])

        if agent["attitude_align"] == "Nadir(Centric)":
            self.align_reference    = "Nadir(Centric)"
            self.angle_domains      = {"pitch": [-90, 90], "roll": [-180, 180]}
            self.constraint_reference = "Velocity"
            self.constraint_axes      = "1 0 0"
        else:
            raise NotImplementedError(
                "Only 'Nadir(Centric)' attitude alignment is supported."
            )

    def get_item(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        raise ValueError(f"AttitudeManager has no attribute '{name}'.")

    def update_roll_pitch(
        self, delta_pitch: float, delta_roll: float
    ) -> tuple[float, float]:
        """
        Increment pitch and roll using body-fixed composition of rotations.
        Order: roll (x-axis), pitch (y-axis), yaw=0.

        Robustness notes
        ----------------
        * Non-finite deltas (NaN / Inf from a diverging PPO network) are
          silently replaced with 0 so the server never crashes mid-episode.
        * After thousands of quaternion multiplications, floating-point drift
          can make the stored quaternion slightly non-unit.  scipy normalises
          on construction, but we re-normalise the composed result explicitly
          to prevent gradual accumulation from ever producing a zero-norm quat.
        """
        # Clamp non-finite inputs rather than crashing
        if not np.isfinite(delta_pitch):
            delta_pitch = 0.0
        if not np.isfinite(delta_roll):
            delta_roll = 0.0

        current_rot     = R.from_euler("xyz",
                                       [self.current_roll,
                                        self.current_pitch, 0.0],
                                       degrees=True)
        incremental_rot = R.from_euler("xyz",
                                       [delta_roll, delta_pitch, 0.0],
                                       degrees=True)
        new_rot   = current_rot * incremental_rot
        # Re-normalise to prevent quaternion drift over long episodes
        q         = new_rot.as_quat()
        q_norm    = np.linalg.norm(q)
        if q_norm < 1e-10:
            # Degenerate — reset to identity rather than crash
            new_rot = R.identity()
        else:
            new_rot = R.from_quat(q / q_norm)
        new_euler       = new_rot.as_euler("xyz", degrees=True)
        self.current_roll  = float(new_euler[0])
        self.current_pitch = float(new_euler[1])
        return self.current_pitch, self.current_roll

    # ── Attitude matrix (used by AccessChecker) ───────────────────────────

    def body_to_eci_matrix(
        self, r_eci: np.ndarray, v_eci: np.ndarray
    ) -> np.ndarray:
        """
        Build the 3×3 rotation matrix  body → ECI  for a nadir-pointing
        satellite with the current pitch and roll offsets.

        Local frame definition (aligned with STK AlignConstrain PR):
          z_body  = nadir  (−r̂)
          x_body  = along-track  (v̂ projected onto nadir-perpendicular plane)
          y_body  = cross-track  (z_body × x_body)

        Then pitch rotates about y_body, roll rotates about x_body.
        """
        r_hat = r_eci / np.linalg.norm(r_eci)
        nadir = -r_hat                              # points toward Earth

        v_hat = v_eci / np.linalg.norm(v_eci)
        along = v_hat - np.dot(v_hat, nadir) * nadir
        along_mag = np.linalg.norm(along)
        if along_mag < 1e-9:
            along = np.array([1.0, 0.0, 0.0])
        else:
            along /= along_mag

        cross = np.cross(nadir, along)              # y_body (right-hand)

        # Unperturbed body frame: columns = x, y, z
        M0 = np.column_stack([along, cross, nadir])  # body→ECI, pitch=roll=0

        # Apply pitch (rotation about y-body) then roll (rotation about x-body)
        R_pitch = R.from_euler("y", self.current_pitch, degrees=True).as_matrix()
        R_roll  = R.from_euler("x", self.current_roll,  degrees=True).as_matrix()
        return M0 @ R_pitch @ R_roll


# ─────────────────────────────────────────────────────────────────────────────
# SensorManager  (no STK sensor object)
# ─────────────────────────────────────────────────────────────────────────────

class SensorManager:
    """
    Tracks sensor az/el pointing.  The STK sensor COM-object is gone;
    geometry is handled by AccessChecker.
    """

    def __init__(self, agent: dict):
        self.class_name        = "Sensor Manager"
        self.pattern           = agent.get("pattern", "Simple Conic")
        self.cone_angle        = float(agent["cone_angle"])        # half-angle, deg
        self.max_slew          = float(agent["max_sensor_slew"])   # deg/s
        self.current_azimuth   = float(agent["initial_azimuth"])   # deg
        self.current_elevation = float(agent["initial_elevation"]) # deg
        self.resolution        = float(agent.get("resolution", 0.1))

    def get_item(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        raise ValueError(f"SensorManager has no attribute '{name}'.")

    def update_azimuth(self, delta: float) -> float:
        self.current_azimuth = (self.current_azimuth + delta) % 360.0
        return self.current_azimuth

    def update_elevation(self, delta: float) -> float:
        self.current_elevation = float(
            np.clip(self.current_elevation + delta, -90.0, 90.0)
        )
        return self.current_elevation

    # ── Boresight vector in body frame ────────────────────────────────────

    def boresight_body(self) -> np.ndarray:
        """
        Unit vector of the sensor boresight in the satellite body frame.

        Convention (matching STK SetPointingFixedAzEl with Nadir(Centric)):
          el=90 az=any  → pointing toward nadir (+z_body)
          el=0  az=0    → pointing along-track  (+x_body)
          az measured clockwise from along-track in the x-y plane
        """
        el  = np.radians(self.current_elevation)
        az  = np.radians(self.current_azimuth)
        # Standard spherical to Cartesian with el measured from x-y plane
        x   =  np.cos(el) * np.cos(az)
        y   =  np.cos(el) * np.sin(az)
        z   =  np.sin(el)
        return np.array([x, y, z])


# ─────────────────────────────────────────────────────────────────────────────
# AccessChecker  (replaces STK sensor.GetAccessToObject + AER data providers)
# ─────────────────────────────────────────────────────────────────────────────

class AccessChecker:
    """
    Pure-geometry satellite-to-target visibility check.

    A target is "accessed" when:
      1.  The target is above the Earth's limb as seen from the satellite.
      2.  The angular distance between the sensor boresight and the target
          direction is ≤ cone_angle (sensor half-angle).

    The elevation angle returned mimics STK's AER "Elevation" field:
    angle above the local horizontal at the satellite (positive = above horizon).
    """

    def check_access(
        self,
        r_sat_eci:   np.ndarray,
        v_sat_eci:   np.ndarray,
        r_tgt_eci:   np.ndarray,
        attitude_mg: "AttitudeManager",
        sensor_mg:   "SensorManager",
    ) -> tuple[bool, float]:
        """
        Parameters
        ----------
        r_sat_eci  : (3,) satellite ECI position (km)
        v_sat_eci  : (3,) satellite ECI velocity (km/s)
        r_tgt_eci  : (3,) target   ECI position (km)
        attitude_mg: AttitudeManager instance
        sensor_mg  : SensorManager  instance

        Returns
        -------
        has_access   : bool
        elevation_deg: elevation above/below the local horizon (deg).
                       0 = at Earth limb, positive = within Earth disk,
                       negative = below limb (occluded).
                       Matches STK AER Elevation for satellite→ground.
        """
        r_sat_mag = np.linalg.norm(r_sat_eci)
        r_sat_hat = r_sat_eci / r_sat_mag
        nadir_hat = -r_sat_hat          # unit vector pointing toward Earth

        r_diff     = r_tgt_eci - r_sat_eci
        r_diff_mag = np.linalg.norm(r_diff)
        r_diff_hat = r_diff / r_diff_mag

        # 1. Elevation of the satellite as seen from the target (true horizon check).
        #
        #    The previous code used:
        #        rho = arccos(RT / |r_sat|)          (Earth limb half-angle)
        #        nadir_angle = angle(nadir, sat→tgt)
        #        reject if nadir_angle > rho
        #
        #    This is INCORRECT for surface targets.  nadir_angle > rho means the
        #    target direction falls outside the Earth-disk cone as seen from the
        #    satellite — but a surface target can be well above its own local horizon
        #    even when its direction from the satellite lies outside that cone.
        #    The practical result: targets beyond ~12° latitude were rejected even
        #    though the true geometric limit (el_from_target = 0) is ~37° for a
        #    1629 km orbit.
        #
        #    Correct check: is the satellite above the target's local horizon?
        #        r_tgt_up  = r_tgt / |r_tgt|   (local vertical at target)
        #        el = arcsin( dot((r_sat - r_tgt)/|r_sat - r_tgt|, r_tgt_up) )
        #        visible iff el >= 0
        #
        #    This is equivalent to the ray sat→target not intersecting the Earth,
        #    and is the standard AER elevation used in STK.
        r_tgt_up      = r_tgt_eci / np.linalg.norm(r_tgt_eci)   # local up at target
        sat_from_tgt  = -r_diff / r_diff_mag                      # target → satellite
        sin_el        = float(np.clip(np.dot(sat_from_tgt, r_tgt_up), -1.0, 1.0))
        elevation_deg = float(np.degrees(np.arcsin(sin_el)))      # −90…+90 deg

        # 2. Horizon check: target is occluded when satellite is below target's horizon
        if elevation_deg < 0.0:
            return False, elevation_deg

        # 5. Sensor cone check
        body_to_eci   = attitude_mg.body_to_eci_matrix(r_sat_eci, v_sat_eci)
        boresight_eci = body_to_eci @ sensor_mg.boresight_body()
        cos_angle     = np.clip(np.dot(boresight_eci, r_diff_hat), -1.0, 1.0)
        off_angle     = float(np.degrees(np.arccos(cos_angle)))

        has_access = off_angle <= sensor_mg.cone_angle
        return has_access, elevation_deg


# ─────────────────────────────────────────────────────────────────────────────
# FeaturesManager  (LLA and orbital elements from propagator, not STK)
# ─────────────────────────────────────────────────────────────────────────────

class FeaturesManager:
    """
    RL state / action bookkeeping.

    STK data-provider calls replaced by:
    - LLA    → propagator r_eci + coordinates.eci_to_ecef + ecef_to_geodetic
    - Orbital elements → propagator.propagate() dict
    """

    def __init__(self, agent: dict, rewarder: "Rewarder"):
        self.class_name   = "Features Manager"
        self.agent_config = agent
        self.rewarder     = rewarder
        self._set_properties(agent)

    def _set_properties(self, agent: dict):
        self.state          = {}
        self.action         = {}
        self.states_features = self._extend_states_features(
            list(agent["states_features"]),
            extendable_groups=[["lat_", "lon_", "priority_"]]
        )
        self.actions_features = agent["actions_features"]
        self.target_memory    = 0
        self.aux_state        = {"detic_lat": None, "detic_lon": None,
                                 "detic_alt": None}
        self.detic_log        = {
            "max_samples": 50, "all_time_counter": 0,
            "prev_lat": [], "prev_lon": [], "prev_alt": [],
            "prev_times": np.array([]),
            "curr_lat": None, "curr_lon": None, "curr_alt": None,
            "counter": 0, "curr_step_gap": 1, "prev_step_gap": 1,
        }

        for state in self.states_features:
            self.state[state] = agent.get(state, None)
            if state.startswith("lat_"):
                self.target_memory += 1

        for action in self.actions_features:
            self.action[action] = 0

    @staticmethod
    def _extend_states_features(
        array: list, extendable_groups: list[list[str]]
    ) -> list:
        max_index = [0] * len(extendable_groups)
        for key in array:
            for i, group in enumerate(extendable_groups):
                if key.startswith(group[0]):
                    idx = int(key.split("_")[1])
                    if idx > max_index[i]:
                        max_index[i] = idx

        for i, group in enumerate(extendable_groups):
            for j in range(1, max_index[i] + 1):
                if f"{group[0]}{j}" in array:
                    for k in range(len(group)):
                        try:
                            array.remove(f"{group[k]}{j}")
                        except ValueError:
                            pass
                array += [f"{group[k]}{j}" for k in range(len(group))]

        return array

    # ── State / action getters & setters ──────────────────────────────────

    def get_state(self, name: Optional[str] = None):
        return self.state if name is None else self.state[name]

    def update_state(self, name: str, value):
        if name not in self.state:
            raise ValueError(f"State variable '{name}' does not exist.")
        self.state[name] = value

    def update_aux_state(self, name: str, value):
        if name not in self.aux_state:
            raise ValueError(f"Aux-state variable '{name}' does not exist.")
        self.aux_state[name] = value

    def get_aux_state(self, name: Optional[str] = None):
        return self.aux_state if name is None else self.aux_state[name]

    def update_action(self, name: str, value):
        if name not in self.action:
            raise ValueError(f"Action '{name}' does not exist.")
        self.action[name] = value

    # ── Orbital element updates ────────────────────────────────────────────

    def update_orbital_elements(self, elems: dict):
        """
        Update state from a dict returned by OrbitalPropagator.propagate().
        """
        if self.agent_config["coordinate_system"] == "Classical":
            for key in ["a", "e", "i", "raan", "aop", "ta"]:
                if key in self.state:
                    self.update_state(key, elems[key])
        elif self.agent_config["coordinate_system"] == "Cartesian":
            for key in ["x", "y", "z", "vx", "vy", "vz"]:
                if key in self.state:
                    self.update_state(key, elems[key])

    # ── Attitude / sensor state updates ──────────────────────────────────

    def update_attitude_state(self, pitch: float, roll: float):
        if "pitch" in self.state:
            self.update_state("pitch", pitch)
        if "roll" in self.state:
            self.update_state("roll", roll)

    def update_sensor_state(self, az: float, el: float):
        if "az" in self.state:
            self.update_state("az", az)
        if "el" in self.state:
            self.update_state("el", el)

    def update_detic_state(self):
        for k in ("detic_lat", "detic_lon", "detic_alt"):
            if k in self.state:
                self.update_state(k, self.aux_state[k])

    # ── LLA computation (replaces STK DataProviders "LLA State") ─────────

    def get_LLA_state(
        self,
        r_eci: np.ndarray,
        et:    float,
    ) -> tuple[float, float, float]:
        """
        Return geodetic (lat°, lon°, alt km) of the satellite at the given ECI
        position and ephemeris time.

        The PCHIP interpolation that was here previously caused a critical bug:
        it was called once per step (not multiple times within a step), making
        every interpolation an *extrapolation* beyond the known data range.
        For an inclined orbit the latitude oscillates sinusoidally; extrapolation
        along the tangent at the last sample produces latitudes far beyond the
        inclination limit (e.g. 74° for a 45° orbit).

        Fix: always compute directly via the full ECI→ECEF→geodetic chain.
        This is the correct approach for per-step LLA updates.
        """
        r_ecef          = eci_to_ecef(r_eci, et)
        lat, lon, alt   = ecef_to_geodetic(r_ecef)
        # Cache in aux_state (update_entire_aux_state reads these)
        self.detic_log["curr_lat"] = lat
        self.detic_log["curr_lon"] = lon
        self.detic_log["curr_alt"] = alt
        self.detic_log["all_time_counter"] += 1
        return float(lat), float(lon), float(alt)

    def update_entire_aux_state(
        self, r_eci: np.ndarray, et: float
    ):
        """Compute and cache LLA from the current ECI position."""
        lat, lon, alt = self.get_LLA_state(r_eci, et)
        self.aux_state["detic_lat"] = lat
        self.aux_state["detic_lon"] = lon
        self.aux_state["detic_alt"] = alt

    # ── Target memory ─────────────────────────────────────────────────────

    def update_target_memory(
        self, preferred_zones: pd.DataFrame, all_zones: pd.DataFrame
    ):
        if not preferred_zones.empty:
            n = min(self.target_memory, preferred_zones.shape[0])
            if "priority" in preferred_zones.columns:
                w = preferred_zones["priority"].astype(float)
                w = w / w.sum()
                seeking = preferred_zones.sample(n, weights=w, replace=True, ignore_index=True)
            else:
                seeking = preferred_zones.sample(n, replace=True, ignore_index=True)

            if n < self.target_memory:
                seeking = pd.concat(
                    [seeking,
                     all_zones.sample(self.target_memory - n, replace=True, ignore_index=True)],
                    ignore_index=True,
                )
        else:
            seeking = all_zones.sample(self.target_memory, replace=True, ignore_index=True)

        for i in range(self.target_memory):
            self.update_state(f"lat_{i+1}",      seeking["lat [deg]"][i])
            self.update_state(f"lon_{i+1}",      seeking["lon [deg]"][i])
            self.update_state(
                f"priority_{i+1}",
                seeking["priority"][i]
                * self.rewarder.f_reobs(seeking["n_obs"][i] + 1),
            )


# ─────────────────────────────────────────────────────────────────────────────
# TargetManager  (no STK objects — targets stored as lat/lon only)
# ─────────────────────────────────────────────────────────────────────────────

class TargetManager:
    """
    Ground-target / event-zone catalogue.

    The STK target COM-objects are replaced by plain dicts / DataFrames.
    Geometric FoR filtering is done via the Haversine formula.
    """

    def __init__(self, start_time: str, n_of_visible_targets: int):
        self.class_name           = "Target Manager"
        self.df                   = pd.DataFrame()
        self.date_mg              = DateManager(start_time, start_time)
        self.newest_time          = start_time
        self.n_of_visible_targets = n_of_visible_targets
        self.max_id               = 0

    # ── Zone management ───────────────────────────────────────────────────

    def append_zone(
        self,
        name: str, target: dict, type_: str,
        lat: float, lon: float, priority: float,
        start_time: str, end_time: str,
        n_obs: int = 0, last_seen: str = "",
    ):
        row = pd.DataFrame({
            "name":               [name],
            "object":             [target],
            "type":               [type_],
            "lat [deg]":          [lat],
            "lon [deg]":          [lon],
            "priority":           [priority],
            "start_time":         [start_time],
            "end_time":           [end_time],
            "numeric_start_date": [self.date_mg.num_of_date(
                                       self.date_mg.simplify_date(start_time))],
            "numeric_end_date":   [self.date_mg.num_of_date(
                                       self.date_mg.simplify_date(end_time))],
            "n_obs":              [n_obs],
            "last seen":          [last_seen],
        })
        self.df     = pd.concat([self.df, row], ignore_index=True)
        self.max_id += 1

    def erase_zone(self, name: str):
        self.df = self.df[self.df["name"] != name]

    def get_zone_by_name(self, name: str) -> pd.DataFrame:
        z = self.df[self.df["name"] == name]
        if z.empty:
            raise ValueError(f"Zone '{name}' not found.")
        if z.shape[0] > 1:
            raise ValueError(f"Zone '{name}' appears multiple times.")
        return z

    def get_n_obs(self, name: str) -> int:
        return int(self.get_zone_by_name(name)["n_obs"].values[0])

    def get_last_seen(self, name: str) -> str:
        return str(self.get_zone_by_name(name)["last seen"].values[0])

    def get_priority(self, name: str) -> float:
        return float(self.get_zone_by_name(name)["priority"].values[0])

    def plus_one_obs(self, name: str):
        self.df.loc[self.df["name"] == name, "n_obs"] += 1

    def update_last_seen(self, name: str, date: str):
        self.df.loc[self.df["name"] == name, "last seen"] = date

    # ── Expiry helpers ─────────────────────────────────────────────────────

    def n_of_zones_to_add(self, time: str) -> int:
        if self.date_mg.is_newer_than(time, self.newest_time):
            self.newest_time = time
            t = self.date_mg.num_of_date(self.date_mg.simplify_date(time))
            unloadable = self.get_unloadable_zones_before(t)
            return self.n_of_visible_targets - self.df.shape[0] + unloadable.shape[0]
        return 0

    def get_unloadable_zones_before(self, lowest_et: float) -> pd.DataFrame:
        return self.df[self.df["numeric_end_date"] < lowest_et]

    def unload_zones_before(self, lowest_et: float):
        self.df = self.df[self.df["numeric_end_date"] >= lowest_et]

    # ── Field of Regard filtering ─────────────────────────────────────────

    def get_FoR_window_df(
        self,
        date_mg:     DateManager,
        features_mg: FeaturesManager,
        margin_pct:  float = 10.0,
        return_D_FoR: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, float]:
        last_et    = date_mg.num_of_date(date_mg.simplify_date(date_mg.last_date))
        current_et = date_mg.num_of_date(date_mg.simplify_date(date_mg.current_date))

        win = self.df[self.df["numeric_end_date"]   >= last_et]
        win = win[win["numeric_start_date"] <= current_et]

        detic_lat = features_mg.get_aux_state("detic_lat")
        detic_lon = features_mg.get_aux_state("detic_lon")
        detic_alt = features_mg.get_aux_state("detic_alt")

        win = win.copy()
        win["distance"] = win.apply(
            lambda r: self.haversine(
                detic_lat, detic_lon, r["lat [deg]"], r["lon [deg]"]
            ), axis=1
        )

        D_FoR  = self.calculate_D_FoR(detic_alt)
        margin = D_FoR * (1.0 + margin_pct / 100.0)
        win    = win[win["distance"] <= margin]

        if return_D_FoR:
            return win, margin
        return win

    def calculate_D_FoR(self, altitude: float) -> float:
        return RT * np.arccos(np.clip(RT / (RT + altitude), -1.0, 1.0))

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2) -> float:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return RT * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    @staticmethod
    def haversine_angle(lat1, lon1, lat2, lon2) -> float:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


# ─────────────────────────────────────────────────────────────────────────────
# GridManager  (pure-Python, no STK coverage grids)
# ─────────────────────────────────────────────────────────────────────────────

class GridManager:
    """
    Simple lat/lon coverage grid.

    STK's CoverageDefinition / FigureOfMerit is replaced by checking
    whether each grid point falls inside the sensor cone on the ground,
    using the AccessChecker.
    """

    def __init__(self, resolution_deg: float = 5.0):
        self.class_name    = "Grid Manager"
        self.resolution    = resolution_deg
        self.grid_data: Optional[dict] = None

    def build_global_grid(self):
        """Precompute all lat/lon grid points."""
        lats = np.arange(-90.0 + self.resolution/2,
                          90.0, self.resolution)
        lons = np.arange(-180.0 + self.resolution/2,
                          180.0, self.resolution)
        lon_g, lat_g = np.meshgrid(lons, lats)
        self.grid_data = {
            "lats [deg]": lat_g.ravel(),
            "lons [deg]": lon_g.ravel(),
        }

    def get_seen_points(
        self,
        r_sat_eci:   np.ndarray,
        v_sat_eci:   np.ndarray,
        et:          float,
        attitude_mg: AttitudeManager,
        sensor_mg:   SensorManager,
    ) -> list[tuple[float, float]]:
        """
        Return the list of (lat, lon) grid points currently inside the
        sensor footprint.  Uses vectorised geometry for speed.
        """
        if self.grid_data is None:
            self.build_global_grid()

        from scripts.coordinates import geodetic_to_eci

        lats = self.grid_data["lats [deg]"]
        lons = self.grid_data["lons [deg]"]

        # Sensor boresight in ECI
        body_to_eci   = attitude_mg.body_to_eci_matrix(r_sat_eci, v_sat_eci)
        boresight_eci = body_to_eci @ sensor_mg.boresight_body()

        cos_cone = np.cos(np.radians(sensor_mg.cone_angle))

        # Vectorised check for all grid points
        r_sat_u = r_sat_eci / np.linalg.norm(r_sat_eci)
        seen = []
        for lat, lon in zip(lats, lons):
            r_tgt = geodetic_to_eci(lat, lon, 0.0, et)
            d     = r_tgt - r_sat_eci
            d_u   = d / np.linalg.norm(d)
            if np.dot(boresight_eci, d_u) >= cos_cone:
                seen.append((float(lat), float(lon)))

        return seen


# ─────────────────────────────────────────────────────────────────────────────
# Rewarder  (same logic; access intervals now come from AccessChecker)
# ─────────────────────────────────────────────────────────────────────────────

class Rewarder:
    """
    Reward calculator.

    The STK data-provider API is replaced: instead of access_data_provider
    and aer_data_provider objects the method receives plain Python dicts.
    """

    def __init__(self, agents_config: dict, target_mg: TargetManager):
        self.class_name    = "Rewarder"
        self.seen_events   = []
        self.target_mg     = target_mg
        self.agents_config = agents_config

    def calculate_reward(
        self,
        access_events:   list[dict],
        delta_time:      float,
        grid_points_seen: Optional[list[tuple[float, float]]],
        FoR_window_df:   pd.DataFrame,
        D_FoR:           float,
        date_mg:         DateManager,
        sensor_mg:       SensorManager,
        features_mg:     FeaturesManager,
        angle_domains:   dict,
        attitude_max_slew: float = 10.0,   # BUG FIX 3: needed for pitch/roll slew check
    ) -> float:
        """
        Parameters
        ----------
        access_events : list of dicts, each with keys:
            "name"         str   — target name
            "start_time"   str   — STK date of access start
            "stop_time"    str   — STK date of access stop
            "elevation"    float — best elevation angle (deg) during access
        grid_points_seen : list of (lat, lon) tuples, or None
        (remaining params match the original signature)
        """
        reward = 0.0

        reward += (self.slew_constraint(delta_time, sensor_mg,
                                        features_mg, angle_domains,
                                        attitude_max_slew)
                   * self.agents_config["slew_weight"])

        if self.agents_config["use_grid"] and grid_points_seen is not None:
            reward += self.grid_rewards(grid_points_seen, FoR_window_df, D_FoR)

        min_duration = self.agents_config["min_duration"]

        for ev in access_events:
            event_name     = ev["name"]
            start_et       = date_mg.num_of_date(
                date_mg.simplify_date(ev["start_time"]))
            stop_et        = date_mg.num_of_date(
                date_mg.simplify_date(ev["stop_time"]))
            max_zen_angle  = abs(float(ev["elevation"]))

            print(f"\nEvent: {event_name}")
            print(f"Seen from {ev['start_time']} to {ev['stop_time']}.")
            print(f"Duration: {stop_et - start_et:.1f} s")

            if (stop_et - start_et) > min_duration:
                zone_n_obs    = self.target_mg.get_n_obs(event_name)
                zone_last_seen = self.target_mg.get_last_seen(event_name)
                zone_priority = self.target_mg.get_priority(event_name)

                if zone_n_obs != 0:
                    if zone_n_obs < 0:
                        raise ValueError(
                            "Number of observations cannot be negative."
                        )
                    last_seen_et = date_mg.num_of_date(
                        date_mg.simplify_date(zone_last_seen))
                    self.target_mg.update_last_seen(event_name, ev["stop_time"])

                    if (last_seen_et - min_duration) < start_et:
                        self.target_mg.plus_one_obs(event_name)
                        n_obs = self.target_mg.get_n_obs(event_name)
                        ri    = self.f_ri(zone_priority, max_zen_angle, n_obs)
                        reward += ri
                        print(f"Observed {event_name} with el={max_zen_angle:.2f}° "
                              f"reward={ri:.4f} (total={reward:.4f}).")
                    else:
                        print(f"Observation of {event_name} belongs to "
                              "a previous interval — not counted.")
                else:
                    self.target_mg.plus_one_obs(event_name)
                    self.target_mg.update_last_seen(event_name, ev["stop_time"])
                    ri = self.f_ri(zone_priority, max_zen_angle, 1)
                    reward += ri
                    print(f"First observed {event_name} with el={max_zen_angle:.2f}° "
                          f"reward={ri:.4f} (total={reward:.4f}).")
            else:
                print(f"Observation of {event_name} has insufficient duration.")

        print()
        return reward

    def f_ri(self, priority: float, max_zen_angle: float, n_obs: int) -> float:
        return (priority ** self.agents_config["priority_weight"]
                * self.f_reobs(n_obs)
                * self.f_theta(max_zen_angle))

    def f_theta(self, max_zen_angle: float) -> float:
        return math.sin(math.radians(max_zen_angle)) ** self.agents_config["zenith_weight"]

    def f_reobs(self, n_obs: int) -> float:
        return (1.0 / n_obs ** self.agents_config["reobs_decay"]) if n_obs > 0 else 1.0

    def slew_constraint(
        self,
        delta_time:  float,
        sensor_mg:   SensorManager,
        features_mg: FeaturesManager,
        angle_domains: dict,
        attitude_max_slew: float = 10.0,
    ) -> float:
        # BUG FIX 3: The original code only awarded the +5 / -10 signal for
        # d_az / d_el actions, never for d_pitch / d_roll.  The agent
        # therefore received no positive reward from staying within attitude
        # slew limits, leaving a systematic gap in the reward signal.
        #
        # Additionally the az/el rate check divided by delta_time (thousands
        # of seconds), making the computed rate always far below max_slew and
        # thus always awarding +5 regardless of the actual slew magnitude.
        # We now compare the action magnitude directly to max_slew (deg/step),
        # which is the natural unit for the simplified (non-kinematic) model.
        r = 0.0
        for diff in features_mg.action:
            value    = features_mg.action[diff]
            # Second-line NaN guard: features_mg.action should always be clean
            # after the sanitisation in _update_agent, but defend here too so
            # a future code path cannot bypass the upstream check.
            if not np.isfinite(value):
                value = 0.0
            movement = abs(value)

            if diff in ("d_az", "d_el"):
                r += -10.0 if movement > sensor_mg.max_slew else 5.0
            elif diff in ("d_pitch", "d_roll"):
                r += -10.0 if movement > attitude_max_slew else 5.0

            key    = diff.split("_")[1]   # "pitch", "roll", "az", or "el"
            domain = abs(angle_domains.get(key, [-180, 180])[1]
                        - angle_domains.get(key, [-180, 180])[0])
            r += -movement / domain
        return r

    def grid_rewards(
        self,
        grid_points_seen: list[tuple[float, float]],
        FoR_window_df:    pd.DataFrame,
        D_FoR:            float,
    ) -> float:
        reward = 0.0
        FoR_window_df = FoR_window_df.reset_index(drop=True)
        for lat, lon in grid_points_seen:
            for i in range(FoR_window_df.shape[0]):
                distance   = self.target_mg.haversine(
                    lat, lon,
                    FoR_window_df["lat [deg]"][i],
                    FoR_window_df["lon [deg]"][i],
                )
                if distance < D_FoR:
                    ev_name    = FoR_window_df["name"][i]
                    n_obs      = self.target_mg.get_n_obs(ev_name)
                    priority   = self.target_mg.get_priority(ev_name)
                    A  = self.agents_config["grid_weight"] * priority * self.f_reobs(n_obs)
                    B  = np.pi / (2.0 * D_FoR)
                    r  = A * np.cos(B * distance) ** self.agents_config["grid_decay"]
                    reward += max(r, 0.0)
        return reward


# ─────────────────────────────────────────────────────────────────────────────
# Plotter  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class Plotter:
    """Plots and exports training reward curves."""

    def __init__(self, out_folder_path: str = "output"):
        self.class_name     = "Plotter"
        self.rewards        = pd.DataFrame()
        self.out_folder_path = out_folder_path

    def store_reward(self, reward: float):
        self.rewards = pd.concat(
            [self.rewards, pd.DataFrame([reward])], ignore_index=True
        )

    def _save(self, filename: str):
        os.makedirs(self.out_folder_path, exist_ok=True)
        plt.savefig(f"{self.out_folder_path}/{filename}", dpi=500)

    def plot_rewards(self):
        if self.rewards.empty:
            raise ValueError("No rewards to plot.")
        plt.clf()
        plt.plot(self.rewards)
        plt.xlabel("Step"); plt.ylabel("Reward"); plt.title("Rewards over time")
        self._save("rewards.png")

    def plot_rewards_smoothed(self, window_size: int = 0):
        if self.rewards.empty:
            raise ValueError("No rewards to plot.")
        w = self._correct_window(window_size)
        plt.clf()
        plt.plot(self.rewards.rolling(w).mean())
        plt.xlabel("Step"); plt.ylabel("Smoothed reward")
        plt.title("Smoothed rewards")
        self._save("rewards_smoothed.png")

    def plot_cumulative_rewards(self):
        if self.rewards.empty:
            raise ValueError("No rewards to plot.")
        plt.clf()
        plt.plot(self.rewards.cumsum())
        plt.xlabel("Episode"); plt.ylabel("Cumulative reward")
        plt.title("Cumulative reward over time")
        self._save("cumulative_rewards.png")

    def plot_cumulative_rewards_smoothed_per_steps(self, window_size: int = 10):
        if self.rewards.empty:
            raise ValueError("No rewards to plot.")
        w = self._correct_window(window_size)
        plt.clf()
        cum = self.rewards.rolling(w).mean().cumsum()
        cum = cum.div(pd.Series(range(1, len(cum) + 1)), axis=0)
        plt.plot(cum)
        plt.xlabel("Step"); plt.ylabel("Cumulative reward / step")
        plt.title("Cumulative reward per step")
        self._save("cumulative_rewards_smoothed_per_steps.png")

    def plot_all(self, window_size: int = 0):
        self.plot_rewards()
        self.plot_rewards_smoothed(window_size)
        self.plot_cumulative_rewards()
        self.plot_cumulative_rewards_smoothed_per_steps(window_size)
        os.makedirs(self.out_folder_path, exist_ok=True)
        self.rewards.to_csv(f"{self.out_folder_path}/rewards.csv", index=False)

    def _correct_window(self, w: int) -> int:
        if w == 0:
            w = max(1, len(self.rewards) // 10)
        return max(1, w)
