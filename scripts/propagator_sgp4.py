"""
scripts/propagator_sgp4.py
==========================
SGP4-based orbit propagator, drop-in replacement for OrbitalPropagator.

Backed by Brandon Rhodes' sgp4 library (WGS-84 gravity model, improved
opsmode).  The output interface is identical to OrbitalPropagator:

    prop = SGP4Propagator(agent_config, "SGP4", epoch_et)
    r_eci, v_eci, elems = prop.propagate(t_et)

Supported propagator_type strings
----------------------------------
"SGP4"       Initialise from classical elements in agents-configuration.json.
             The osculating elements are converted to SGP4 mean elements via
             a first-order J2 Kozai-Brouwer transform (see tle_utils.py).

"SGP4-TLE"   Initialise from raw TLE strings supplied in the agent config:
               "tle_line1": "1 25544U ..."
               "tle_line2": "2 25544 ..."
             When using this mode the epoch and orbital elements come from the
             TLE itself; initial elements in the JSON are ignored except for
             "coordinate_system" (used only for the elems dict format).

Frame notes
-----------
SGP4 outputs states in TEME (True Equator Mean Equinox).  teme_to_eci() in
tle_utils.py rotates them to J2000 (GCRF) using IAU 76/FK5 precession +
dominant IAU 1980 nutation terms.  Residual error is < 30 m at LEO for
propagation spans up to ~14 days — negligible for RL training.
"""

from __future__ import annotations

import math
import numpy as np

from scripts.propagator_base import BasePropagator
from scripts.tle_utils import (
    elements_to_satrec,
    tle_to_satrec,
    teme_to_eci,
    et_to_jd,
    satrec_epoch_et,
    _J2000_JD,
)
from scripts.propagator import cartesian_to_classical   # re-use existing helper


class SGP4Propagator(BasePropagator):
    """
    Orbit propagator backed by the sgp4 library.

    Parameters
    ----------
    agent_config    : flattened agent dict from DataFromJSON.get_dict()
    propagator_type : "SGP4" or "SGP4-TLE"
    epoch_et        : simulation start time (s past J2000)
    """

    SUPPORTED_TYPES = {"SGP4", "SGP4-TLE"}

    def __init__(
        self,
        agent_config: dict,
        propagator_type: str,
        epoch_et: float,
    ) -> None:
        if propagator_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"SGP4Propagator does not support '{propagator_type}'. "
                f"Choose from {self.SUPPORTED_TYPES}."
            )

        self.propagator_type = propagator_type
        self.epoch_et        = epoch_et
        self.coord_system    = agent_config.get("coordinate_system", "Classical")

        # ── Build Satrec ──────────────────────────────────────────────────
        if propagator_type == "SGP4-TLE":
            self._init_from_tle(agent_config)
        else:
            self._init_from_elements(agent_config, epoch_et)

    # ── Initialisation helpers ────────────────────────────────────────────

    def _init_from_elements(self, agent_config: dict, epoch_et: float) -> None:
        """
        Build Satrec from classical osculating elements in the config dict.
        Handles both "Classical" and "Cartesian" coordinate systems.
        """
        if self.coord_system == "Classical":
            a        = float(agent_config["a"])
            e        = float(agent_config["e"])
            i_deg    = float(agent_config["i"])
            raan_deg = float(agent_config["raan"])
            aop_deg  = float(agent_config["aop"])
            ta_deg   = float(agent_config["ta"])

        elif self.coord_system == "Cartesian":
            r0 = np.array([float(agent_config["x"]),
                           float(agent_config["y"]),
                           float(agent_config["z"])])
            v0 = np.array([float(agent_config["vx"]),
                           float(agent_config["vy"]),
                           float(agent_config["vz"])])
            a, e, i_rad, raan_rad, aop_rad, ta_rad = cartesian_to_classical(r0, v0)
            i_deg    = math.degrees(i_rad)
            raan_deg = math.degrees(raan_rad)
            aop_deg  = math.degrees(aop_rad)
            ta_deg   = math.degrees(ta_rad)
        else:
            raise ValueError(
                f"Unknown coordinate_system '{self.coord_system}'. "
                "Use 'Classical' or 'Cartesian'."
            )

        bstar = float(agent_config.get("bstar", 0.0))

        self._satrec = elements_to_satrec(
            a, e, i_deg, raan_deg, aop_deg, ta_deg, epoch_et, bstar
        )

        # Store osculating elements for reference
        self._a0    = a
        self._e0    = e
        self._i0    = i_deg
        self._raan0 = raan_deg
        self._aop0  = aop_deg
        self._ta0   = ta_deg

    def _init_from_tle(self, agent_config: dict) -> None:
        """Parse TLE strings from the agent config."""
        try:
            line1 = agent_config["tle_line1"]
            line2 = agent_config["tle_line2"]
        except KeyError as err:
            raise ValueError(
                "propagator_type 'SGP4-TLE' requires 'tle_line1' and "
                f"'tle_line2' in the agent config. Missing: {err}"
            ) from err

        self._satrec = tle_to_satrec(line1, line2)

        # Override epoch_et to match the TLE epoch
        self.epoch_et = satrec_epoch_et(self._satrec)

        # Store reference elements from TLE (mean, not osculating)
        self._a0    = None    # not directly available from TLE
        self._e0    = self._satrec.ecco
        self._i0    = math.degrees(self._satrec.inclo)
        self._raan0 = math.degrees(self._satrec.nodeo)
        self._aop0  = math.degrees(self._satrec.argpo)
        self._ta0   = None    # mean anomaly stored instead

    # ── Main propagation method ───────────────────────────────────────────

    def propagate(
        self, t_et: float
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Propagate the orbit to ephemeris time *t_et*.

        Parameters
        ----------
        t_et : seconds past J2000

        Returns
        -------
        r_eci  : (3,) ECI (J2000) position (km)
        v_eci  : (3,) ECI (J2000) velocity (km s⁻¹)
        elems  : dict with a, e, i, raan, aop, ta (deg), x, y, z, vx, vy, vz
        """
        # ── Julian date split for sgp4 API ───────────────────────────────
        # Splitting as (J2000 JD, fractional days since J2000) is
        # numerically stable for dates near J2000.
        jd = _J2000_JD
        fr = t_et / 86_400.0

        # ── Call sgp4 ─────────────────────────────────────────────────────
        err, r_teme, v_teme = self._satrec.sgp4(jd, fr)

        if err != 0:
            from sgp4.api import SGP4_ERRORS
            raise RuntimeError(
                f"SGP4 propagation failed at ET={t_et:.1f} s "
                f"(err={err}: {SGP4_ERRORS.get(err, 'unknown')}). "
                "The satellite may have decayed or the time is outside the "
                "valid propagation range."
            )

        r_teme = np.array(r_teme)
        v_teme = np.array(v_teme)  # km/s already

        # ── TEME → J2000 (ECI) ───────────────────────────────────────────
        r_eci, v_eci = teme_to_eci(r_teme, v_teme, t_et)

        # ── Derive classical elements from Cartesian ──────────────────────
        a, e, inc_rad, raan_rad, aop_rad, ta_rad = cartesian_to_classical(
            r_eci, v_eci
        )

        elems = {
            "a":    a,
            "e":    e,
            "i":    math.degrees(inc_rad),
            "raan": math.degrees(raan_rad),
            "aop":  math.degrees(aop_rad),
            "ta":   math.degrees(ta_rad),
            "x":  r_eci[0], "y":  r_eci[1], "z":  r_eci[2],
            "vx": v_eci[0], "vy": v_eci[1], "vz": v_eci[2],
        }
        return r_eci, v_eci, elems

    # ── Convenience properties ────────────────────────────────────────────

    @property
    def satrec(self):
        """Underlying sgp4 Satrec object (for advanced use)."""
        return self._satrec

    @property
    def bstar(self) -> float:
        """B* drag coefficient (m⁻¹)."""
        return self._satrec.bstar

    @property
    def tle_epoch_et(self) -> float:
        """Epoch of the underlying TLE / sgp4init call (s past J2000)."""
        return satrec_epoch_et(self._satrec)
