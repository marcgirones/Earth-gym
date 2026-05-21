"""
scripts/propagator_base.py
==========================
Abstract base class that every propagator must implement.

Both OrbitalPropagator (analytical) and SGP4Propagator (sgp4 library)
inherit from this class, so instances.py can use either interchangeably.

Contract
--------
__init__(agent_config, propagator_type, epoch_et)
    agent_config    : flattened agent dict from DataFromJSON.get_dict()
    propagator_type : string key identifying the model
    epoch_et        : simulation start time in ephemeris time (s past J2000)

propagate(t_et) -> (r_eci, v_eci, elems)
    t_et    : ephemeris time to propagate to (s past J2000)
    r_eci   : (3,) ECI position  (km)
    v_eci   : (3,) ECI velocity  (km s⁻¹)
    elems   : dict with keys a, e, i, raan, aop, ta (angles in deg, km, km/s)
              plus x, y, z, vx, vy, vz for convenience
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BasePropagator(ABC):
    """Abstract orbit propagator. All propagators must implement propagate()."""

    #: Set of propagator-type strings recognised by this class.
    SUPPORTED_TYPES: set[str] = set()

    @abstractmethod
    def __init__(
        self,
        agent_config: dict,
        propagator_type: str,
        epoch_et: float,
    ) -> None: ...

    @abstractmethod
    def propagate(
        self, t_et: float
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Propagate the orbit to ephemeris time *t_et*.

        Returns
        -------
        r_eci  : (3,) position  (km)      — ECI / J2000 frame
        v_eci  : (3,) velocity  (km s⁻¹)  — ECI / J2000 frame
        elems  : dict
            a    (km)   semi-major axis
            e    (—)    eccentricity
            i    (deg)  inclination
            raan (deg)  right ascension of ascending node
            aop  (deg)  argument of perigee
            ta   (deg)  true anomaly
            x, y, z     (km)    ECI position components
            vx, vy, vz  (km/s)  ECI velocity components
        """
        ...


# ── Factory ───────────────────────────────────────────────────────────────────

def make_propagator(
    agent_config: dict,
    propagator_type: str,
    epoch_et: float,
) -> BasePropagator:
    """
    Instantiate the correct propagator for *propagator_type*.

    Analytical types  →  OrbitalPropagator  (propagator.py)
        "TwoBody", "J2Perturbation", "J4Perturbation"

    SGP4 types        →  SGP4Propagator     (propagator_sgp4.py)
        "SGP4", "SGP4-TLE"

    Parameters
    ----------
    agent_config    : flattened agent dict
    propagator_type : one of the strings above
    epoch_et        : simulation epoch (s past J2000)

    Returns
    -------
    A BasePropagator instance ready to call .propagate(t_et) on.
    """
    _ANALYTICAL = {"TwoBody", "J2Perturbation", "J4Perturbation"}
    _SGP4       = {"SGP4", "SGP4-TLE"}

    if propagator_type in _ANALYTICAL:
        from scripts.propagator import OrbitalPropagator
        return OrbitalPropagator(agent_config, propagator_type, epoch_et)

    if propagator_type in _SGP4:
        from scripts.propagator_sgp4 import SGP4Propagator
        return SGP4Propagator(agent_config, propagator_type, epoch_et)

    raise ValueError(
        f"Unknown propagator_type '{propagator_type}'. "
        f"Valid options: {sorted(_ANALYTICAL | _SGP4)}"
    )
