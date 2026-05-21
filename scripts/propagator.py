"""
scripts/propagator.py
=====================
Analytical orbit propagator replacing STK's built-in propagators.

Supported models
----------------
TwoBody         — pure Keplerian, no perturbations
J2Perturbation  — secular J2 oblateness effects (RAAN, AoP, M drift)
J4Perturbation  — secular J2 + J4 oblateness effects

All angles are stored internally in radians; the public API mirrors the
STK convention (degrees for angles, km for distances, km/s for velocities).
"""

from __future__ import annotations
import numpy as np
from scripts.propagator_base import BasePropagator

# ─── Earth constants (WGS-84) ────────────────────────────────────────────────
MU = 398600.4418     # km³ s⁻²  gravitational parameter
RE = 6378.137        # km       equatorial radius
J2 = 1.08262668e-3   # —        2nd zonal harmonic
J4 = -1.62336e-6     # —        4th zonal harmonic (already negative in WGS-84)


# ─── Low-level math helpers ──────────────────────────────────────────────────

def _true_to_eccentric(ta: float, e: float) -> float:
    """True anomaly → eccentric anomaly (rad)."""
    return 2.0 * np.arctan2(
        np.sqrt(1.0 - e) * np.sin(ta / 2.0),
        np.sqrt(1.0 + e) * np.cos(ta / 2.0),
    )


def _eccentric_to_true(E: float, e: float) -> float:
    """Eccentric anomaly → true anomaly (rad)."""
    return 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E / 2.0),
        np.sqrt(1.0 - e) * np.cos(E / 2.0),
    )


def _true_to_mean(ta: float, e: float) -> float:
    """True anomaly → mean anomaly (rad, in [0, 2π])."""
    E = _true_to_eccentric(ta, e)
    return (E - e * np.sin(E)) % (2.0 * np.pi)


def solve_kepler(M: float | np.ndarray, e: float,
                 tol: float = 1e-10, max_iter: int = 50) -> float | np.ndarray:
    """
    Solve Kepler's equation  M = E − e·sin(E)  by Newton–Raphson.

    Parameters
    ----------
    M        : mean anomaly (rad) — scalar or ndarray
    e        : eccentricity
    tol      : convergence tolerance (rad)
    max_iter : maximum iterations

    Returns
    -------
    E : eccentric anomaly (rad)
    """
    E = np.asarray(M, dtype=float).copy()
    for _ in range(max_iter):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if np.max(np.abs(dE)) < tol:
            break
    return float(E) if np.ndim(E) == 0 else E


def _perifocal_to_eci_matrix(raan: float, inc: float, aop: float) -> np.ndarray:
    """
    3×3 rotation matrix from the perifocal (PQW) frame to ECI (J2000/ICRF).
    Uses the classical 3-1-3 Euler sequence: Rz(−Ω)·Rx(−i)·Rz(−ω).
    """
    co, so = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc),  np.sin(inc)
    cw, sw = np.cos(aop),  np.sin(aop)
    return np.array([
        [ co*cw - so*sw*ci,  -co*sw - so*cw*ci,  so*si],
        [ so*cw + co*sw*ci,  -so*sw + co*cw*ci, -co*si],
        [ sw*si,              cw*si,              ci   ],
    ])


def classical_to_cartesian(
    a: float, e: float, inc: float, raan: float, aop: float, ta: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert classical orbital elements to ECI Cartesian state.

    Parameters (angles in **radians**)
    ----------
    a    : semi-major axis (km)
    e    : eccentricity
    inc  : inclination (rad)
    raan : right ascension of ascending node (rad)
    aop  : argument of perigee (rad)
    ta   : true anomaly (rad)

    Returns
    -------
    r_eci : (3,) position  (km)
    v_eci : (3,) velocity  (km s⁻¹)
    """
    p = a * (1.0 - e**2)
    r_mag = p / (1.0 + e * np.cos(ta))

    r_pf = r_mag * np.array([np.cos(ta), np.sin(ta), 0.0])
    v_pf = np.sqrt(MU / p) * np.array([-np.sin(ta), e + np.cos(ta), 0.0])

    Q = _perifocal_to_eci_matrix(raan, inc, aop)
    return Q @ r_pf, Q @ v_pf


def cartesian_to_classical(
    r: np.ndarray, v: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    """
    Convert ECI Cartesian state to classical orbital elements.

    Returns
    -------
    a (km), e, inc (rad), raan (rad), aop (rad), ta (rad)
    """
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    h_vec = np.cross(r, v)
    h_mag = np.linalg.norm(h_vec)
    n_vec = np.cross(np.array([0.0, 0.0, 1.0]), h_vec)
    n_mag = np.linalg.norm(n_vec)

    eps = v_mag**2 / 2.0 - MU / r_mag
    a   = -MU / (2.0 * eps)

    e_vec = np.cross(v, h_vec) / MU - r / r_mag
    e     = np.linalg.norm(e_vec)

    inc  = np.arccos(np.clip(h_vec[2] / h_mag, -1.0, 1.0))

    raan = np.arccos(np.clip(n_vec[0] / n_mag, -1.0, 1.0))
    if n_vec[1] < 0.0:
        raan = 2.0 * np.pi - raan

    aop  = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n_mag * e), -1.0, 1.0))
    if e_vec[2] < 0.0:
        aop = 2.0 * np.pi - aop

    ta   = np.arccos(np.clip(np.dot(e_vec, r) / (e * r_mag), -1.0, 1.0))
    if np.dot(r, v) < 0.0:
        ta = 2.0 * np.pi - ta

    return a, e, inc, raan, aop, ta


# ─── Propagator class ────────────────────────────────────────────────────────

class OrbitalPropagator(BasePropagator):
    """
    Analytical orbit propagator (drop-in replacement for STK's propagator).

    Usage
    -----
    >>> prop = OrbitalPropagator(agent_config, "J2Perturbation", epoch_et)
    >>> r, v, elems = prop.propagate(t_et)   # t_et: seconds past J2000
    """

    SUPPORTED_TYPES = {"TwoBody", "J2Perturbation", "J4Perturbation"}

    def __init__(
        self,
        agent_config: dict,
        propagator_type: str,
        epoch_et: float,
    ) -> None:
        """
        Parameters
        ----------
        agent_config    : flattened agent dict (output of DataFromJSON.get_dict())
        propagator_type : "TwoBody" | "J2Perturbation" | "J4Perturbation"
        epoch_et        : simulation start time as ephemeris time (s past J2000)
        """
        if propagator_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unknown propagator '{propagator_type}'. "
                f"Choose from {self.SUPPORTED_TYPES}."
            )
        self.propagator_type = propagator_type
        self.epoch_et        = epoch_et
        self.coord_system    = agent_config["coordinate_system"]

        # ── Parse initial orbital elements ───────────────────────────────
        if self.coord_system == "Classical":
            self._a0    = float(agent_config["a"])
            self._e0    = float(agent_config["e"])
            self._inc0  = np.radians(float(agent_config["i"]))
            self._raan0 = np.radians(float(agent_config["raan"]))
            self._aop0  = np.radians(float(agent_config["aop"]))
            self._ta0   = np.radians(float(agent_config["ta"]))
        elif self.coord_system == "Cartesian":
            r0 = np.array([float(agent_config["x"]),
                           float(agent_config["y"]),
                           float(agent_config["z"])])
            v0 = np.array([float(agent_config["vx"]),
                           float(agent_config["vy"]),
                           float(agent_config["vz"])])
            (self._a0, self._e0,
             self._inc0, self._raan0,
             self._aop0, self._ta0) = cartesian_to_classical(r0, v0)
        else:
            raise ValueError(
                f"Unknown coordinate_system '{self.coord_system}'. "
                "Use 'Classical' or 'Cartesian'."
            )

        self._n0 = np.sqrt(MU / self._a0**3)              # mean motion (rad/s)
        self._M0 = _true_to_mean(self._ta0, self._e0)     # mean anomaly at epoch

        # ── Secular perturbation drift rates (rad/s) ─────────────────────
        self._raan_dot   = 0.0
        self._aop_dot    = 0.0
        self._M_dot_corr = 0.0  # correction to n0 due to oblateness

        if propagator_type in {"J2Perturbation", "J4Perturbation"}:
            self._add_j2_rates()
        if propagator_type == "J4Perturbation":
            self._add_j4_rates()

    # ── Secular drift helpers ────────────────────────────────────────────

    def _add_j2_rates(self) -> None:
        a, e, i = self._a0, self._e0, self._inc0
        p   = a * (1.0 - e**2)
        fac = (3.0 / 2.0) * J2 * (RE / p)**2
        eta = np.sqrt(1.0 - e**2)

        self._raan_dot   += -self._n0 * fac * np.cos(i)
        self._aop_dot    +=  self._n0 * fac * (2.0 - 2.5 * np.sin(i)**2)
        self._M_dot_corr +=  self._n0 * fac * eta * (1.0 - 1.5 * np.sin(i)**2)

    def _add_j4_rates(self) -> None:
        a, e, i = self._a0, self._e0, self._inc0
        p   = a * (1.0 - e**2)
        eta = np.sqrt(1.0 - e**2)
        # J4 < 0 in WGS-84 convention
        fac4 = -(15.0 / 32.0) * J4 * (RE / p)**4

        self._raan_dot   += self._n0 * fac4 * np.cos(i) * (1.0 - 1.25 * np.sin(i)**2)
        self._aop_dot    += self._n0 * fac4 * (4.0/3.0 - 5.0 * np.sin(i)**2
                                                + 8.75 * np.sin(i)**4)
        self._M_dot_corr += self._n0 * fac4 * eta * (1.0 - 3.75 * np.sin(i)**2
                                                      + 4.375 * np.sin(i)**4)

    # ── Main propagation method ──────────────────────────────────────────

    def propagate(self, t_et: float) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Propagate the orbit to ephemeris time *t_et*.

        Parameters
        ----------
        t_et : seconds past J2000

        Returns
        -------
        r_eci  : (3,) position vector (km)
        v_eci  : (3,) velocity vector (km s⁻¹)
        elems  : dict of orbital elements
                 angles in degrees, distances in km / km s⁻¹
        """
        dt = t_et - self.epoch_et

        a    = self._a0
        e    = self._e0
        inc  = self._inc0
        raan = self._raan0 + self._raan_dot * dt
        aop  = self._aop0  + self._aop_dot  * dt
        M    = (self._M0 + (self._n0 + self._M_dot_corr) * dt) % (2.0 * np.pi)

        E  = solve_kepler(M, e)
        ta = _eccentric_to_true(E, e)

        r_eci, v_eci = classical_to_cartesian(a, e, inc, raan, aop, ta)

        elems = {
            "a":    a,
            "e":    e,
            "i":    np.degrees(inc),
            "raan": np.degrees(raan),
            "aop":  np.degrees(aop),
            "ta":   np.degrees(ta),
            "x":  r_eci[0], "y":  r_eci[1], "z":  r_eci[2],
            "vx": v_eci[0], "vy": v_eci[1], "vz": v_eci[2],
        }
        return r_eci, v_eci, elems
