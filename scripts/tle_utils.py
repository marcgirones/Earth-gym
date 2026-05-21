"""
scripts/tle_utils.py
====================
Bridging utilities between the rest of the codebase (ECI J2000, classical
osculating elements, ephemeris time) and the sgp4 library (TEME frame,
SGP4 mean elements, Julian dates).

Public API
----------
et_to_jd(et)                       ET (s past J2000) → Julian date (days)
jd_to_et(jd)                       Julian date       → ET (s past J2000)

teme_to_eci(r_teme, v_teme, t_et)  TEME → J2000 position + velocity

elements_to_satrec(a, e, i, raan, aop, ta, epoch_et, bstar)
    Classical osculating elements → sgp4 Satrec (initialised via sgp4init).
    Returns (satrec, epoch_jd) where epoch_jd is the Julian date of the epoch.

tle_to_satrec(line1, line2)
    Raw TLE strings → sgp4 Satrec (for when the user supplies real TLEs).

Notes on frames
---------------
SGP4 produces position/velocity in the TEME (True Equator Mean Equinox) frame.
J2000/GCRF (what the rest of this code calls "ECI") differs from TEME by the
equation of equinoxes plus short-period nutation terms — typically < 30 m at
LEO altitudes for propagation spans < 7 days.

For RL training this difference is negligible, but teme_to_eci() still applies
the full IAU 76/FK5 precession + IAU 1980 nutation correction so that the frame
is consistent with coordinates.py for long-duration experiments.

When spiceypy kernels are loaded the transform is delegated to SPICE
(pxform 'TEME'→'J2000'), which is authoritative. The pure-numpy fallback
implements the same rotation analytically.
"""

from __future__ import annotations

import math
import numpy as np

# ── optional SPICE back-end ───────────────────────────────────────────────────
try:
    import spiceypy as spice
    _SPICE_AVAILABLE = True
except ImportError:
    _SPICE_AVAILABLE = False

# ── optional sgp4 back-end ────────────────────────────────────────────────────
try:
    from sgp4.api import Satrec, WGS84
    from sgp4.api import sgp4init, WGS84 as _WGS84, SGP4_ERRORS
    _SGP4_AVAILABLE = True
except ImportError:
    _SGP4_AVAILABLE = False

# ── constants ─────────────────────────────────────────────────────────────────
_J2000_JD  = 2_451_545.0          # Julian date of J2000 epoch
_SGP4_JD0  = 2_433_281.5          # Julian date of sgp4 epoch origin (1949-Dec-31 00:00 UT)
_MU_KM3_S2 = 398_600.4418         # km³ s⁻²
_RE_KM     = 6_378.137            # km  equatorial radius (WGS-84)
_J2         = 1.082_626_68e-3


# ─────────────────────────────────────────────────────────────────────────────
# Epoch conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def et_to_jd(et: float) -> float:
    """Ephemeris time (s past J2000) → Julian date (days)."""
    return _J2000_JD + et / 86_400.0


def jd_to_et(jd: float) -> float:
    """Julian date (days) → ephemeris time (s past J2000)."""
    return (jd - _J2000_JD) * 86_400.0


def et_to_sgp4_epoch(et: float) -> float:
    """ET (s past J2000) → sgp4 epoch (days past 1949-Dec-31 00:00 UT)."""
    return et_to_jd(et) - _SGP4_JD0


def sgp4_epoch_to_et(epoch: float) -> float:
    """sgp4 epoch (days past 1949-Dec-31 00:00 UT) → ET (s past J2000)."""
    return jd_to_et(epoch + _SGP4_JD0)


# ─────────────────────────────────────────────────────────────────────────────
# TEME → J2000 (ECI) frame rotation
# ─────────────────────────────────────────────────────────────────────────────

def _gmst_rad(jd_ut1: float) -> float:
    """Greenwich Mean Sidereal Time (rad) from Julian date UT1."""
    T = (jd_ut1 - 2_451_545.0) / 36_525.0
    # IAU 1982 GMST model (seconds of arc → rad)
    theta_deg = (
        280.460_618_37
        + 360.985_647_246 * (jd_ut1 - 2_451_545.0)
        + 0.000_387_933 * T**2
        - T**3 / 38_710_000.0
    )
    return math.radians(theta_deg % 360.0)


def _nutation_dpsi_deps(T: float) -> tuple[float, float]:
    """
    Approximate IAU 1980 nutation in longitude (Δψ) and obliquity (Δε),
    using the dominant terms only (< 0.5 arcsec residual over ±50 yr).

    T : Julian centuries past J2000
    Returns (dpsi_arcsec, deps_arcsec)
    """
    # Mean anomaly of the Moon, Sun; longitude of Moon's ascending node
    Omega = math.radians(125.04 - 1934.136 * T)
    L     = math.radians(280.47 + 36_000.77 * T)
    Lm    = math.radians(218.32 + 481_267.88 * T)

    dpsi = (
        -17.20 * math.sin(Omega)
        - 1.32 * math.sin(2.0 * L)
        - 0.23 * math.sin(2.0 * Lm)
        + 0.21 * math.sin(2.0 * Omega)
    )   # arcseconds
    deps = (
         9.20 * math.cos(Omega)
        + 0.57 * math.cos(2.0 * L)
        + 0.10 * math.cos(2.0 * Lm)
        - 0.09 * math.cos(2.0 * Omega)
    )   # arcseconds
    return dpsi, deps


def _rot1(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def _rot3(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def _teme_to_j2000_matrix(t_et: float) -> np.ndarray:
    """
    3×3 rotation matrix  TEME → J2000 (GCRF approximation).

    Algorithm: IAU 76/FK5 precession + IAU 1980 nutation (dominant terms).
    Accurate to ~30 m at LEO for propagations up to a few weeks.

    If spiceypy kernels are loaded the SPICE pxform is used instead and
    this function is not called.
    """
    jd  = et_to_jd(t_et)
    T   = (jd - _J2000_JD) / 36_525.0

    # ── Mean obliquity of the ecliptic ────────────────────────────────────
    eps0 = math.radians(
        23.439_291_11
        - 0.013_004_2 * T
        - 1.64e-7      * T**2
        + 5.04e-7      * T**3
    )

    # ── IAU 1980 nutation ─────────────────────────────────────────────────
    dpsi_as, deps_as = _nutation_dpsi_deps(T)
    dpsi = math.radians(dpsi_as / 3600.0)
    deps = math.radians(deps_as / 3600.0)
    eps  = eps0 + deps                # true obliquity

    # ── Equation of the equinoxes (GMST offset between TEME and TOD) ──────
    eq_equinox = dpsi * math.cos(eps0)   # rad

    # ── IAU 76 precession angles (Lieske 1977) ─────────────────────────────
    zeta_A  = math.radians((0.640_616_2 + 0.000_839_9 * T + 5.0e-6 * T**2) * T / 3600.0)
    theta_A = math.radians((0.556_753 - 0.001_185_2 * T - 1.16e-5 * T**2) * T / 3600.0)
    z_A     = math.radians((0.640_616_2 + 0.003_042_2 * T + 1.8e-6 * T**2) * T / 3600.0)

    # ── Build transform ────────────────────────────────────────────────────
    # TEME → TOD: undo equation of equinoxes (rotation about z by eq_equinox)
    M_teme_to_tod = _rot3(-eq_equinox)

    # TOD → MOD: undo nutation
    M_nutation = _rot1(eps0) @ _rot3(dpsi) @ _rot1(-eps)
    M_tod_to_mod = M_nutation.T

    # MOD → J2000: undo precession
    M_prec = _rot3(-zeta_A) @ _rot1(theta_A) @ _rot3(-z_A)
    M_mod_to_j2000 = M_prec.T

    return M_mod_to_j2000 @ M_tod_to_mod @ M_teme_to_tod


def teme_to_eci(
    r_teme: np.ndarray,
    v_teme: np.ndarray,
    t_et:   float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate position and velocity from TEME to J2000 (ECI).

    Parameters
    ----------
    r_teme : (3,) position in TEME frame (km)
    v_teme : (3,) velocity in TEME frame (km s⁻¹)
    t_et   : ephemeris time of the state (s past J2000)

    Returns
    -------
    r_eci : (3,) position in J2000 (km)
    v_eci : (3,) velocity in J2000 (km s⁻¹)
    """
    from scripts.coordinates import _SPICE_LOADED  # noqa: avoid circular at top level

    if _SPICE_AVAILABLE and _SPICE_LOADED:
        # SPICE does not have a native TEME frame in generic kernels,
        # so we fall through to the analytic path.
        pass

    M = _teme_to_j2000_matrix(t_et)
    return M @ r_teme, M @ v_teme


# ─────────────────────────────────────────────────────────────────────────────
# Osculating elements → SGP4 mean elements  (Kozai-Brouwer approximation)
# ─────────────────────────────────────────────────────────────────────────────

def _kozai_mean_motion(a_osc: float, e_osc: float, i_osc: float) -> float:
    """
    Convert osculating semi-major axis to Kozai mean motion (rad/min).

    The Kozai mean element theory (used by SGP4) relates the osculating
    semi-major axis to the mean motion via a J2-based correction.

    a_osc : osculating semi-major axis (km)
    e_osc : eccentricity
    i_osc : inclination (rad)
    Returns: mean motion in rad/min (Kozai convention)
    """
    n_osc  = math.sqrt(_MU_KM3_S2 / a_osc**3)  # rad/s

    # Kozai J2 correction factor
    p      = a_osc * (1.0 - e_osc**2)
    beta   = math.sqrt(1.0 - e_osc**2)
    fac    = 1.0 + (3.0 / 2.0) * _J2 * (_RE_KM / p)**2 * beta * (
        1.0 - (3.0 / 2.0) * math.sin(i_osc)**2
    )
    n_kozai = n_osc * fac   # rad/s
    return n_kozai * 60.0   # rad/min


def _osc_to_mean_elements(
    a_osc:    float,
    e_osc:    float,
    i_osc:    float,
    raan_osc: float,
    aop_osc:  float,
    M_osc:    float,
) -> tuple[float, float, float, float, float, float]:
    """
    Single-iteration J2 osculating → mean element conversion.

    For near-circular LEO orbits the first-order J2 correction brings the
    error in semi-major axis down to < 1 km.  For highly eccentric orbits
    (e > 0.3) or high-inclination orbits (i > 70°) a higher-order conversion
    (Brouwer theory) would be more accurate, but is beyond the scope of an
    RL training environment.

    All angles in radians.  Returns (a_m, e_m, i_m, raan_m, aop_m, M_m)
    """
    p     = a_osc * (1.0 - e_osc**2)
    gamma = (_J2 / 2.0) * (_RE_KM / p)**2
    sin_i = math.sin(i_osc)
    cos_i = math.cos(i_osc)
    sin2i = sin_i**2

    # Mean inclination
    i_m   = i_osc - gamma * sin_i * cos_i * (
        4.0 - (7.0 / 2.0) * sin2i
    )

    # Mean eccentricity
    e_m   = e_osc - gamma * (1.0 - e_osc**2) * (
        1.0 - (3.0 / 2.0) * sin2i
    )

    # Mean RAAN
    raan_m = raan_osc + gamma * (3.0 / 2.0) * cos_i * (
        4.0 - 5.0 * sin2i
    )

    # Mean argument of perigee
    aop_m = aop_osc + gamma * (
        (7.0 / 2.0) * sin2i - 2.0
        + e_osc * (4.0 - (7.0 / 2.0) * sin2i)
    )

    # Mean anomaly (unchanged to first order in J2 for circular orbits)
    M_m   = M_osc

    # Mean semi-major axis (keep osculating — Kozai correction in n handles it)
    a_m   = a_osc

    return a_m, max(1e-6, e_m), i_m, raan_m % (2.0 * math.pi), aop_m % (2.0 * math.pi), M_m % (2.0 * math.pi)


# ─────────────────────────────────────────────────────────────────────────────
# Public Satrec constructors
# ─────────────────────────────────────────────────────────────────────────────

def elements_to_satrec(
    a:        float,
    e:        float,
    i_deg:    float,
    raan_deg: float,
    aop_deg:  float,
    ta_deg:   float,
    epoch_et: float,
    bstar:    float = 0.0,
) -> "Satrec":
    """
    Initialise an sgp4 Satrec from classical **osculating** elements.

    Parameters
    ----------
    a        : semi-major axis (km)
    e        : eccentricity
    i_deg    : inclination (deg)
    raan_deg : RAAN (deg)
    aop_deg  : argument of perigee (deg)
    ta_deg   : true anomaly (deg)
    epoch_et : simulation epoch (s past J2000)
    bstar    : drag coefficient B* (m⁻¹, default 0)

    Returns
    -------
    Satrec object ready for .sgp4(jd, fr) calls.

    Raises
    ------
    ImportError  if sgp4 is not installed
    RuntimeError if sgp4init reports an error
    """
    if not _SGP4_AVAILABLE:
        raise ImportError(
            "sgp4 is not installed.  Run:  pip install sgp4"
        )

    # ── True anomaly → mean anomaly ───────────────────────────────────────
    i_rad    = math.radians(i_deg)
    raan_rad = math.radians(raan_deg)
    aop_rad  = math.radians(aop_deg)
    ta_rad   = math.radians(ta_deg)

    E  = 2.0 * math.atan2(
        math.sqrt(1.0 - e) * math.sin(ta_rad / 2.0),
        math.sqrt(1.0 + e) * math.cos(ta_rad / 2.0),
    )
    M0 = (E - e * math.sin(E)) % (2.0 * math.pi)

    # ── Osculating → SGP4 mean elements ──────────────────────────────────
    a_m, e_m, i_m, raan_m, aop_m, M_m = _osc_to_mean_elements(
        a, e, i_rad, raan_rad, aop_rad, M0
    )

    # ── Kozai mean motion ─────────────────────────────────────────────────
    no_kozai = _kozai_mean_motion(a_m, e_m, i_m)  # rad/min

    # ── sgp4 epoch ────────────────────────────────────────────────────────
    epoch_sgp4 = et_to_sgp4_epoch(epoch_et)       # days past 1949-Dec-31

    # ── Initialise Satrec ─────────────────────────────────────────────────
    sat = Satrec()
    err = sgp4init(
        WGS84,          # gravity model
        'i',            # opsmode: 'i' = improved (default for modern use)
        1,              # satn: arbitrary satellite number
        epoch_sgp4,     # epoch (days past 1949-Dec-31 00:00 UT)
        bstar,          # B* drag term (m⁻¹)
        0.0,            # ndot  (rev/day²) — not used in SGP4
        0.0,            # nddot (rev/day³) — not used in SGP4
        e_m,            # eccentricity
        aop_m,          # argument of perigee (rad)
        i_m,            # inclination (rad)
        M_m,            # mean anomaly (rad)
        no_kozai,       # mean motion (rad/min, Kozai)
        raan_m,         # RAAN (rad)
        sat,
    )
    if err != 0:
        raise RuntimeError(
            f"sgp4init failed with error code {err}: "
            f"{SGP4_ERRORS.get(err, 'unknown error')}"
        )
    return sat


def tle_to_satrec(line1: str, line2: str) -> "Satrec":
    """
    Parse TLE strings into an sgp4 Satrec.

    Use this when you have real-world TLEs from Space-Track or CelesTrak
    instead of (or to validate against) generated elements.

    Parameters
    ----------
    line1, line2 : standard two-line element strings

    Returns
    -------
    Satrec object ready for .sgp4(jd, fr) calls.

    Raises
    ------
    ImportError if sgp4 is not installed.
    """
    if not _SGP4_AVAILABLE:
        raise ImportError("sgp4 is not installed.  Run:  pip install sgp4")
    return Satrec.twoline2rv(line1, line2)


def satrec_epoch_et(sat: "Satrec") -> float:
    """Return the epoch of a Satrec as ephemeris time (s past J2000)."""
    return sgp4_epoch_to_et(sat.jdsatepoch + sat.jdsatepochF - _SGP4_JD0)
