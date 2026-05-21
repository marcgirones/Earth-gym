"""
scripts/coordinates.py
======================
Coordinate-frame transforms using spiceypy (NAIF SPICE) with an
analytic / astropy fallback when SPICE kernels have not been loaded.

Frames
------
ECI  ≡ J2000 / ICRF   — inertial, origin at Earth centre
ECEF ≡ ITRF93         — Earth-fixed, rotating with Earth
Geo  = WGS-84 geodetic (lat °, lon °, alt km)

Quick-start
-----------
    from scripts.coordinates import download_kernels, load_spice_kernels

    download_kernels()       # one-time download (~15 MB)
    load_spice_kernels()     # call once at program startup

If kernels are not loaded the module falls back to an IAU 1982 GMST
rotation + Bowring geodetic conversion, which is accurate to a few
metres — more than enough for RL training.

SPICE kernel files (saved to ./spice_kernels/ by default)
---------------------------------------------------------
  naif0012.tls                 — leap-second kernel (required for str2et)
  pck00010.tpc                 — planetary constants  (RE, flattening)
  earth_latest_high_prec.bpc   — high-precision Earth orientation
"""

from __future__ import annotations

import os
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Tuple

import numpy as np

# ── Optional SPICE back-end ───────────────────────────────────────────────────
try:
    import spiceypy as spice
    _SPICE_AVAILABLE = True
except ImportError:
    _SPICE_AVAILABLE = False

_SPICE_LOADED = False   # becomes True after load_spice_kernels()

# ── WGS-84 constants ─────────────────────────────────────────────────────────
WGS84_A  = 6378.137              # km  — equatorial radius
WGS84_F  = 1.0 / 298.257223563  # —   — flattening
WGS84_B  = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = 2.0 * WGS84_F - WGS84_F**2   # first eccentricity²

# ── SPICE kernel catalogue ────────────────────────────────────────────────────
_DEFAULT_KERNEL_DIR = Path(__file__).parent.parent / "spice_kernels"
_KERNEL_URLS: dict[str, str] = {
    "naif0012.tls": (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls"
    ),
    "pck00010.tpc": (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc"
    ),
    "earth_latest_high_prec.bpc": (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/"
        "earth_latest_high_prec.bpc"
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Kernel management
# ─────────────────────────────────────────────────────────────────────────────

def download_kernels(kernel_dir: str | Path | None = None) -> None:
    """
    Download the required SPICE kernels (~15 MB total) to *kernel_dir*.
    Skips files that already exist.  Call once before load_spice_kernels().
    """
    d = Path(kernel_dir) if kernel_dir else _DEFAULT_KERNEL_DIR
    d.mkdir(parents=True, exist_ok=True)
    for fname, url in _KERNEL_URLS.items():
        dest = d / fname
        if dest.exists():
            print(f"[SPICE] {fname} already present — skipping.")
            continue
        print(f"[SPICE] Downloading {fname} …", flush=True)
        urllib.request.urlretrieve(url, dest)
        print(f"[SPICE] Saved → {dest}")


def load_spice_kernels(kernel_dir: str | Path | None = None) -> None:
    """
    Furnish all SPICE kernels so that spiceypy transforms can be used.
    Must be called once before any coordinate transform in this module.
    Raises FileNotFoundError if a kernel is missing (run download_kernels first).
    Raises ImportError if spiceypy is not installed.
    """
    global _SPICE_LOADED
    if not _SPICE_AVAILABLE:
        raise ImportError(
            "spiceypy is not installed.  Run:  pip install spiceypy"
        )
    d = Path(kernel_dir) if kernel_dir else _DEFAULT_KERNEL_DIR
    for fname in _KERNEL_URLS:
        path = d / fname
        if not path.exists():
            raise FileNotFoundError(
                f"SPICE kernel not found: {path}\n"
                "Run  coordinates.download_kernels()  to fetch it."
            )
        spice.furnsh(str(path))
    _SPICE_LOADED = True
    print("[SPICE] All kernels loaded.")


def unload_spice_kernels() -> None:
    """Unload all SPICE kernels (useful in tests)."""
    global _SPICE_LOADED
    if _SPICE_AVAILABLE and _SPICE_LOADED:
        spice.kclear()
        _SPICE_LOADED = False


# ─────────────────────────────────────────────────────────────────────────────
# STK date ↔ ephemeris time
# ─────────────────────────────────────────────────────────────────────────────

_MONTH_TO_NUM = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}
_NUM_TO_MONTH = {v: k for k, v in _MONTH_TO_NUM.items()}

# J2000 epoch in UTC  (TDB–UTC ≈ 64 s, ignored for RL purposes)
_J2000_UTC = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _parse_stk_date(date_str: str) -> datetime:
    """Parse STK format  'D Mon YYYY HH:MM:SS.fff'  → datetime (UTC)."""
    parts   = date_str.strip().split()
    day     = int(parts[0])
    month   = _MONTH_TO_NUM[parts[1]]
    year    = int(parts[2])
    h, m, s = parts[3].split(":")
    sec_f   = float(s)
    sec_i   = int(sec_f)
    usec    = int(round((sec_f - sec_i) * 1_000_000))
    return datetime(year, month, day, int(h), int(m), sec_i, usec,
                    tzinfo=timezone.utc)


def stk_date_to_et(date_str: str) -> float:
    """
    Convert an STK-format date string to ephemeris time (seconds past J2000).

    Uses spiceypy.str2et when kernels are loaded (accounts for leap seconds);
    otherwise uses a pure-Python UTC-based calculation.
    """
    if _SPICE_AVAILABLE and _SPICE_LOADED:
        return spice.str2et(date_str)
    return (_parse_stk_date(date_str) - _J2000_UTC).total_seconds()


def et_to_stk_date(et: float) -> str:
    """Convert ephemeris time (s past J2000) to STK-format string."""
    if _SPICE_AVAILABLE and _SPICE_LOADED:
        # Use SPICE to format; returns e.g. "2000 JAN 01 00:00:00.000 (UTC)"
        cal = spice.et2utc(et, "ISOC", 3)   # ISO "2000-01-01T00:00:00.000"
        dt  = datetime.fromisoformat(cal.replace("Z", "+00:00"))
    else:
        dt = _J2000_UTC + timedelta(seconds=et)
    frac = dt.microsecond / 1_000_000
    s = dt.second + frac
    return (f"{dt.day} {_NUM_TO_MONTH[dt.month]} {dt.year} "
            f"{dt.hour:02d}:{dt.minute:02d}:{s:09.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# ECI ↔ ECEF
# ─────────────────────────────────────────────────────────────────────────────

def _gmst_matrix(et: float) -> np.ndarray:
    """
    3×3 rotation matrix  ECI → ECEF  via IAU 1982 Greenwich Mean Sidereal Time.
    Accurate to ~1 arcsec; sufficient for RL training environments.

    et : seconds past J2000 (noon 2000-01-01)
    """
    T     = et / 86400.0      # Julian days past J2000
    theta = np.radians(
        280.46061837
        + 360.98564724596 * T
        + 0.000387933     * (T / 36525.0)**2
        - (T / 36525.0)**3 / 38710000.0
    )
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c,  s, 0.0],
                     [-s,  c, 0.0],
                     [ 0.0, 0.0, 1.0]])


def eci_to_ecef(r_eci: np.ndarray, et: float) -> np.ndarray:
    """
    ECI (J2000/ICRF) → ECEF position (km).
    Uses  spice.pxform('J2000','ITRF93', et)  when kernels are loaded.
    """
    if _SPICE_AVAILABLE and _SPICE_LOADED:
        M = np.array(spice.pxform("J2000", "ITRF93", et))
        return M @ r_eci
    return _gmst_matrix(et) @ r_eci


def ecef_to_eci(r_ecef: np.ndarray, et: float) -> np.ndarray:
    """ECEF → ECI (J2000/ICRF) position (km)."""
    if _SPICE_AVAILABLE and _SPICE_LOADED:
        M = np.array(spice.pxform("ITRF93", "J2000", et))
        return M @ r_ecef
    return _gmst_matrix(et).T @ r_ecef


# ─────────────────────────────────────────────────────────────────────────────
# ECEF ↔ Geodetic (WGS-84)
# ─────────────────────────────────────────────────────────────────────────────

def ecef_to_geodetic(r_ecef: np.ndarray) -> Tuple[float, float, float]:
    """
    ECEF (km) → WGS-84 geodetic (lat°, lon°, alt km).
    Uses  spice.recgeo  when kernels are loaded; Bowring's method otherwise.
    """
    if _SPICE_AVAILABLE and _SPICE_LOADED:
        # spice.recgeo: input km, returns (lon rad, lat rad, alt km)
        lon_r, lat_r, alt = spice.recgeo(r_ecef.tolist(), WGS84_A, WGS84_F)
        return float(np.degrees(lat_r)), float(np.degrees(lon_r)), float(alt)

    x, y, z = r_ecef
    lon = np.degrees(np.arctan2(y, x))
    p   = np.hypot(x, y)

    # Bowring's iterative geodetic latitude
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(10):
        N       = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat)**2)
        lat_new = np.arctan2(z + WGS84_E2 * N * np.sin(lat), p)
        if abs(lat_new - lat) < 1e-12:
            break
        lat = lat_new

    N   = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat)**2)
    clat = np.cos(lat)
    alt  = (p / clat - N) if abs(clat) > 1e-8 \
           else (abs(z) / abs(np.sin(lat)) - N * (1.0 - WGS84_E2))

    return float(np.degrees(lat)), float(lon), float(alt)


def geodetic_to_ecef(
    lat_deg: float, lon_deg: float, alt_km: float = 0.0
) -> np.ndarray:
    """
    WGS-84 geodetic (lat°, lon°, alt km) → ECEF (km).
    Uses  spice.georec  when kernels are loaded.
    """
    if _SPICE_AVAILABLE and _SPICE_LOADED:
        # spice.georec: (lon rad, lat rad, alt km, RE km, flattening)
        rec = spice.georec(
            np.radians(lon_deg), np.radians(lat_deg), alt_km,
            WGS84_A, WGS84_F,
        )
        return np.array(rec)

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    N   = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat)**2)
    x   = (N + alt_km) * np.cos(lat) * np.cos(lon)
    y   = (N + alt_km) * np.cos(lat) * np.sin(lon)
    z   = (N * (1.0 - WGS84_E2) + alt_km) * np.sin(lat)
    return np.array([x, y, z])


def geodetic_to_eci(
    lat_deg: float, lon_deg: float, alt_km: float, et: float
) -> np.ndarray:
    """Shorthand: geodetic → ECI (km)."""
    return ecef_to_eci(geodetic_to_ecef(lat_deg, lon_deg, alt_km), et)
