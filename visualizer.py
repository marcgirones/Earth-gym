"""
visualizer.py
=============
Option 1 — matplotlib static image generator.

Reads telemetry.jsonl and writes PNG images to the output folder.
Called automatically from ppo_earth_gym.py at every checkpoint.

Images generated
----------------
ground_track_<step>.png      Satellite ground track on a world map, coloured
                             by reward (blue=low → red=high).  Current sensor
                             footprint circle drawn at the last position.
                             Target zones shown as yellow stars.

reward_curve_<step>.png      Training reward + smoothed eval reward over all
                             steps logged so far.  Includes a learning-rate
                             secondary axis.

coverage_heatmap_<step>.png  2D histogram (lat × lon) of all positions visited.
                             Shows which regions of Earth the satellite has
                             observed.

all_<step>.png               Four-panel composite of all three plots + a
                             zoomed reward-vs-step scatter.  Best for quick
                             visual inspection.

Dependencies
------------
matplotlib (required)
cartopy    (optional — improves map rendering; falls back to imshow if absent)
scipy      (optional — used for Gaussian smoothing on the heatmap)
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

from telemetry import TelemetryLogger

# ── optional cartopy ──────────────────────────────────────────────────────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _CARTOPY = True
except ImportError:
    _CARTOPY = False

# ── optional scipy ────────────────────────────────────────────────────────────
try:
    from scipy.ndimage import gaussian_filter
    _SCIPY = True
except ImportError:
    _SCIPY = False

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_REWARD_CMAP = plt.cm.RdYlGn   # green=high, red=low


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Visualizer] saved → {path}")


def _reward_colors(rewards: list[float]) -> np.ndarray:
    """Map reward values to RGBA via the RdYlGn colormap."""
    arr  = np.array(rewards, dtype=float)
    vmin, vmax = arr.min(), arr.max()
    if vmin == vmax:
        return _REWARD_CMAP(np.full_like(arr, 0.5))
    return _REWARD_CMAP((arr - vmin) / (vmax - vmin))


def _smooth(values: list[float], w: int = 20) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if len(arr) < 3:
        return arr
    # Cap w so the kernel never exceeds the data length.
    # np.convolve(mode="same") returns max(len(arr), len(kernel)) elements;
    # when w > len(arr) that produces more values than data points, causing
    # shape mismatches in every downstream plot call.
    w = min(w, len(arr))
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Ground track
# ─────────────────────────────────────────────────────────────────────────────

def _footprint_radius_deg(alt_km: float, cone_deg: float = 30.0) -> float:
    """
    Angular radius of the sensor footprint on the ground (degrees of arc).

    Uses exact spherical geometry:
      rho   = arccos(RT / (RT + alt))          [Earth limb half-angle]
      nadir = arcsin(sin(cone) * (RT+alt)/RT)  [nadir angle at footprint edge]
      earth_angle = 90 - cone - nadir          [Earth central angle]
    footprint radius = RT * earth_angle (in radians).

    Falls back to 0 if cone_deg >= rho (sensor points beyond the limb).
    """
    RT       = 6371.0
    r_sat    = RT + max(alt_km, 0.0)
    cone_rad = np.radians(cone_deg)
    rho      = np.arccos(RT / r_sat)        # max Earth half-angle

    if cone_rad >= rho:
        return float(np.degrees(rho))       # clamp to full visible disk

    # Spherical law of sines: sin(nadir)/RT = sin(90+cone)/(RT+alt)
    sin_nadir  = np.sin(np.pi / 2.0 + cone_rad) * r_sat / RT
    sin_nadir  = np.clip(sin_nadir, -1.0, 1.0)
    nadir_rad  = np.arcsin(sin_nadir)
    earth_angle = np.pi / 2.0 - cone_rad - (np.pi - nadir_rad)
    earth_angle = abs(earth_angle)
    return float(np.degrees(earth_angle))


def _interpolate_ground_track(
    records: list[dict],
    n_sub: int = 20,
) -> tuple[list[float], list[float], list[float]]:
    """
    Produce a dense ground track by propagating the orbit between telemetry steps.

    Each telemetry record is logged once per RL step (delta_time apart).
    With delta_time = T/8, consecutive records are 1/8 of an orbit apart —
    good for training but still only 8 dots per orbit in the plot.
    With n_sub=20 we sample 20 sub-points per step → 160 dots per orbit,
    giving a smooth sinusoidal band on the map.

    Returns
    -------
    lats, lons, rewards  — one value per sub-sample, repeated reward per step
    """
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from scripts.propagator_base import make_propagator
        from scripts.coordinates import (
            stk_date_to_et, eci_to_ecef, ecef_to_geodetic
        )
        import json
        cfg = json.load(open("src/agents-configuration.json"))
        general = cfg.get("general", {})
        ag_raw  = cfg["agents"][0]
        ag_flat = {}
        for v in ag_raw.values():
            if isinstance(v, dict):
                ag_flat.update(v)
        # Flatten initial_orbital_elements sub-dict
        if "initial_orbital_elements" in ag_flat:
            ag_flat.update(ag_flat.pop("initial_orbital_elements"))

        prop_type = general.get("propagator", "TwoBody")
        et0       = stk_date_to_et(general["start_time"])
        prop      = make_propagator(ag_flat, prop_type, et0)

        # Use the recorded timestamps to drive sub-step propagation.
        # Each record has "ts" (wall-clock) but NOT simulation time.
        # We reconstruct simulation time from step index and delta_time.
        # Read delta_time from the config if available, fall back to 890.1s.
        # Derive delta_time from scenario window and max step in records.
        # This automatically picks up any change to the config.
        from scripts.coordinates import stk_date_to_et as _s2et
        _et0      = _s2et(general["start_time"])
        _stop_et  = _s2et(general["stop_time"])
        _ep_dur   = _stop_et - _et0
        _max_step = max((r.get("step", 0) for r in records), default=99)
        delta_time = _ep_dur / (_max_step + 1)

        lats, lons, rews = [], [], []
        for rec in records:
            step   = rec.get("step", 0)
            reward = rec.get("reward", 0.0)
            t_step = et0 + step * delta_time
            for k in range(n_sub):
                t = t_step + k * delta_time / n_sub
                r_eci, _, _ = prop.propagate(t)
                r_ecef      = eci_to_ecef(r_eci, t)
                lat, lon, _ = ecef_to_geodetic(r_ecef)
                lats.append(lat)
                lons.append(lon)
                rews.append(reward)
        return lats, lons, rews

    except Exception:
        # Fallback: just return the recorded positions (1 dot per step)
        lats = [r.get("sat_lat", r.get("lat", 0.0)) for r in records]
        lons = [r.get("sat_lon", r.get("lon", 0.0)) for r in records]
        rews = [r.get("reward", 0.0) for r in records]
        return lats, lons, rews


def plot_ground_track(
    records: list[dict],
    out_path: str,
    cone_angle_deg: float = 10.0,
    max_points: int = 200,
) -> None:
    """
    Ground track showing satellite orbit line + boresight intercept dots + targets.
    """
    if not records:
        return

    records = records[-max_points:]

    # Satellite orbit as thin reference line
    sat_lats, sat_lons, _ = _interpolate_ground_track(records, n_sub=20)

    # Boresight intercepts — where sensor actually pointed (one per step)
    has_bs   = any("boresight_lat" in r for r in records)
    bs_lats  = [r.get("boresight_lat", r.get("sat_lat", r.get("lat", 0.0))) for r in records]
    bs_lons  = [r.get("boresight_lon", r.get("sat_lon", r.get("lon", 0.0))) for r in records]
    bs_rews  = [r["reward"] for r in records]
    bs_colors = _reward_colors(bs_rews)

    alts = [r.get("sat_alt", r.get("alt", 1629.0)) for r in records]

    # Accumulate ALL targets seen across ALL records (unique by lat/lon)
    seen = {}
    for r in records:
        for lat, lon, pri in r.get("targets", []):
            key = (round(lat, 2), round(lon, 2))
            if key not in seen or pri > seen[key]:
                seen[key] = pri
    tgt_lats = [k[0] for k in seen]
    tgt_lons = [k[1] for k in seen]
    tgt_pri  = [seen[k] for k in seen]

    if _CARTOPY:
        fig = plt.figure(figsize=(14, 7))
        ax  = fig.add_subplot(1, 1, 1,
                              projection=ccrs.Robinson(central_longitude=0))
        ax.set_global()
        ax.add_feature(cfeature.LAND,      facecolor="#e8e4dc", zorder=0)
        ax.add_feature(cfeature.OCEAN,     facecolor="#c9dce8", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=1)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, zorder=1)
        ax.gridlines(linewidth=0.3, color="gray", alpha=0.4, linestyle="--")

        # 1. Satellite orbit line
        ax.plot(sat_lons, sat_lats, color="steelblue", linewidth=0.5,
                alpha=0.35, transform=ccrs.PlateCarree(), zorder=2)

        # 2. Boresight scatter
        ax.scatter(bs_lons, bs_lats, c=bs_colors, s=6,
                   transform=ccrs.PlateCarree(), zorder=3, rasterized=True)

        # 3. Targets
        if tgt_lats:
            ax.scatter(tgt_lons, tgt_lats, c=tgt_pri, cmap="YlOrRd",
                       s=60, marker="*", edgecolors="k", linewidths=0.3,
                       transform=ccrs.PlateCarree(), zorder=4,
                       vmin=0, vmax=1, label="Targets (FoR history)")

        # 4. Footprint at boresight
        if bs_lats:
            last   = records[-1]
            fp_lat = last.get("boresight_lat", bs_lats[-1])
            fp_lon = last.get("boresight_lon", bs_lons[-1])
            fp_deg = _footprint_radius_deg(alts[-1], cone_angle_deg)
            ax.tissot(rad_lon=fp_deg, rad_lat=fp_deg,
                      lons=[fp_lon], lats=[fp_lat],
                      n_samples=64, facecolor="gold", alpha=0.25,
                      transform=ccrs.PlateCarree(), zorder=5)

    else:
        fig, ax = plt.subplots(figsize=(14, 7))
        world = np.ones((180, 360, 3)) * 0.85
        ax.imshow(world, extent=[-180, 180, -90, 90], aspect="auto",
                  zorder=0, alpha=0.5)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
        ax.grid(linewidth=0.3, alpha=0.4)

        # 1. Satellite orbit line
        ax.plot(sat_lons, sat_lats, color="steelblue", linewidth=0.5,
                alpha=0.35, zorder=2)

        # 2. Boresight scatter
        ax.scatter(bs_lons, bs_lats, c=bs_colors, s=6, zorder=3, rasterized=True)

        # 3. Targets
        if tgt_lats:
            sc = ax.scatter(tgt_lons, tgt_lats, c=tgt_pri, cmap="YlOrRd",
                            s=80, marker="*", edgecolors="k", linewidths=0.3,
                            zorder=4, vmin=0, vmax=1)
            fig.colorbar(sc, ax=ax, shrink=0.4, pad=0.01, label="Target priority")

        # 4. Footprint at boresight
        if bs_lats:
            last   = records[-1]
            fp_lat = last.get("boresight_lat", bs_lats[-1])
            fp_lon = last.get("boresight_lon", bs_lons[-1])
            fp_deg = _footprint_radius_deg(alts[-1], cone_angle_deg)
            circ   = Circle((fp_lon, fp_lat), fp_deg,
                            facecolor="gold", alpha=0.3, edgecolor="darkorange",
                            linewidth=1.2, zorder=5)
            ax.add_patch(circ)

    sm = plt.cm.ScalarMappable(cmap=_REWARD_CMAP,
                                norm=mcolors.Normalize(min(bs_rews), max(bs_rews)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, label="Reward")
    ax.set_title(
        "Ground track  ·  line = satellite orbit  ·  "
        "dots = boresight intercept (coloured by reward)  ·  ★ = FoR targets"
    )

    step = records[-1].get("step", "?")
    ax.set_title(f"Satellite ground track — batch {step}  "
                 f"({len(records)} positions)")
    if tgt_lats:
        ax.legend(loc="lower left", fontsize=8)

    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Reward curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_reward_curves(
    records: list[dict],
    logs:    dict,
    out_path: str,
) -> None:
    """
    Training reward, smoothed reward, and eval reward over training steps.
    """
    if not records:
        return

    steps   = [r["step"]   for r in records]
    rewards = [r["reward"] for r in records]
    smooth  = _smooth(rewards, w=max(1, len(rewards) // 20)).tolist()

    eval_steps   = [r["step"] for r in records if r.get("eval_reward") is not None]
    eval_rewards = [r["eval_reward"] for r in records
                    if r.get("eval_reward") is not None]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    fig.subplots_adjust(hspace=0.35)

    # Top: step-level reward
    ax = axes[0]
    ax.plot(steps, rewards, alpha=0.3, linewidth=0.8, color="steelblue",
            label="Step reward")
    ax.plot(steps, smooth, linewidth=1.8, color="navy", label="Smoothed")
    if eval_steps:
        ax.scatter(eval_steps, eval_rewards, s=40, color="darkorange",
                   zorder=5, label="Eval reward", marker="D")
    ax.set_ylabel("Reward")
    ax.set_xlabel("Batch step")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_title("Training reward over time")

    # Bottom: cumulative reward + eval sum
    ax2 = axes[1]
    cum = np.cumsum(rewards)
    ax2.plot(steps, cum, color="seagreen", linewidth=1.6, label="Cumulative reward")

    if logs.get("eval reward (sum)"):
        eval_sum_steps = [i * 10 for i in range(len(logs["eval reward (sum)"]))]
        ax2.plot(eval_sum_steps, np.cumsum(logs["eval reward (sum)"]),
                 color="darkorange", linewidth=1.4,
                 linestyle="--", label="Cumulative eval reward")

    ax2.set_ylabel("Cumulative reward")
    ax2.set_xlabel("Batch step")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(linewidth=0.3, alpha=0.5)
    ax2.set_title("Cumulative reward")

    step = records[-1].get("step", "?")
    fig.suptitle(f"Reward curves — batch {step}", fontsize=13, y=1.01)
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Coverage heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_coverage_heatmap(
    records: list[dict],
    out_path: str,
    bins:     int = 72,   # 5° × 5° grid
) -> None:
    """
    2D histogram of all sensor boresight ground intercept positions visited.

    The heatmap shows WHERE THE SENSOR POINTED, not where the satellite was.
    For i=0 the satellite is always at lat=0, but the agent pitches/rolls the
    sensor to aim at off-equatorial targets — those boresight positions are
    what matters for evaluating coverage quality.

    Falls back to satellite sub-track positions if boresight data is absent.
    """
    if not records:
        return

    # Prefer boresight intercept (where sensor pointed) over sub-satellite point
    has_boresight = any("boresight_lat" in r for r in records)
    if has_boresight:
        lats = np.array([r.get("boresight_lat", r.get("sat_lat", r.get("lat", 0.0)))
                         for r in records])
        lons = np.array([r.get("boresight_lon", r.get("sat_lon", r.get("lon", 0.0)))
                         for r in records])
        label = "Boresight intercept count"
    else:
        # Fallback: dense interpolated satellite track
        lats_l, lons_l, _ = _interpolate_ground_track(records, n_sub=20)
        lats = np.array(lats_l)
        lons = np.array(lons_l)
        label = "Visit count (satellite track)"

    h, xedges, yedges = np.histogram2d(
        lons, lats,
        bins=[bins, bins // 2],
        range=[[-180, 180], [-90, 90]],
    )
    h = h.T   # lat × lon

    if _SCIPY:
        h = gaussian_filter(h.astype(float), sigma=1.2)

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(
        h, origin="lower",
        extent=[-180, 180, -90, 90],
        aspect="auto",
        cmap="hot_r",
        interpolation="bilinear",
    )
    fig.colorbar(im, ax=ax, shrink=0.7, label=label)

    # Overlay all targets seen across records
    seen = {}
    for r in records:
        for tlat, tlon, tpri in r.get("targets", []):
            key = (round(tlat, 2), round(tlon, 2))
            if key not in seen or tpri > seen[key]:
                seen[key] = tpri
    if seen:
        t_lats = [k[0] for k in seen]
        t_lons = [k[1] for k in seen]
        t_pri  = [seen[k] for k in seen]
        ax.scatter(t_lons, t_lats, c=t_pri, cmap="cool",
                   s=25, marker="*", edgecolors="white", linewidths=0.2,
                   zorder=5, vmin=0, vmax=1, alpha=0.7, label="Targets")

    # Overlay target zones (last frame)
    tgt_lats, tgt_lons, tgt_pri = [], [], []
    for r in records[-1:]:
        for lat, lon, pri in r.get("targets", []):
            tgt_lats.append(lat); tgt_lons.append(lon); tgt_pri.append(pri)

    if tgt_lats:
        ax.scatter(tgt_lons, tgt_lats, c=tgt_pri, cmap="Blues_r",
                   s=80, marker="*", edgecolors="cyan", linewidths=0.5,
                   zorder=3, vmin=0, vmax=1, label="Targets")
        ax.legend(loc="lower left", fontsize=8)

    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.grid(linewidth=0.3, color="white", alpha=0.4)

    step = records[-1].get("step", "?")
    ax.set_title(
        f"Coverage heatmap — {len(records)} positions  (batch {step})"
    )
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Composite four-panel
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(
    records:        list[dict],
    logs:           dict,
    out_path:       str,
    cone_angle_deg: float = 10.0,
    max_track_pts:  int   = 1000,
) -> None:
    """
    Four-panel composite: ground track, reward, coverage heatmap, reward scatter.
    """
    if not records:
        return

    rec = records[-max_track_pts:]

    # Satellite orbit track (dense sub-step interpolation)
    sat_lats, sat_lons, rewards = _interpolate_ground_track(rec, n_sub=20)

    # Boresight intercepts (one per telemetry record — where the sensor looked)
    has_bs = any("boresight_lat" in r for r in rec)
    bs_lats = [r.get("boresight_lat", r.get("sat_lat", 0.0)) for r in rec]
    bs_lons = [r.get("boresight_lon", r.get("sat_lon", 0.0)) for r in rec]
    bs_rew  = [r["reward"] for r in rec]

    steps   = [r["step"]   for r in records]
    all_rew = [r["reward"] for r in records]

    fig = plt.figure(figsize=(18, 10))
    gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # ── Panel A: ground track + boresight scatter ─────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])

    # 1. Satellite sub-track as a thin grey line (orbit reference)
    ax_a.plot(sat_lons, sat_lats, color="steelblue", linewidth=0.6,
              alpha=0.4, zorder=2, label="Satellite track")

    # 2. Boresight intercepts as coloured dots (what the sensor actually saw)
    bs_colors = _reward_colors(bs_rew)
    ax_a.scatter(bs_lons, bs_lats, c=bs_colors, s=8,
                 rasterized=True, zorder=3,
                 label="Boresight intercept" if has_bs else "Sub-satellite")

    # 3. Footprint circle at the most recent boresight position
    last_rec = rec[-1]
    fp_lat = last_rec.get("boresight_lat", sat_lats[-1])
    fp_lon = last_rec.get("boresight_lon", sat_lons[-1])
    fp_alt = last_rec.get("sat_alt", last_rec.get("alt", 1629.0))
    fp_deg = _footprint_radius_deg(fp_alt, cone_angle_deg)
    from matplotlib.patches import Circle as _Circle
    ax_a.add_patch(_Circle((fp_lon, fp_lat), fp_deg,
                            facecolor="gold", alpha=0.25,
                            edgecolor="darkorange", linewidth=1.0, zorder=4))
    ax_a.plot(fp_lon, fp_lat, marker="+", color="darkorange",
              markersize=8, markeredgewidth=1.5, zorder=5,
              label="Latest boresight")

    # 4. Targets — accumulated across all recent records
    seen_tgts = {}
    for r in rec:
        for tlat, tlon, tpri in r.get("targets", []):
            key = (round(tlat, 2), round(tlon, 2))
            if key not in seen_tgts or tpri > seen_tgts[key]:
                seen_tgts[key] = tpri
    if seen_tgts:
        t_lats = [k[0] for k in seen_tgts]
        t_lons = [k[1] for k in seen_tgts]
        t_pri  = [seen_tgts[k] for k in seen_tgts]
        ax_a.scatter(t_lons, t_lats, c=t_pri, cmap="YlOrRd",
                     s=40, marker="*", edgecolors="k", linewidths=0.2,
                     zorder=4, vmin=0, vmax=1, label="FoR targets")

    ax_a.set_xlim(-180, 180); ax_a.set_ylim(-90, 90)
    ax_a.set_xlabel("Lon (°)"); ax_a.set_ylabel("Lat (°)")
    ax_a.set_title("Ground track + sensor boresight (recent)\n"
                   "line = satellite orbit · dots = where sensor pointed · ✛ circle = latest footprint")
    ax_a.legend(fontsize=7, loc="lower left")
    ax_a.grid(linewidth=0.2, alpha=0.4)
    sm = plt.cm.ScalarMappable(cmap=_REWARD_CMAP,
                                norm=mcolors.Normalize(min(bs_rew), max(bs_rew)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax_a, shrink=0.8, pad=0.02)

    # ── Panel B: reward time series ───────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    smooth = _smooth(all_rew).tolist()
    ax_b.plot(steps, all_rew, alpha=0.25, linewidth=0.6, color="steelblue")
    ax_b.plot(steps, smooth, linewidth=1.8, color="navy")
    if logs.get("eval reward"):
        ev_steps = [i * 10 for i in range(len(logs["eval reward"]))]
        ax_b.scatter(ev_steps, logs["eval reward"], s=20,
                     color="darkorange", zorder=5, marker="D")
    ax_b.set_xlabel("Batch"); ax_b.set_ylabel("Reward")
    ax_b.set_title("Reward over time")
    ax_b.grid(linewidth=0.2, alpha=0.4)

    # ── Panel C: coverage heatmap (boresight intercept positions) ────────────
    ax_c = fig.add_subplot(gs[1, 0])
    has_bs = any("boresight_lat" in r for r in records)
    if has_bs:
        hl = np.array([r.get("boresight_lat", r.get("sat_lat", 0.0)) for r in records])
        wl = np.array([r.get("boresight_lon", r.get("sat_lon", 0.0)) for r in records])
        htitle = "Coverage heatmap — sensor boresight (all time)"
    else:
        hl_l, wl_l, _ = _interpolate_ground_track(records, n_sub=20)
        hl, wl = np.array(hl_l), np.array(wl_l)
        htitle = "Coverage heatmap — satellite track (all time)"
    all_lats = hl
    all_lons = wl
    h, _, _ = np.histogram2d(all_lons, all_lats, bins=[72, 36],
                              range=[[-180, 180], [-90, 90]])
    h = h.T
    if _SCIPY:
        h = gaussian_filter(h.astype(float), sigma=1.0)
    im = ax_c.imshow(h, origin="lower", extent=[-180, 180, -90, 90],
                     aspect="auto", cmap="hot_r", interpolation="bilinear")
    fig.colorbar(im, ax=ax_c, shrink=0.8, pad=0.02)

    # Overlay accumulated targets on heatmap
    seen_c = {}
    for r in records:
        for tlat, tlon, tpri in r.get("targets", []):
            key = (round(tlat, 2), round(tlon, 2))
            if key not in seen_c or tpri > seen_c[key]:
                seen_c[key] = tpri
    if seen_c:
        c_lats = [k[0] for k in seen_c]
        c_lons = [k[1] for k in seen_c]
        c_pri  = [seen_c[k] for k in seen_c]
        ax_c.scatter(c_lons, c_lats, c=c_pri, cmap="cool",
                     s=20, marker="*", edgecolors="white", linewidths=0.2,
                     zorder=5, vmin=0, vmax=1, alpha=0.7)

    ax_c.set_xlabel("Lon (°)"); ax_c.set_ylabel("Lat (°)")
    ax_c.set_title(htitle)

    # ── Panel D: cumulative reward ────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.plot(steps, np.cumsum(all_rew), color="seagreen", linewidth=1.6)
    if logs.get("eval reward (sum)"):
        ev_steps = [i * 10 for i in range(len(logs["eval reward (sum)"]))]
        ax_d.plot(ev_steps, np.cumsum(logs["eval reward (sum)"]),
                  color="darkorange", linestyle="--", linewidth=1.4)
    ax_d.set_xlabel("Batch"); ax_d.set_ylabel("Cumulative reward")
    ax_d.set_title("Cumulative reward")
    ax_d.grid(linewidth=0.2, alpha=0.4)

    step = records[-1].get("step", "?")
    fig.suptitle(f"Earth-Gym training snapshot — batch {step}", fontsize=14)
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_all(
    out_dir:        str,
    step:           int,
    logs:           dict,
    cone_angle_deg: float = 10.0,
    max_records:    int   = 5000,
) -> None:
    """
    Generate all four image types for checkpoint *step*.
    Called from ppo_earth_gym.py inside the checkpoint block.

    Parameters
    ----------
    out_dir        : output folder (same as --out arg)
    step           : current batch index
    logs           : the training-loop logs dict (for eval rewards)
    cone_angle_deg : sensor half-angle (must match agents-configuration.json)
    max_records    : cap on how many telemetry records to load (avoids OOM)
    """
    telemetry_path = Path(out_dir) / "telemetry.jsonl"
    records        = TelemetryLogger.load(telemetry_path, last_n=max_records)

    if not records:
        print("[Visualizer] No telemetry records yet — skipping.")
        return

    img_dir = Path(out_dir) / "images"
    img_dir.mkdir(exist_ok=True)

    tag = f"{step:05d}"

    plot_ground_track(
        records, str(img_dir / f"ground_track_{tag}.png"),
        cone_angle_deg=cone_angle_deg,
    )
    plot_reward_curves(
        records, logs, str(img_dir / f"reward_curve_{tag}.png")
    )
    plot_coverage_heatmap(
        records, str(img_dir / f"coverage_heatmap_{tag}.png")
    )
    plot_all(
        records, logs, str(img_dir / f"all_{tag}.png"),
        cone_angle_deg=cone_angle_deg,
    )

    print(f"[Visualizer] 4 images written to {img_dir}/")
