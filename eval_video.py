"""
eval_video.py
=============
Render a video of evaluation episodes from eval_telemetry.jsonl.

Each frame shows
----------------
  Left   — Orthographic globe (pure-math projection, no external map tiles):
              • Ground track coloured by step reward (red→green)
              • Animated satellite marker
              • Sensor footprint circle projected on the surface
              • FoR target memory slots (coloured stars, sized by priority)
              • White star flash on access events
  Right  — Three stacked live charts:
              • Step reward (colour-coded bars)
              • Cumulative reward (filled line)
              • Pitch & roll attitude (dual line)
  Bottom — HUD text: sim-time · lat/lon/alt · pitch/roll · reward · cumulative

Usage
-----
    python eval_video.py --input output/eval_telemetry.jsonl
    python eval_video.py --input output/eval_telemetry.jsonl --episode 2 --fps 20
    python eval_video.py --input output/eval_telemetry.jsonl --format gif

Flags
-----
  --input    Path to eval_telemetry.jsonl   (default: output/eval_telemetry.jsonl)
  --out      Output directory               (default: same dir as --input)
  --episode  Episode number to render       (default: all)
  --fps      Frames per second             (default: 15)
  --dpi      Render DPI                    (default: 120)
  --trail    Max past positions on globe    (default: 180)
  --cone     Sensor cone half-angle (deg)  (default: 10)
  --format   mp4 or gif                    (default: mp4)
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.collections import LineCollection

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input",   default="output/eval_telemetry.jsonl")
parser.add_argument("--out",     default="videos/")
parser.add_argument("--episode", default=None, type=int)
parser.add_argument("--fps",     default=15,   type=int)
parser.add_argument("--dpi",     default=120,  type=int)
parser.add_argument("--trail",   default=180,  type=int)
parser.add_argument("--cone",    default=10.0, type=float)
parser.add_argument("--format",  default="mp4", choices=["mp4","gif"])
args = parser.parse_args()

IN_PATH = Path(args.input)
OUT_DIR = Path(args.out) if args.out else IN_PATH.parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

RT_KM = 6371.0

# ── Colour palette ────────────────────────────────────────────────────────────
BG      = "#0a0e1a"
BG_CARD = "#111827"
C_BLUE  = "#3b82f6"
C_GREEN = "#22c55e"
C_RED   = "#ef4444"
C_CYAN  = "#06b6d4"
C_YELL  = "#eab308"
C_ORNG  = "#f97316"
C_PURP  = "#a855f7"
C_GREY  = "#6b7280"
C_WHITE = "#f9fafb"


# ── Coastline data ────────────────────────────────────────────────────────────
COASTLINE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector"
    "/master/geojson/ne_110m_coastline.geojson"
)
COASTLINE_CACHE = Path("/tmp/_ne110m_coastline.json")


def _load_coastlines() -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (lons, lats) numpy arrays for each coastline segment."""
    if not COASTLINE_CACHE.exists():
        print("  Downloading coastline data … ", end="", flush=True)
        with urllib.request.urlopen(COASTLINE_URL, timeout=15) as r:
            COASTLINE_CACHE.write_bytes(r.read())
        print("done")
    gj = json.loads(COASTLINE_CACHE.read_text())
    segs: list[tuple[np.ndarray, np.ndarray]] = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        parts = ([geom["coordinates"]]
                 if geom["type"] == "LineString"
                 else geom["coordinates"])
        for part in parts:
            arr = np.asarray(part, dtype=float)
            segs.append((arr[:, 0], arr[:, 1]))
    return segs

COASTLINES = _load_coastlines()


# ── Orthographic projection helpers ──────────────────────────────────────────

def ortho(lat_deg, lon_deg, clat, clon):
    """
    Project (lat_deg, lon_deg) onto unit-sphere orthographic plane centred at
    (clat, clon).  Returns (x, y, visible) where visible is a boolean mask.
    Works on scalars or numpy arrays.
    """
    la  = np.radians(np.asarray(lat_deg, dtype=float))
    lo  = np.radians(np.asarray(lon_deg, dtype=float))
    la0 = math.radians(clat)
    lo0 = math.radians(clon)
    cos_c = (math.sin(la0) * np.sin(la)
             + math.cos(la0) * np.cos(la) * np.cos(lo - lo0))
    x = np.cos(la) * np.sin(lo - lo0)
    y = (math.cos(la0) * np.sin(la)
         - math.sin(la0) * np.cos(la) * np.cos(lo - lo0))
    return x, y, cos_c > 0


def footprint_latlon(b_lat, b_lon, alt_km, cone_deg, n=80):
    """
    Small-circle on the sphere centred on the boresight ground point.
    Returns (lats, lons) for n+1 points.
    """
    r_sat = RT_KM + max(alt_km, 0.0)
    rho   = math.acos(RT_KM / r_sat)                       # Earth limb angle
    cone  = math.radians(min(cone_deg, math.degrees(rho)))  # clamp to horizon

    # Elevation correction → angular radius on ground
    sin_nd  = math.sin(math.pi / 2 + cone) * r_sat / RT_KM
    nd      = math.asin(min(1.0, sin_nd))
    ang_rad = math.degrees(abs(math.pi / 2 - cone - (math.pi - nd)))

    blat_r, blon_r = math.radians(b_lat), math.radians(b_lon)
    ang_r = math.radians(ang_rad)
    lats, lons = [], []
    for i in range(n + 1):
        bearing = 2 * math.pi * i / n
        lat2 = math.asin(
            math.sin(blat_r) * math.cos(ang_r)
            + math.cos(blat_r) * math.sin(ang_r) * math.cos(bearing)
        )
        lon2 = blon_r + math.atan2(
            math.sin(bearing) * math.sin(ang_r) * math.cos(blat_r),
            math.cos(ang_r) - math.sin(blat_r) * math.sin(lat2),
        )
        lats.append(math.degrees(lat2))
        lons.append(math.degrees(lon2))
    return np.array(lats), np.array(lons)


def rew_color(r, rmin, rmax):
    t = max(0.0, min(1.0, (r - rmin) / (rmax - rmin + 1e-9)))
    if t >= 0.5:
        return (2*(1-t)*0.94, 0.76, 0.12)
    else:
        return (0.94, 2*t*0.76, 0.12)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_episodes(path: Path) -> dict[int, list[dict]]:
    eps: dict[int, list[dict]] = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                eps[rec.get("eval_episode", 1)].append(rec)
            except json.JSONDecodeError:
                pass
    for k in eps:
        eps[k].sort(key=lambda r: r.get("eval_step", 0))
    return eps


# ── Figure builder ────────────────────────────────────────────────────────────

def make_fig():
    fig = plt.figure(figsize=(19, 10), facecolor=BG)

    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0.01, right=0.99, top=0.92, bottom=0.07,
        wspace=0.06,
        width_ratios=[6, 4],
    )

    ax_g = fig.add_subplot(gs[0, 0], facecolor="#050d1a")

    rgs = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs[0, 1],
        hspace=0.52, height_ratios=[1, 1, 1]
    )
    ax_r  = fig.add_subplot(rgs[0])   # step reward
    ax_cr = fig.add_subplot(rgs[1])   # cumulative reward
    ax_at = fig.add_subplot(rgs[2])   # attitude

    for ax in (ax_r, ax_cr, ax_at):
        ax.set_facecolor(BG_CARD)
        for sp in ax.spines.values():
            sp.set_edgecolor(C_GREY)
            sp.set_linewidth(0.4)
        ax.tick_params(colors=C_WHITE, labelsize=6.5)
        ax.yaxis.label.set_color(C_GREY)

    ax_g.set_aspect("equal")
    ax_g.axis("off")

    return fig, ax_g, ax_r, ax_cr, ax_at


# ── Globe drawing ─────────────────────────────────────────────────────────────

def draw_globe_static(ax, clat, clon):
    """Redraw the static globe features (called every frame because projection centre moves)."""
    ax.cla()
    ax.set_facecolor("#050d1a")
    ax.set_aspect("equal")
    ax.axis("off")

    # Sphere outline
    th = np.linspace(0, 2*math.pi, 300)
    ax.fill(np.cos(th), np.sin(th), color="#0b1e3d", zorder=0)
    ax.plot(np.cos(th), np.sin(th), color="#1e3a5f", linewidth=0.8, zorder=1)

    # Ocean shading already done by fill above. Draw land patches via coastlines.
    # Lat/lon grid
    for lat in range(-60, 91, 30):
        lo_arr = np.linspace(-180, 180, 720)
        x, y, vis = ortho(lat, lo_arr, clat, clon)
        segs = []
        seg_x, seg_y = [], []
        for i in range(len(lo_arr)):
            if vis[i]:
                seg_x.append(x[i]); seg_y.append(y[i])
            else:
                if seg_x:
                    segs.append(list(zip(seg_x, seg_y)))
                seg_x, seg_y = [], []
        if seg_x:
            segs.append(list(zip(seg_x, seg_y)))
        lc = LineCollection(segs, colors="#1e3a5f", linewidths=0.3,
                            linestyles="--", zorder=2)
        ax.add_collection(lc)

    for lon in range(-180, 181, 30):
        la_arr = np.linspace(-90, 90, 360)
        x, y, vis = ortho(la_arr, lon, clat, clon)
        segs = []
        seg_x, seg_y = [], []
        for i in range(len(la_arr)):
            if vis[i]:
                seg_x.append(x[i]); seg_y.append(y[i])
            else:
                if seg_x:
                    segs.append(list(zip(seg_x, seg_y)))
                seg_x, seg_y = [], []
        if seg_x:
            segs.append(list(zip(seg_x, seg_y)))
        lc = LineCollection(segs, colors="#1e3a5f", linewidths=0.3,
                            linestyles="--", zorder=2)
        ax.add_collection(lc)

    # Coastlines
    for (lons, lats) in COASTLINES:
        x, y, vis = ortho(lats, lons, clat, clon)
        segs = []
        seg_x, seg_y = [], []
        for i in range(len(lons)):
            if vis[i]:
                seg_x.append(x[i]); seg_y.append(y[i])
            else:
                if seg_x:
                    segs.append(list(zip(seg_x, seg_y)))
                seg_x, seg_y = [], []
        if seg_x:
            segs.append(list(zip(seg_x, seg_y)))
        lc = LineCollection(segs, colors="#3a7a3a", linewidths=0.6, zorder=3)
        ax.add_collection(lc)

    # Equator highlight
    lo_arr = np.linspace(-180, 180, 720)
    x, y, vis = ortho(0.0, lo_arr, clat, clon)
    segs = []; seg_x, seg_y = [], []
    for i in range(len(lo_arr)):
        if vis[i]:
            seg_x.append(x[i]); seg_y.append(y[i])
        else:
            if seg_x: segs.append(list(zip(seg_x, seg_y)))
            seg_x, seg_y = [], []
    if seg_x: segs.append(list(zip(seg_x, seg_y)))
    lc = LineCollection(segs, colors="#1e4040", linewidths=0.6, zorder=3)
    ax.add_collection(lc)

    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)


# ── Episode renderer ──────────────────────────────────────────────────────────

def render_episode(recs: list[dict], ep_id: int, batch_id: int):
    n = len(recs)
    print(f"  Episode {ep_id} (batch {batch_id}): {n} frames ", end="", flush=True)

    rewards   = [r["reward"]   for r in recs]
    r_min, r_max = min(rewards), max(rewards)
    cum_rews  = list(np.cumsum(rewards))

    sat_lats  = [r["sat_lat"]  for r in recs]
    sat_lons  = [r["sat_lon"]  for r in recs]
    sat_alts  = [r["sat_alt"]  for r in recs]
    b_lats    = [r.get("boresight_lat", r["sat_lat"]) for r in recs]
    b_lons    = [r.get("boresight_lon", r["sat_lon"]) for r in recs]
    pitches   = [r.get("pitch", 0.0) for r in recs]
    rolls     = [r.get("roll",  0.0) for r in recs]
    sim_times = [r.get("sim_time", "") for r in recs]

    fig, ax_g, ax_r, ax_cr, ax_at = make_fig()

    # ── Static chart setup ────────────────────────────────────────────────
    steps = list(range(n))

    # Step reward bars
    ax_r.set_xlim(-0.5, n - 0.5)
    span = max(abs(r_min), abs(r_max), 0.01)
    ax_r.set_ylim(-span * 1.15, span * 1.15)
    ax_r.axhline(0, color=C_GREY, linewidth=0.4, linestyle="--")
    ax_r.set_title("Step reward", color=C_WHITE, fontsize=8, pad=3)
    bars = ax_r.bar(steps, [0]*n, width=0.8, color=C_GREY, alpha=0.3)

    # Cumulative reward
    c_min = min(0.0, min(cum_rews)); c_max = max(0.0, max(cum_rews))
    ax_cr.set_xlim(0, n); ax_cr.set_ylim(c_min - 0.05, c_max + 0.05)
    ax_cr.axhline(0, color=C_GREY, linewidth=0.4, linestyle="--")
    ax_cr.set_title("Cumulative reward", color=C_WHITE, fontsize=8, pad=3)
    cum_line, = ax_cr.plot([], [], color=C_GREEN, linewidth=1.5)
    cum_fills = []

    # Attitude
    p_lo = min(pitches) - 3;  p_hi = max(pitches) + 3
    r_lo = min(rolls)  - 3;   r_hi = max(rolls)  + 3
    ax_at.set_xlim(0, n)
    ax_at.set_ylim(min(p_lo, r_lo), max(p_hi, r_hi))
    ax_at.axhline(0, color=C_GREY, linewidth=0.4, linestyle="--")
    ax_at.set_title("Attitude", color=C_WHITE, fontsize=8, pad=3)
    pitch_ln, = ax_at.plot([], [], color=C_ORNG, linewidth=1.3, label="pitch")
    roll_ln,  = ax_at.plot([], [], color=C_PURP, linewidth=1.3, label="roll")
    ax_at.legend(fontsize=6, loc="upper right",
                 facecolor=BG_CARD, edgecolor=C_GREY, labelcolor=C_WHITE)

    # ── Title / HUD ───────────────────────────────────────────────────────
    fig.text(0.01, 0.965,
             f"Earth-Gym  ·  Eval episode {ep_id}  ·  Training batch {batch_id}",
             color=C_WHITE, fontsize=11, fontweight="bold", va="top",
             path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

    hud = fig.text(0.01, 0.038, "", color=C_WHITE, fontsize=7.5,
                   fontfamily="monospace", va="bottom",
                   path_effects=[pe.withStroke(linewidth=1.5, foreground=BG)])

    step_lbl = fig.text(0.40, 0.965, "", color=C_GREY, fontsize=9,
                        va="top", ha="right")

    # Legend patches (globe)
    legend_els = [
        mpatches.Patch(color=C_CYAN,  label="Footprint"),
        mpatches.Patch(color=C_YELL,  label="Targets"),
        mpatches.Patch(color=C_WHITE, label="Access ⚡"),
    ]
    ax_g.legend(handles=legend_els, loc="lower left",
                facecolor=BG_CARD, edgecolor=C_GREY,
                labelcolor=C_WHITE, fontsize=7, framealpha=0.8)

    # ── Per-frame update ──────────────────────────────────────────────────
    def update(fi: int):
        rec   = recs[fi]
        s_lat = sat_lats[fi]; s_lon = sat_lons[fi]; s_alt = sat_alts[fi]
        b_lat = b_lats[fi];   b_lon = b_lons[fi]
        pitch = pitches[fi];  roll  = rolls[fi]
        rew   = rewards[fi]

        # Centre globe on satellite
        clat = max(-80.0, min(80.0, s_lat))
        clon = s_lon

        # Redraw static globe (projection centre changes every frame)
        draw_globe_static(ax_g, clat, clon)

        # ── Coloured ground trail ─────────────────────────────────────
        start = max(0, fi - args.trail)
        for i in range(start, fi):
            x0, y0, v0 = ortho(sat_lats[i],   sat_lons[i],   clat, clon)
            x1, y1, v1 = ortho(sat_lats[i+1], sat_lons[i+1], clat, clon)
            if v0 and v1:
                fade  = 0.3 + 0.7 * (i - start) / max(fi - start, 1)
                color = rew_color(rewards[i], r_min, r_max)
                ax_g.plot([x0, x1], [y0, y1], "-",
                          color=color, linewidth=1.5,
                          alpha=fade * 0.85, zorder=5)

        # ── Footprint circle ──────────────────────────────────────────
        try:
            fp_lats, fp_lons = footprint_latlon(b_lat, b_lon, s_alt, args.cone)
            segs = []; seg_x, seg_y = [], []
            xs, ys, vis = ortho(fp_lats, fp_lons, clat, clon)
            for i in range(len(fp_lats)):
                if vis[i]:
                    seg_x.append(xs[i]); seg_y.append(ys[i])
                else:
                    if seg_x: segs.append(list(zip(seg_x, seg_y)))
                    seg_x, seg_y = [], []
            if seg_x: segs.append(list(zip(seg_x, seg_y)))
            ax_g.add_collection(
                LineCollection(segs, colors=C_CYAN, linewidths=1.0,
                               alpha=0.65, zorder=6))
        except Exception:
            pass

        # ── Boresight cross ───────────────────────────────────────────
        bx, by, bv = ortho(b_lat, b_lon, clat, clon)
        if bv:
            ax_g.plot(bx, by, "+", color=C_CYAN, markersize=9,
                      markeredgewidth=1.5, zorder=8)

        # ── FoR targets ───────────────────────────────────────────────
        targets = rec.get("targets", [])
        for t in targets:
            if len(t) < 2: continue
            t_lat, t_lon = t[0], t[1]
            pri = t[2] if len(t) > 2 else 0.5
            tx, ty, tv = ortho(t_lat, t_lon, clat, clon)
            if tv:
                ax_g.scatter(tx, ty, s=20 + 35*pri, marker="*",
                             color=C_YELL, alpha=0.8, zorder=9,
                             edgecolors="none")

        # ── Access event flash ────────────────────────────────────────
        access_events = rec.get("access_events", [])
        if access_events:
            # Flash on the nearest target in the FoR window
            for ev in access_events:
                if targets:
                    t = targets[0]
                    ex, ey, ev2 = ortho(t[0], t[1], clat, clon)
                    if ev2:
                        ax_g.scatter(ex, ey, s=160, marker="*",
                                     color=C_WHITE, zorder=12,
                                     edgecolors=C_YELL, linewidths=0.8)
                        ax_g.scatter(ex, ey, s=400, marker="o",
                                     color="none", zorder=11,
                                     edgecolors=C_WHITE, linewidths=1.0,
                                     alpha=0.4)

        # ── Satellite dot ─────────────────────────────────────────────
        sx, sy, sv = ortho(s_lat, s_lon, clat, clon)
        if sv:
            ax_g.scatter(sx, sy, s=80, color=C_WHITE, zorder=13,
                         edgecolors=C_BLUE, linewidths=2.0)
            # Velocity arrow: direction of next position
            if fi + 1 < n:
                nx, ny, nv = ortho(sat_lats[fi+1], sat_lons[fi+1], clat, clon)
                if nv:
                    dx = nx - sx; dy = ny - sy
                    norm = math.sqrt(dx*dx + dy*dy) or 1
                    ax_g.annotate("",
                        xy=(sx + dx/norm*0.08, sy + dy/norm*0.08),
                        xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->",
                                        color=C_BLUE, lw=1.5),
                        zorder=14)

        ax_g.set_xlim(-1.08, 1.08)
        ax_g.set_ylim(-1.08, 1.08)

        # ── Step reward bars ──────────────────────────────────────────
        for j, bar in enumerate(bars):
            if j <= fi:
                bar.set_height(rewards[j])
                bar.set_facecolor(rew_color(rewards[j], r_min, r_max))
                bar.set_alpha(0.85)
            else:
                bar.set_height(0)
                bar.set_alpha(0.2)

        # ── Cumulative reward ─────────────────────────────────────────
        xs_c = list(range(fi + 1))
        ys_c = cum_rews[:fi + 1]
        cum_line.set_data(xs_c, ys_c)

        for p in cum_fills:
            try: p.remove()
            except Exception: pass
        cum_fills.clear()
        if xs_c:
            pos = ax_cr.fill_between(xs_c, ys_c, 0,
                where=[y >= 0 for y in ys_c],
                color=C_GREEN, alpha=0.18, interpolate=True)
            neg = ax_cr.fill_between(xs_c, ys_c, 0,
                where=[y < 0 for y in ys_c],
                color=C_RED, alpha=0.18, interpolate=True)
            cum_fills.extend([pos, neg])

        # ── Attitude ──────────────────────────────────────────────────
        xs_a = list(range(fi + 1))
        pitch_ln.set_data(xs_a, pitches[:fi + 1])
        roll_ln.set_data(xs_a, rolls[:fi + 1])

        # ── HUD ───────────────────────────────────────────────────────
        n_acc = len(access_events)
        hud.set_text(
            f"Time: {sim_times[fi] or '—':<22}  "
            f"Lat {s_lat:+8.3f}°  Lon {s_lon:+9.3f}°  "
            f"Alt {s_alt:.1f} km  |  "
            f"Pitch {pitch:+6.1f}°  Roll {roll:+6.1f}°  |  "
            f"Reward {rew:+.4f}  Cum {cum_rews[fi]:+.4f}"
            + (f"  ⚡ {n_acc} ACCESS" if n_acc else "")
        )
        step_lbl.set_text(f"Step {fi+1}/{n}")

    # ── Write video ───────────────────────────────────────────────────────
    out_path = OUT_DIR / f"eval_ep{ep_id:03d}_batch{batch_id:05d}.{args.format}"
    writer = (
        FFMpegWriter(fps=args.fps,
                     metadata={"title": f"Earth-Gym Eval Ep{ep_id}"},
                     extra_args=["-vcodec","libx264","-crf","23",
                                 "-pix_fmt","yuv420p"])
        if args.format == "mp4"
        else PillowWriter(fps=args.fps)
    )

    with writer.saving(fig, str(out_path), dpi=args.dpi):
        for fi in range(n):
            update(fi)
            writer.grab_frame()
            if fi % 10 == 0:
                print(f"{fi+1}..", end="", flush=True)

    plt.close(fig)
    print(f" ✓  →  {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not IN_PATH.exists():
        sys.exit(f"ERROR: {IN_PATH} not found.")

    eps = load_episodes(IN_PATH)
    if not eps:
        sys.exit("ERROR: no records found.")

    targets = sorted(eps.keys())
    if args.episode is not None:
        if args.episode not in eps:
            sys.exit(f"ERROR: episode {args.episode} not in file. "
                     f"Available: {targets}")
        targets = [args.episode]

    print(f"Found {len(eps)} episode(s).  Rendering {len(targets)}.  "
          f"fps={args.fps}  dpi={args.dpi}  format={args.format}")

    for ep_id in targets:
        recs      = eps[ep_id]
        batch_id  = recs[0].get("train_batch", 0)
        render_episode(recs, ep_id, batch_id)


if __name__ == "__main__":
    main()
