"""
dashboard.py
============
Option 3 — CesiumJS 3D live dashboard.

Run in a separate terminal while ppo_earth_gym.py is training:

    python dashboard.py --out output/ --port 8050

Then open  http://localhost:8050  in any browser.

What you see
------------
Left panel (2/3 width):
  • CesiumJS 3D rotating Earth globe
  • Satellite trail (last N positions), coloured by reward
    (green = positive, red = negative)
  • Animated dot at the current satellite position
  • Sensor footprint ellipse on the ground
  • Target zones shown as coloured billboards (yellow star icons)

Right panel (1/3 width):
  • Live reward time-series chart (Chart.js)
  • Current step, total frames, mean reward, best reward
  • Last-position readout (lat, lon, alt, pitch, roll)
  • Attitude indicator (pitch/roll gauge SVG)
  • Auto-refreshes every 5 seconds

Requirements
------------
pip install flask flask-cors

CesiumJS Ion token
------------------
CesiumJS works without a token but shows a watermark and uses a
low-resolution Earth texture.  For high-resolution imagery:
  1.  Sign up for a free account at https://ion.cesium.com
  2.  Copy your default access token
  3.  Set the env variable:  export CESIUM_ION_TOKEN="eyJhb..."
  Or pass --ion-token "eyJhb..."

Without a token the dashboard uses OpenStreetMap imagery, which looks
good and requires no registration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from flask import Flask, jsonify, render_template_string, send_from_directory
try:
    from flask_cors import CORS
except ImportError:
    # flask-cors is optional — CORS headers are only needed when the
    # dashboard is served from a different origin than the training script.
    class CORS:
        def __init__(self, app, **kwargs): pass

from telemetry import TelemetryLogger

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Earth-Gym CesiumJS live dashboard.")
parser.add_argument("--out",       default="output/", help="Training output folder")
parser.add_argument("--port",      default=8050,      type=int)
parser.add_argument("--host",      default="0.0.0.0")
parser.add_argument("--trail",     default=500,        type=int,
                    help="Number of past positions to show on globe")
parser.add_argument("--refresh",   default=10,          type=int,
                    help="Dashboard auto-refresh interval (seconds)")
parser.add_argument("--cone",      default=10.0,       type=float,
                    help="Sensor cone half-angle (deg) — must match config")
parser.add_argument("--ion-token", default="",
                    help="CesiumJS Ion access token (optional)")
args, _ = parser.parse_known_args()


ION_TOKEN = args.ion_token or os.environ.get("CESIUM_ION_TOKEN", "")
ION_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJmODIyYWEzMC1jYTFjLTRkZTYtYWQ0Mi0yOTA5NDY2ZDc4MTEiLCJpZCI6NDMxNjA1LCJpc3MiOiJodHRwczovL2lvbi5jZXNpdW0uY29tIiwiYXVkIjoidW5kZWZpbmVkX2RlZmF1bHQiLCJpYXQiOjE3Nzg3NzEzMDJ9.qTaN0BMcW9yXa43dLJC5RE3seyucZKa7R-1eblSn1QU"

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────

def _telemetry_path() -> Path:
    return Path(args.out) / "telemetry.jsonl"


def _reward_to_color(r: float, r_min: float, r_max: float) -> str:
    """Map a reward value to a hex colour (#RRGGBB) via green→yellow→red."""
    span = r_max - r_min if r_max != r_min else 1.0
    t    = max(0.0, min(1.0, (r - r_min) / span))
    if t >= 0.5:
        # green → yellow
        red   = int((1.0 - t) * 2 * 255)
        green = 200
    else:
        # yellow → red
        red   = 200
        green = int(t * 2 * 255)
    return f"#{red:02X}{green:02X}30"


@app.route("/api/telemetry")
def api_telemetry():
    """
    Return the last N telemetry records as JSON for the CesiumJS frontend.
    """
    records = TelemetryLogger.load(_telemetry_path(), last_n=args.trail)
    if not records:
        return jsonify({"trail": [], "current": None, "summary": {}})

    rewards   = [r["reward"] for r in records]
    r_min, r_max = min(rewards), max(rewards)

    trail = [
        {
            "lat":   r["sat_lat"],
            "lon":   r["sat_lon"],
            "alt":   r["sat_alt"],
            "color": _reward_to_color(r["reward"], r_min, r_max),
        }
        for r in records
    ]

    current   = records[-1]
    eval_recs = [r for r in records if r.get("eval_reward") is not None]

    summary = {
        "step":          current["step"],
        "frame":         current.get("frame", "—"),
        "lat":           round(current["sat_lat"],   4),
        "lon":           round(current["sat_lon"],   4),
        "alt":           round(current["sat_alt"],   2),
        "pitch":         round(current.get("pitch", 0.0), 2),
        "roll":          round(current.get("roll",  0.0), 2),
        "boresight_lat": round(current.get("boresight_lat", current["sat_lat"]), 4),
        "boresight_lon": round(current.get("boresight_lon", current["sat_lon"]), 4),
        "reward":        round(current["reward"], 5),
        "mean_reward":   round(sum(rewards) / len(rewards), 5),
        "best_reward":   round(max(rewards), 5),
        "total_records": len(records),   # no second load — reuse same list
        "eval_rewards":  [r["eval_reward"] for r in eval_recs[-50:]],
        "eval_steps":    [r["step"]        for r in eval_recs[-50:]],
    }

    return jsonify({
        "trail":    trail,
        "current":  current,
        "summary":  summary,
        "r_min":    r_min,
        "r_max":    r_max,
        "targets":  current.get("targets", []),
        "cone_deg": args.cone,
    })


@app.route("/api/rewards")
def api_rewards():
    """Return full reward history for chart rendering."""
    records = TelemetryLogger.load(_telemetry_path())
    return jsonify({
        "steps":   [r["step"]   for r in records],
        "rewards": [r["reward"] for r in records],
    })


@app.route("/api/orbit")
def api_orbit():
    """
    Return a dense interpolated orbital ground track for the 3D globe.

    The telemetry has one record per RL step (delta_time = 890 s = T/8).
    That gives only 8 dots per orbit — far too sparse for a smooth 3D trail.
    This endpoint re-runs the propagator between consecutive telemetry records
    at N_SUB sub-steps each, producing ~160 points per orbit so CesiumJS can
    draw a smooth sinusoidal polyline.

    Returns a list of {lat, lon, alt_m, reward} dicts, one per sub-sample.
    """
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

    records = TelemetryLogger.load(_telemetry_path(), last_n=args.trail)
    if not records:
        return jsonify([])

    try:
        from scripts.propagator_base import make_propagator
        from scripts.coordinates import stk_date_to_et, eci_to_ecef, ecef_to_geodetic
        import json as _json

        cfg     = _json.load(open("src/agents-configuration.json"))
        general = cfg.get("general", {})
        ag_raw  = cfg["agents"][0]
        ag_flat = {}
        for v in ag_raw.values():
            if isinstance(v, dict):
                ag_flat.update(v)
        if "initial_orbital_elements" in ag_flat:
            ag_flat.update(ag_flat.pop("initial_orbital_elements"))

        et0      = stk_date_to_et(general["start_time"])
        prop     = make_propagator(ag_flat, general.get("propagator", "TwoBody"), et0)
        N_SUB    = 20
        DELTA    = 890.1   # must match earth_gym_env delta_time

        rewards  = [r["reward"] for r in records]
        r_min, r_max = min(rewards), max(rewards)

        points = []
        for rec in records:
            step   = rec.get("step", 0)
            reward = rec.get("reward", 0.0)
            span   = max(0.0, min(1.0,
                         (reward - r_min) / (r_max - r_min + 1e-9)))
            for k in range(N_SUB):
                # Telemetry step i is logged AFTER env.step() advances by DELTA,
                # so the satellite at step i is at simulation time (i+1)*DELTA.
                # The sub-points must therefore cover [(step)*DELTA, (step+1)*DELTA],
                # which means the base must be (step+1)*DELTA - DELTA + k*DELTA/N_SUB
                # = step*DELTA + k*DELTA/N_SUB  ← that's one step early.
                # Adding DELTA shifts the entire window forward by one step so the
                # last sub-point aligns with the satellite's actual logged position.
                t = et0 + (step + 1) * DELTA + k * DELTA / N_SUB
                r_eci, _, _ = prop.propagate(t)
                r_ecef      = eci_to_ecef(r_eci, t)
                lat, lon, alt = ecef_to_geodetic(r_ecef)
                points.append({
                    "lat":   round(lat, 5),
                    "lon":   round(lon, 5),
                    "alt_m": round(alt * 1000, 0),
                    "t":     round(span, 4),
                })
        return jsonify(points)

    except Exception as exc:
        # Fallback: just return the raw telemetry positions (1 per step)
        return jsonify([
            {
                "lat":   r["sat_lat"],
                "lon":   r["sat_lon"],
                "alt_m": r["sat_alt"] * 1000,
                "t":     0.5,
            }
            for r in records
        ])


@app.route("/images/<path:filename>")
def serve_image(filename):
    img_dir = Path(args.out) / "images"
    return send_from_directory(str(img_dir), filename)


@app.route("/api/targets_all")
def api_targets_all():
    """
    Return ALL active targets from the gym server's TargetManager.

    The telemetry only carries the 5 FoR observation slots (lat_1..5).
    This endpoint talks directly to the gym socket server to fetch all
    100 currently active targets with their priority and observation count,
    so the dashboard can render them all on the globe.
    """
    gym_host = os.environ.get("GYM_HOST", "localhost")
    gym_port = int(os.environ.get("GYM_PORT", 5555))
    try:
        import socket as _socket
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        s.settimeout(3.0)
        s.connect((gym_host, gym_port))
        import json as _json
        s.sendall(_json.dumps({"command": "get_targets"}).encode())
        data = _json.loads(s.recv(65536).decode())
        s.close()
        return jsonify(data.get("targets_all", []))
    except Exception as exc:
        # Gym server not running — return empty list gracefully
        return jsonify([])


@app.route("/api/images")
def api_images():
    """List available checkpoint images."""
    img_dir = Path(args.out) / "images"
    if not img_dir.exists():
        return jsonify([])
    files = sorted(img_dir.glob("all_*.png"))
    return jsonify([f.name for f in files[-10:]])


# ─────────────────────────────────────────────────────────────────────────────
# HTML page
# ─────────────────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Earth-Gym Live Dashboard</title>
<link rel="stylesheet" href="https://cesium.com/downloads/cesiumjs/releases/1.117/Build/Cesium/Widgets/widgets.css">
<script src="https://cesium.com/downloads/cesiumjs/releases/1.117/Build/Cesium/Cesium.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0a0e1a; color: #e0e6f0; height: 100vh;
       display: flex; flex-direction: column; overflow: hidden; }
header { background: #111827; border-bottom: 1px solid #1e2d45;
         padding: 8px 20px; display: flex; align-items: center; gap: 16px; }
header h1 { font-size: 15px; font-weight: 600; letter-spacing: 0.5px; color: #93c5fd; }
.badge { background: #1e3a5f; color: #60a5fa; font-size: 11px;
         padding: 2px 8px; border-radius: 10px; font-weight: 500; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: #4ade80;
              animation: pulse 2s infinite; margin-left: auto; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.status-dot.stale { background: #f59e0b; animation: none; }
.main { display: flex; flex: 1; overflow: hidden; }
#cesiumContainer { flex: 1; min-width: 0; }
.sidebar { width: 340px; min-width: 340px; background: #0f172a;
           border-left: 1px solid #1e2d45;
           display: flex; flex-direction: column; overflow-y: auto; }
.panel { padding: 14px 16px; border-bottom: 1px solid #1e2d45; }
.panel-title { font-size: 10px; font-weight: 600; letter-spacing: 1px;
               color: #64748b; text-transform: uppercase; margin-bottom: 10px; }
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.stat { background: #1e293b; border-radius: 8px; padding: 10px 12px; }
.stat-label { font-size: 10px; color: #64748b; margin-bottom: 4px; }
.stat-value { font-size: 18px; font-weight: 700; color: #e2e8f0;
              font-variant-numeric: tabular-nums; }
.stat-value.positive { color: #4ade80; }
.stat-value.negative { color: #f87171; }
.stat-value.neutral  { color: #93c5fd; }
.chart-wrap { height: 180px; position: relative; }
.gauge-wrap { display: flex; gap: 12px; align-items: center; justify-content: center;
              padding: 8px 0; }
.coord-row { display: flex; justify-content: space-between; font-size: 12px;
             padding: 4px 0; border-bottom: 1px solid #1e2d451a; }
.coord-label { color: #64748b; }
.coord-value { color: #cbd5e1; font-variant-numeric: tabular-nums; }
.img-thumb { width: 100%; border-radius: 6px; cursor: pointer;
             border: 1px solid #1e2d45; margin-top: 6px;
             transition: opacity .2s; }
.img-thumb:hover { opacity: 0.85; }
.refresh-bar { height: 3px; background: #1e3a5f; position: relative; overflow: hidden; }
.refresh-bar-inner { height: 100%; background: #3b82f6; width: 0%;
                     transition: width linear; }
</style>
</head>
<body>
<header>
  <h1>🛰 Earth-Gym Live Dashboard</h1>
  <span class="badge" id="step-badge">Waiting for data…</span>
  <span class="badge" id="frame-badge"></span>
  <div class="status-dot" id="status-dot"></div>
</header>
<div class="refresh-bar"><div class="refresh-bar-inner" id="refresh-bar"></div></div>
<div class="main">
  <div id="cesiumContainer"></div>
  <div class="sidebar">

    <div class="panel">
      <div class="panel-title">Training summary</div>
      <div class="stat-grid">
        <div class="stat">
          <div class="stat-label">Last reward</div>
          <div class="stat-value" id="s-reward">—</div>
        </div>
        <div class="stat">
          <div class="stat-label">Mean reward</div>
          <div class="stat-value neutral" id="s-mean">—</div>
        </div>
        <div class="stat">
          <div class="stat-label">Best reward</div>
          <div class="stat-value positive" id="s-best">—</div>
        </div>
        <div class="stat">
          <div class="stat-label">Total frames</div>
          <div class="stat-value neutral" id="s-frames">—</div>
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">Live reward</div>
      <div class="chart-wrap">
        <canvas id="rewardChart"></canvas>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">Satellite position</div>
      <div class="coord-row"><span class="coord-label">Latitude</span>
        <span class="coord-value" id="p-lat">—</span></div>
      <div class="coord-row"><span class="coord-label">Longitude</span>
        <span class="coord-value" id="p-lon">—</span></div>
      <div class="coord-row"><span class="coord-label">Altitude</span>
        <span class="coord-value" id="p-alt">—</span></div>
      <div class="coord-row"><span class="coord-label">Pitch</span>
        <span class="coord-value" id="p-pitch">—</span></div>
      <div class="coord-row"><span class="coord-label">Roll</span>
        <span class="coord-value" id="p-roll">—</span></div>
    </div>

    <div class="panel">
      <div class="panel-title">Attitude</div>
      <div class="gauge-wrap">
        <svg id="gauge-svg" width="120" height="120" viewBox="-60 -60 120 120">
          <circle r="55" fill="none" stroke="#1e293b" stroke-width="8"/>
          <circle r="55" fill="none" stroke="#334155" stroke-width="1"/>
          <!-- Horizon line (rotates with roll) -->
          <g id="horizon-group">
            <rect x="-55" y="-3" width="110" height="6" rx="2" fill="#3b82f6" opacity="0.7"/>
            <rect x="-55" y="-55" width="110" height="52" rx="0" fill="#1d4ed8" opacity="0.3"/>
          </g>
          <!-- Pitch marker (vertical translation) -->
          <g id="pitch-group">
            <line x1="-20" y1="0" x2="20" y2="0" stroke="#f59e0b" stroke-width="2"/>
          </g>
          <!-- Fixed aircraft symbol -->
          <line x1="-30" y1="0" x2="-10" y2="0" stroke="#f59e0b" stroke-width="3" stroke-linecap="round"/>
          <line x1="10"  y1="0" x2="30"  y2="0" stroke="#f59e0b" stroke-width="3" stroke-linecap="round"/>
          <circle r="4" fill="#f59e0b"/>
        </svg>
        <div style="font-size:11px;color:#64748b;text-align:center;line-height:1.8">
          Pitch<br><span id="g-pitch" style="color:#e2e8f0;font-size:14px">0.0°</span><br>
          Roll<br><span id="g-roll" style="color:#e2e8f0;font-size:14px">0.0°</span>
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">Target legend</div>
      <div style="font-size:11px;line-height:2">
        <span style="color:#fbbf24">●</span> In Field of Regard (FoR)<br>
        <span style="color:#4ade80">●</span> In FoR + observed<br>
        <span style="color:#22d3ee">●</span> Observed, outside FoR<br>
        <span style="color:#f87171">●</span> Not yet observed<br>
        <span style="color:#94a3b8">Size ∝ priority &nbsp;·&nbsp; label = priority×10</span>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">Checkpoint images</div>
      <div id="img-list"><span style="font-size:11px;color:#64748b">
        No images yet. Generated every 100 batches.</span></div>
    </div>

  </div>
</div>

<script>
const REFRESH_S = {{ refresh }};
const CONE_DEG  = {{ cone }};
const ION_TOKEN = "{{ ion_token }}";

if (ION_TOKEN) Cesium.Ion.defaultAccessToken = ION_TOKEN;

const viewer = new Cesium.Viewer("cesiumContainer", {
  imageryProvider: ION_TOKEN
    ? undefined
    : new Cesium.OpenStreetMapImageryProvider({
        url: "https://tile.openstreetmap.org/"
      }),
  terrainProvider: new Cesium.EllipsoidTerrainProvider(),
  animation:        false,
  baseLayerPicker:  false,
  fullscreenButton: false,
  geocoder:         false,
  homeButton:       false,
  infoBox:          false,
  sceneModePicker:  false,
  selectionIndicator: false,
  timeline:         false,
  navigationHelpButton: false,
  creditContainer:  document.createElement("div"),
});

viewer.scene.backgroundColor = Cesium.Color.fromCssColorString("#0a0e1a");
viewer.scene.globe.enableLighting = true;
viewer.scene.globe.atmosphereLightIntensity = 10.0;

let trailEntities  = [];
let satEntity      = null;
let footprintEntity = null;
let targetEntities = [];
let rewardChart    = null;
let lastStep       = -1;
let coneEntity     = null;

function hexToCesiumColor(hex) {
  return Cesium.Color.fromCssColorString(hex).withAlpha(0.85);
}

// Reward [0,1] → CesiumJS Color  (red=low, yellow=mid, green=high)
function tToColor(t) {
  if (t >= 0.5) {
    const red   = Math.round((1.0 - t) * 2 * 255);
    return Cesium.Color.fromBytes(red, 200, 48, 217);
  } else {
    const green = Math.round(t * 2 * 255);
    return Cesium.Color.fromBytes(200, green, 48, 217);
  }
}

/**
 * Sensor footprint radius on Earth surface (metres).
 * Uses exact spherical geometry — the flat-Earth approximation
 * (altKm * tan(cone)) over-estimates at large cone angles and
 * is inaccurate for LEO altitudes.
 *
 * Derivation: spherical law of sines on Earth–sub-sat–footprint-edge triangle.
 *   rho      = arccos(RT / (RT + alt))        [max Earth half-angle]
 *   nadir    = arcsin(sin(cone + 90°) * (RT+alt) / RT)
 *   θ_earth  = 90° − cone − (180° − nadir)    [Earth central angle]
 *   footprint radius = RT * |θ_earth| (km) × 1000
 */
function footprintRadius(altKm) {
  const RT       = 6371.0;
  const rSat     = RT + Math.max(altKm, 0);
  const coneRad  = CONE_DEG * Math.PI / 180;
  const rho      = Math.acos(RT / rSat);
  if (coneRad >= rho) return rho * RT * 1000;   // clamp to full visible disk
  const sinNadir = Math.sin(Math.PI / 2 + coneRad) * rSat / RT;
  const nadirRad = Math.asin(Math.min(1, sinNadir));
  const earthAng = Math.abs(Math.PI / 2 - coneRad - (Math.PI - nadirRad));
  return earthAng * RT * 1000;   // metres
}

function initChart() {
  const ctx = document.getElementById("rewardChart").getContext("2d");
  rewardChart = new Chart(ctx, {
    type: "line",
    data: {
      labels:   [],
      datasets: [{
        label: "Step reward",
        data: [],
        borderColor: "#3b82f6",
        backgroundColor: "rgba(59,130,246,0.08)",
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
      }, {
        label: "Eval reward",
        data: [],
        borderColor: "#f59e0b",
        backgroundColor: "transparent",
        borderWidth: 2,
        pointRadius: 3,
        pointStyle: "diamond",
        fill: false,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { labels: { color: "#94a3b8", font: { size: 10 } } }
      },
      scales: {
        x: { ticks: { color: "#475569", maxTicksLimit: 6, font: { size: 9 } },
             grid:  { color: "#1e293b" } },
        y: { ticks: { color: "#475569", font: { size: 9 } },
             grid:  { color: "#1e293b" } }
      }
    }
  });
}

function updateGauge(pitch, roll) {
  const pitchPx = Math.max(-40, Math.min(40, pitch * 0.8));
  document.getElementById("pitch-group").setAttribute(
    "transform", `translate(0, ${pitchPx})`);
  document.getElementById("horizon-group").setAttribute(
    "transform", `rotate(${-roll})`);
  document.getElementById("g-pitch").textContent = pitch.toFixed(1) + "°";
  document.getElementById("g-roll").textContent  = roll.toFixed(1)  + "°";
}

async function refresh() {
  try {
    const res  = await fetch("/api/telemetry");
    const data = await res.json();

    if (!data.current) {
      document.getElementById("status-dot").className = "status-dot stale";
      return;
    }

    document.getElementById("status-dot").className = "status-dot";

    const s = data.summary;
    const step = s.step;

    document.getElementById("step-badge").textContent  = `Batch ${step}`;
    document.getElementById("frame-badge").textContent = `${s.total_records} frames`;

    // Stats
    const rv = s.reward;
    document.getElementById("s-reward").textContent = rv.toFixed(4);
    document.getElementById("s-reward").className =
      "stat-value " + (rv > 0 ? "positive" : rv < 0 ? "negative" : "neutral");
    document.getElementById("s-mean").textContent   = s.mean_reward.toFixed(4);
    document.getElementById("s-best").textContent   = s.best_reward.toFixed(4);
    document.getElementById("s-frames").textContent = s.total_records.toLocaleString();

    // Position
    document.getElementById("p-lat").textContent   = s.lat.toFixed(4) + " °";
    document.getElementById("p-lon").textContent   = s.lon.toFixed(4) + " °";
    document.getElementById("p-alt").textContent   = s.alt.toFixed(1) + " km";
    document.getElementById("p-pitch").textContent = s.pitch.toFixed(2) + " °";
    document.getElementById("p-roll").textContent  = s.roll.toFixed(2)  + " °";
    updateGauge(s.pitch, s.roll);

    // Chart
    if (step !== lastStep) {
      lastStep = step;
      const trail = data.trail;
      const N = Math.min(trail.length, 200);
      rewardChart.data.labels = trail.slice(-N).map((_, i) => i);
      rewardChart.data.datasets[0].data = trail.slice(-N).map(t => {
        const norm = (data.r_max - data.r_min);
        return norm ? (t.color ? null : 0) : 0;
      });

      // Reload full reward history for chart
      const rRes = await fetch("/api/rewards");
      const rData = await rRes.json();
      const chunk = 5;
      const labels = [], vals = [];
      for (let i = 0; i < rData.steps.length; i += chunk) {
        const slice = rData.rewards.slice(i, i + chunk);
        labels.push(rData.steps[i]);
        vals.push(slice.reduce((a, b) => a + b, 0) / slice.length);
      }
      rewardChart.data.labels = labels;
      rewardChart.data.datasets[0].data = vals;
      rewardChart.data.datasets[1].data = s.eval_steps.map((st, i) =>
        ({ x: st, y: s.eval_rewards[i] }));
      rewardChart.update("none");
    }

    // ── Globe: dense orbital ground track ─────────────────────────────────
    // Fetch the propagator-interpolated track (20 sub-steps per RL step).
    // This replaces the sparse telemetry trail (1 dot/step ≈ 1/8 orbit)
    // with a smooth sinusoidal polyline on the 3D globe.
    trailEntities.forEach(e => viewer.entities.remove(e));
    trailEntities = [];

    let orbitPoints = null;
    try {
      const oRes = await fetch("/api/orbit");
      orbitPoints  = await oRes.json();
    } catch (_) {}

    if (orbitPoints && orbitPoints.length > 1) {
      // Draw as a single polyline per contiguous segment
      // (split on anti-meridian jumps > 180° to avoid wrap-around artefacts)
      let seg = [];
      for (let i = 0; i < orbitPoints.length; i++) {
        const p = orbitPoints[i];
        if (seg.length > 0) {
          const prev = seg[seg.length - 1];
          if (Math.abs(p.lon - prev.lon) > 180) {
            // Anti-meridian crossing — flush current segment then start new one
            if (seg.length >= 2) {
              const coords = seg.flatMap(s => [s.lon, s.lat, s.alt_m]);
              trailEntities.push(viewer.entities.add({
                polyline: {
                  positions: Cesium.Cartesian3.fromDegreesArrayHeights(coords),
                  width:     2.5,
                  material:  new Cesium.PolylineGlowMaterialProperty({
                    glowPower: 0.12,
                    color:     tToColor(seg[seg.length - 1].t),
                  }),
                  clampToGround: false,
                }
              }));
            }
            seg = [];
          }
        }
        seg.push(p);
      }
      // Flush last segment
      if (seg.length >= 2) {
        const coords = seg.flatMap(s => [s.lon, s.lat, s.alt_m]);
        trailEntities.push(viewer.entities.add({
          polyline: {
            positions: Cesium.Cartesian3.fromDegreesArrayHeights(coords),
            width:     2.5,
            material:  new Cesium.PolylineGlowMaterialProperty({
              glowPower: 0.12,
              color:     tToColor(seg[seg.length - 1].t),
            }),
            clampToGround: false,
          }
        }));
      }
    }

    // ── Satellite entity (current position) ───────────────────────────────
    if (satEntity) viewer.entities.remove(satEntity);
    const trail = data.trail;
    const cur   = trail[trail.length - 1];
    satEntity = viewer.entities.add({
      position: Cesium.Cartesian3.fromDegrees(cur.lon, cur.lat, cur.alt * 1000),
      point: {
        pixelSize:        14,
        color:            Cesium.Color.WHITE,
        outlineColor:     Cesium.Color.fromCssColorString("#3b82f6"),
        outlineWidth:     3,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
        scaleByDistance:  new Cesium.NearFarScalar(1.5e6, 1.5, 1.5e8, 0.7),
      },
      label: {
        text:             "🛰",
        font:             "14px sans-serif",
        pixelOffset:      new Cesium.Cartesian2(0, -22),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      }
    });

    // ── Footprint ellipse (at boresight intercept, not sub-sat point) ─────
    const b_lat = data.summary.boresight_lat;
    const b_lon = data.summary.boresight_lon;
    const altKm = data.summary.alt;
    if (footprintEntity) viewer.entities.remove(footprintEntity);
    footprintEntity = viewer.entities.add({
      position: Cesium.Cartesian3.fromDegrees(b_lon, b_lat, 0),
      ellipse: {
        semiMajorAxis: footprintRadius(altKm),
        semiMinorAxis: footprintRadius(altKm),
        material:      Cesium.Color.fromCssColorString("#fbbf24").withAlpha(0.15),
        outline:       true,
        outlineColor:  Cesium.Color.fromCssColorString("#fbbf24").withAlpha(0.6),
        outlineWidth:  2,
        height:        0,
      }
    });

    // ── Sensor cone (original working logic) ──────────────────────────────
    if (coneEntity) viewer.entities.remove(coneEntity);

    const satPos    = Cesium.Cartesian3.fromDegrees(cur.lon, cur.lat, cur.alt * 1000);
    const targetPos = Cesium.Cartesian3.fromDegrees(b_lon, b_lat, 0);
    const distance  = Cesium.Cartesian3.distance(satPos, targetPos);

    if (!isNaN(distance) && distance > 0) {
      const midPoint = Cesium.Cartesian3.midpoint(satPos, targetPos, new Cesium.Cartesian3());

      // Direction from base (ground) toward apex (satellite)
      const direction = Cesium.Cartesian3.subtract(satPos, targetPos, new Cesium.Cartesian3());
      Cesium.Cartesian3.normalize(direction, direction);

      const orientation = new Cesium.Quaternion();
      const dot = Cesium.Cartesian3.dot(Cesium.Cartesian3.UNIT_Z, direction);

      if (dot < -0.999999) {
        Cesium.Quaternion.fromAxisAngle(Cesium.Cartesian3.UNIT_X, Math.PI, orientation);
      } else if (dot < 0.999999) {
        const axis = Cesium.Cartesian3.cross(Cesium.Cartesian3.UNIT_Z, direction, new Cesium.Cartesian3());
        Cesium.Cartesian3.normalize(axis, axis);
        const angle = Math.acos(dot);
        Cesium.Quaternion.fromAxisAngle(axis, angle, orientation);
      }

      coneEntity = viewer.entities.add({
        position:    midPoint,
        orientation: orientation,
        cylinder: {
          length:       distance,
          topRadius:    0.0,
          bottomRadius: footprintRadius(altKm),
          material:     Cesium.Color.fromCssColorString("#fbbf24").withAlpha(0.08),
          outline:      false,
        }
      });
    }

    // ── Target zones — all active targets ─────────────────────────────────
    // Fetch all 100 targets from the gym server (not just the 5 FoR slots
    // stored in telemetry).  Colour by priority; observed targets turn green.
    targetEntities.forEach(e => viewer.entities.remove(e));
    targetEntities = [];

    let allTargets = [];
    try {
      const tRes  = await fetch("/api/targets_all");
      allTargets  = await tRes.json();
    } catch (_) {}

    // Fall back to the 5 FoR slots from telemetry if server not reachable
    if (allTargets.length === 0) {
      allTargets = (data.targets || []).map(([lat, lon, pri]) =>
        ({ lat, lon, priority: pri, n_obs: 0, name: "" })
      );
    }

    // FoR slots from the current observation (highlight these)
    const forSet = new Set(
      (data.targets || []).map(([lat, lon]) => `${lat.toFixed(3)},${lon.toFixed(3)}`)
    );

    for (const tgt of allTargets) {
      const inFoR    = forSet.has(
        `${tgt.lat.toFixed(3)},${tgt.lon.toFixed(3)}`
      );
      const observed = tgt.n_obs > 0;
      const pri      = tgt.priority ?? 0.5;

      // Colour scheme:
      //   in FoR + observed → bright green
      //   in FoR only       → yellow-orange (agent can see it now)
      //   observed only     → teal (seen before, not in FoR now)
      //   neither           → red-pink (not yet seen, out of FoR)
      let color;
      if (inFoR && observed)    color = Cesium.Color.fromCssColorString("#4ade80");
      else if (inFoR)           color = Cesium.Color.fromCssColorString("#fbbf24");
      else if (observed)        color = Cesium.Color.fromCssColorString("#22d3ee");
      else                      color = Cesium.Color.fromCssColorString("#f87171");

      targetEntities.push(viewer.entities.add({
        position: Cesium.Cartesian3.fromDegrees(tgt.lon, tgt.lat, 0),
        point: {
          pixelSize:  Math.max(4, Math.round(4 + 8 * pri)),
          color:      color.withAlpha(0.55 + 0.45 * pri),
          outlineColor: Cesium.Color.WHITE.withAlpha(0.35),
          outlineWidth: 1,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        label: inFoR ? {
          text:       `${(pri * 10).toFixed(1)}`,
          font:       "10px sans-serif",
          fillColor:  Cesium.Color.WHITE,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 1,
          style:      Cesium.LabelStyle.FILL_AND_OUTLINE,
          pixelOffset: new Cesium.Cartesian2(0, -14),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        } : undefined,
      }));
    }

  } catch (err) {
    console.warn("Dashboard refresh error:", err);
    document.getElementById("status-dot").className = "status-dot stale";
  }

  // Checkpoint images
  try {
    const imgs = await (await fetch("/api/images")).json();
    const div  = document.getElementById("img-list");
    if (imgs.length > 0) {
      div.innerHTML = imgs.reverse().map(f =>
        `<img src="/images/${f}" class="img-thumb"
              title="${f}"
              onclick="window.open('/images/${f}','_blank')">`
      ).join("");
    }
  } catch (_) {}
}

function startRefreshBar() {
  const bar  = document.getElementById("refresh-bar");
  const step = 100 / (REFRESH_S * 20);
  let pct    = 0;
  bar.style.width = "0%";
  bar.style.transition = `width ${REFRESH_S}s linear`;
  requestAnimationFrame(() => { bar.style.width = "100%"; });
}

initChart();
refresh();
setInterval(() => { refresh(); startRefreshBar(); }, REFRESH_S * 1000);
startRefreshBar();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(
        _HTML,
        refresh=args.refresh,
        cone=args.cone,
        ion_token=ION_TOKEN,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"[Dashboard] Serving on  http://localhost:{args.port}")
    print(f"[Dashboard] Reading telemetry from  {args.out}/telemetry.jsonl")
    if not ION_TOKEN:
        print("[Dashboard] No Ion token — using OpenStreetMap imagery.")
        print("            For high-res Earth: export CESIUM_ION_TOKEN=<your_token>")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
