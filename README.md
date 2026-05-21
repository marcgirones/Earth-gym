
## Installation

```bash
pip install -r requirements.txt
```

SPICE kernels (~15 MB) are downloaded automatically on the first run.
To pre-download them manually:

```python
from scripts.coordinates import download_kernels, load_spice_kernels
download_kernels()   # saves to ./spice_kernels/
```

If the kernel download fails (e.g. no internet access), the code falls back to
an IAU 1982 GMST rotation and Bowring geodetic iteration, accurate to a few
metres — more than enough for RL training.

---

## Usage

```bash
python src/main.py \
    --conf  src/agents-configuration.json \
    --evpt  data/event_zones.csv \
    --out   output/ \
    --host  localhost \
    --port  5555
```

Add `--pro 1` to enable cProfile + tracemalloc output.

---

## Project layout

```
earth-gym-oss/
├── src/
│   ├── main.py                    # entry point (socket server)
│   └── agents-configuration.json # scenario / agent config
├── scripts/
│   ├── instances.py               # Gym + SpiceEnvironment (was STKEnvironment)
│   ├── utils.py                   # manager classes (DateManager, Rewarder, ...)
│   ├── propagator.py              # analytical orbit propagator
│   └── coordinates.py             # SPICE frame transforms + geodetic
├── spice_kernels/                 # auto-populated by download_kernels()
├── data/                          # event-zone CSV files
├── docs/
│   ├── demo.py
│   ├── format-launch.json
│   ├── get-next.json
│   └── format-run.sh
└── requirements.txt
```

---

## Propagator models

Select the model in `agents-configuration.json` under `"propagator"`:

| Value | Description |
|---|---|
| `"TwoBody"` | Pure Keplerian (no perturbations) |
| `"J2Perturbation"` | Secular J2 oblateness drift (RAAN, AoP, mean motion) |
| `"J4Perturbation"` | Secular J2 + J4 oblateness drift |

> **HPOP** has no direct open-source equivalent here. For higher fidelity,
> initialise from a TLE with `sgp4` and pass the Cartesian state to
> `OrbitalPropagator`, or swap in a full numerical integrator (e.g.
> `poliastro`, `tudatpy`).

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Vectors, matrix algebra |
| `scipy` | Kepler solver, PCHIP LLA interpolation, attitude rotations |
| `pandas` | Target-zone catalogue, event dataframes |
| `matplotlib` | Reward plots |
| `spiceypy` | SPICE bindings: str2et, pxform, recgeo, georec |
| `psutil` | Memory reporting |
