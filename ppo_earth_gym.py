"""
ppo_earth_gym.py
================
Connects the PPO training loop from main.py to the Earth-Gym satellite
environment via the EarthGymEnv Gymnasium wrapper.

Run from the project root (earth-gym-oss/):
    python ppo_earth_gym.py

Two-terminal mode (server already running separately):
    # Terminal 1 — server
    python src/main.py --conf src/agents-configuration.json \\
                       --evpt data/sample-zones.csv --out output/

    # Terminal 2 — PPO training
    python ppo_earth_gym.py --no-server

The file is intentionally structured in three sections:
  [A]  Earth-Gym boot (server + wrapper)
  [B]  PPO setup copied verbatim from main.py with MODIFICATION notes
  [C]  Training loop identical to main.py

Every line that must differ from main.py is preceded by a comment block
starting with  ## MODIFICATION — read these before editing main.py directly.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import os
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Make sure the project root (earth-gym-oss/) is on sys.path so that the
# local modules (earth_gym_env, scripts.*) are importable regardless of where
# the script is launched from.
# ─────────────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# PyTorch / TorchRL  (same imports as main.py)
# ─────────────────────────────────────────────────────────────────────────────
from torch import multiprocessing
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import Compose, DoubleToFloat, ObservationNorm, StepCounter, TransformedEnv
from torchrl.envs.libs.gym import GymEnv, GymWrapper
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")           # headless — avoids Tk errors on servers
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# [A]  Earth-Gym: server boot + Gymnasium wrapper
# ─────────────────────────────────────────────────────────────────────────────

## MODIFICATION 1 — Replace GymEnv("HalfCheetah-v5") with EarthGymEnv
#
# In main.py:
#     base_env = GymEnv("HalfCheetah-v5", device=device)
#
# Here we import EarthGymEnv (the Gymnasium wrapper in earth_gym_env.py) and
# wrap it with GymEnv so TorchRL can treat it identically to any Gym env.
# GymEnv from TorchRL converts a Gymnasium env to a TorchRL EnvBase, handling
# TensorDict I/O automatically.

from earth_gym_env import EarthGymEnv, ServerProcess
from telemetry  import TelemetryLogger
from visualizer import generate_all as visualize_checkpoint

# ── CLI args (optional — defaults work out of the box) ───────────────────────
parser = argparse.ArgumentParser(description="PPO on Earth-Gym satellite env.")
parser.add_argument("--conf",      default="src/agents-configuration.json",
                    help="Path to agents-configuration.json")
parser.add_argument("--evpt",      default="data/sample-zones.csv",
                    help="Path to event-zones CSV")
parser.add_argument("--out",       default="output/",
                    help="Output folder for plots and reward logs")
parser.add_argument("--host",      default="localhost")
parser.add_argument("--port",      default=5555, type=int)
parser.add_argument("--delta-time",default=5553.5, type=float,
                    help="Seconds per env step (default ≈ one ISS orbital period)")
parser.add_argument("--no-server", action="store_true",
                    help="Skip launching the server (assumes it is already running)")
parser.add_argument("--cone",      default=10.0,  type=float,
                    help="Sensor cone half-angle (deg) — must match agents-configuration.json")
args = parser.parse_args()

os.makedirs(args.out,       exist_ok=True)
os.makedirs("videos",       exist_ok=True)
os.makedirs("graphs",       exist_ok=True)

# ── Launch server (unless --no-server) ───────────────────────────────────────
server_proc = None
if not args.no_server:
    print("[EarthGym] Starting environment server …")
    server_proc = ServerProcess(
        conf_path=args.conf,
        evpt_path=args.evpt,
        out_path=args.out,
        host=args.host,
        port=args.port,
    )
    server_proc.start(timeout=90)
    print("[EarthGym] Server ready.")

# ─────────────────────────────────────────────────────────────────────────────
# [B]  PPO hyper-parameters  (copied verbatim from main.py)
# ─────────────────────────────────────────────────────────────────────────────
is_fork    = multiprocessing.get_start_method() == "fork"
device     = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
num_cells      = 256
lr             = 3e-4
max_grad_norm  = 1.0

## MODIFICATION 2 — Reduce frames_per_batch and total_frames for Earth-Gym
#
# In main.py:
#     frames_per_batch = 1000
#     total_frames     = 2_000_000
#
# Earth-Gym steps are slow (each involves socket I/O + orbit propagation).
# One "frame" here equals one orbital period (~92 min simulated time).
# Recommended starting values:
#     frames_per_batch = 64      (64 orbital steps per batch)
#     total_frames     = 50_000  (≈ 781 batches)
#
# Increase total_frames once you confirm the training loop runs correctly.
#
frames_per_batch = 64
total_frames     = 50_000

sub_batch_size = 16    # reduced from 64 — Earth-Gym batches are smaller
num_epochs     = 10
# ── PPO stability parameters ────────────────────────────────────────────────
# clip_epsilon: how much the new policy can deviate from the old one per update.
# 0.2 (default) is too large here — the policy was updating too aggressively,
# causing it to collapse after batch ~120.  0.1 gives softer, more stable updates.
clip_epsilon   = 0.1
gamma          = 0.99
lmbda          = 0.95
# entropy_eps: weight of the entropy bonus that keeps the policy from becoming
# too deterministic.  1e-4 (default) was far too small — exploration collapsed
# quickly and the agent converged to a narrow local optimum.
# 1e-2 provides a meaningful regularisation signal throughout training.
entropy_eps    = 1e-2

# ─────────────────────────────────────────────────────────────────────────────
# [B]  Environment construction
# ─────────────────────────────────────────────────────────────────────────────

## MODIFICATION 3 — Observation feature list must match agents-configuration.json
#
# In main.py:
#     base_env = GymEnv("HalfCheetah-v5", device=device)
#
# The observation vector is built from "states_features" in the agent config.
# Default config (src/agents-configuration.json) has:
#   ["pitch", "roll", "detic_lat", "detic_lon", "detic_alt",
#    "lat_5", "lon_5", "priority_5"]
#
# lat_5/lon_5/priority_5 expand to 5 target triplets → 15 features total.
# Total observation size: 5 (attitude+LLA) + 15 (targets) = 20.
#
# If you change "states_features" in the JSON you MUST update obs_features here.

obs_features = [
    "pitch", "roll",
    "detic_lat", "detic_lon", "detic_alt",
    "lat_1", "lon_1", "priority_1",
    "lat_2", "lon_2", "priority_2",
    "lat_3", "lon_3", "priority_3",
    "lat_4", "lon_4", "priority_4",
    "lat_5", "lon_5", "priority_5",
]

## MODIFICATION 4 — Action feature list and limit must match the agent config
#
# Default config: "actions_features": ["d_pitch", "d_roll"]
#                 "max_slew_speed": 10  (deg/step)
#
# If you switch to sensor slewing ("d_az", "d_el") change action_features and
# action_limit to match "max_sensor_slew" (default 1 deg/step).
#
action_features = ["d_pitch", "d_roll"]
action_limit    = 10.0   # deg — must match max_slew_speed in config

## MODIFICATION 5 — Wrap EarthGymEnv with GymWrapper (TorchRL adapter)
#
# In main.py:
#     base_env = GymEnv("HalfCheetah-v5", device=device)
#
# GymEnv only accepts a string env name — it cannot wrap a pre-built object.
# Use GymWrapper instead: it takes an already-instantiated Gymnasium env and
# converts it to a TorchRL EnvBase. The rest of the PPO code is unchanged.

_gym_env  = EarthGymEnv(
    host=args.host,
    port=args.port,
    delta_time=args.delta_time,
    obs_features=obs_features,
    action_features=action_features,
    action_limit=action_limit,
)
base_env  = GymWrapper(
    env=_gym_env,
    device=device,
)

## MODIFICATION 6 — ObservationNorm init_stats: reduce num_iter
#
# In main.py:
#     env.transform[0].init_stats(num_iter=1000, ...)
#
# 1000 random steps against Earth-Gym would take a very long time.
# Use num_iter=50 (50 orbital periods) to collect normalization statistics.
# This corresponds to roughly 3 simulated days, which is representative.

env = TransformedEnv(
    base_env,
    Compose(
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        StepCounter(),
    ),
)
env.transform[0].init_stats(num_iter=50, reduce_dim=0, cat_dim=0)
print("normalization constant shape:", env.transform[0].loc.shape)

## MODIFICATION 7 — FlipPenalty Transform is HalfCheetah-specific: REMOVE IT
#
# In main.py the FlipPenalty transform reads obs[..., 1] as the torso angle.
# Earth-Gym observations have a completely different structure (see obs_features
# above), so FlipPenalty will silently penalise the wrong thing or crash.
#
# REMOVE or comment out FlipPenalty entirely.  To add custom reward shaping
# for Earth-Gym, write a new Transform that reads the features by name, e.g.:
#
#   class EarlyCoverageBonus(Transform):
#       def _call(self, tensordict):
#           reward = tensordict.get(("next", "reward"))
#           obs    = tensordict.get("observation")           # shape: [... x n_obs]
#           # index 4 = detic_alt; index 0 = pitch; etc.
#           ...
#           tensordict.set(("next", "reward"), reward + bonus)
#           return tensordict

# ─────────────────────────────────────────────────────────────────────────────
# [B]  Actor and Critic networks  (identical to main.py)
# ─────────────────────────────────────────────────────────────────────────────

actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor(),
)

policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
)

policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low":  env.action_spec.space.low,
        "high": env.action_spec.space.high,
    },
    return_log_prob=True,
)

value_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device),
)

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)

with torch.no_grad():
    td = env.reset()
    policy_module(td)
    value_module(td)

# ─────────────────────────────────────────────────────────────────────────────
# [B]  Collector, replay buffer, loss, optimiser  (identical to main.py)
# ─────────────────────────────────────────────────────────────────────────────

collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda,
    value_network=value_module,
    average_gae=True,
    device=device,
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coeff=entropy_eps,
    critic_coeff=1.0,
    loss_critic_type="smooth_l1",
)

optim     = torch.optim.Adam(loss_module.parameters(), lr)
# CosineAnnealingLR with eta_min=0 was decaying the lr to zero by end of
# training, which caused the observed reward collapse after batch ~120.
# Using a constant lr (no scheduler) gives stable learning throughout.
# If you want decay, use eta_min=lr/10 so the policy never fully stops updating.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, eta_min=lr / 10
)
# ─────────────────────────────────────────────────────────────────────────────
# Plots  (identical to main.py except save path)
# ─────────────────────────────────────────────────────────────────────────────
def plotter(logs):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(logs["reward"])
    plt.title("Training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(logs["step_count"])
    plt.title("Max step count (training)")
    plt.subplot(2, 2, 3)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (evaluation)")
    plt.subplot(2, 2, 4)
    plt.plot(logs["eval step_count"])
    plt.title("Max step count (evaluation)")
    plt.tight_layout()
    plt.savefig("graphs/ppo_earth_gym_results.png")
    print("[Done] Training complete. Plot saved to graphs/ppo_earth_gym_results.png")

# ─────────────────────────────────────────────────────────────────────────────
# [C]  Training loop
# ─────────────────────────────────────────────────────────────────────────────

logs     = defaultdict(list)
pbar     = tqdm(total=total_frames)
eval_str = ""

# ── Telemetry + visualizer ────────────────────────────────────────────────────
# TelemetryLogger writes one JSON record per batch to output/telemetry.jsonl.
# This file is read by both visualizer.py (matplotlib images) and
# dashboard.py (CesiumJS live globe) without any changes to the training logic.
telemetry = TelemetryLogger(args.out)

def _obs_to_dict(td) -> dict:
    """
    Decode the last normalised observation in a TorchRL tensordict back to
    raw (unscaled) feature values, for telemetry logging only.

    Root-cause of the previous lat/lon/alt visualisation bug
    ---------------------------------------------------------
    The original implementation tried to access EarthGymEnv.last_raw_state via:

        base = getattr(env, "base_env", env)  # intended: TransformedEnv → EarthGymEnv
        raw_state = getattr(base, "last_raw_state", {})

    BUT TorchRL's TransformedEnv stores its inner env under `.env`, not `.base_env`.
    So `getattr(env, "base_env", env)` silently returned `env` itself
    (the TransformedEnv), which has no `last_raw_state`.  The result was
    `raw_state = {}` on every call — the fallback path was always taken.

    The fallback inverts ObservationNorm:  raw ≈ obs_norm * scale + loc.
    For features with near-zero variance (detic_alt on a circular orbit,
    detic_lon for a near-equatorial pass during init_stats) the computed
    scale ≈ 0 causes either collapse to loc (mean) or numerical blow-up —
    producing nonsense lat/lon/alt values in the telemetry JSON and therefore
    in the visualisation ground track and coverage heatmap.

    Fix: `_gym_env` is the EarthGymEnv instance, created at module level above
    and accessible directly.  No unwrapping required.
    """
    raw_state: dict = _gym_env.last_raw_state   # always the true server state
    return {feat: float(raw_state[feat])
            for feat in obs_features
            if feat in raw_state}

for i, tensordict_data in enumerate(collector):
    for _ in range(num_epochs):
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata    = replay_buffer.sample(sub_batch_size)
            loss_vals  = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    batch_reward = tensordict_data["next", "reward"].mean().item()
    logs["reward"].append(batch_reward)
    pbar.update(tensordict_data.numel())

    # ── Log telemetry record (one per batch) ──────────────────────────────
    # Decodes the last observation in the batch back to raw feature values
    # so the ground track and attitude state are human-readable.
    telemetry.log_step(
        step=i,
        obs=_obs_to_dict(tensordict_data),
        raw_state=_gym_env.last_raw_state,
        reward=batch_reward,
    )
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} "
        f"(init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"

    if i % 10 == 0:
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            eval_rollout = env.rollout(100, policy_module)

            ## MODIFICATION 8 — Reduce rollout horizon for Earth-Gym
            #
            # In main.py:
            #     eval_rollout = env.rollout(1000, policy_module)
            #
            # 1000 steps × delta_time (≈ 5553 s) = ~64 simulated days per eval.
            # Use 100 steps instead (≈ 6.4 simulated days), which is both
            # faster and fits within the default 7-hour scenario window.

            logs["eval reward"].append(
                eval_rollout["next", "reward"].mean().item()
            )
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(
                eval_rollout["step_count"].max().item()
            )
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout

            ## MODIFICATION 9 — Video recording is not applicable to Earth-Gym
            #
            # In main.py:
            #     if i % 100 == 0:
            #         video_env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
            #         video_env = RecordVideo(...)
            #         ...
            #
            # Earth-Gym has no render_mode / video output.
            # Replace with a checkpoint save and a ground-track log instead.

            # ── Tag the last telemetry record with the eval reward ──────────
            telemetry.log_step(
                step=i,
                obs=_obs_to_dict(tensordict_data),
                raw_state=_gym_env.last_raw_state,
                reward=logs["reward"][-1],
                eval_reward=logs["eval reward"][-1],
            )

            if i % 100 == 0:
                ckpt_path = os.path.join(args.out, f"policy_iter_{i:05d}.pt")
                torch.save(
                    {
                        "iter":          i,
                        "policy_state":  policy_module.state_dict(),
                        "value_state":   value_module.state_dict(),
                        "optim_state":   optim.state_dict(),
                        "logs":          dict(logs),
                    },
                    ckpt_path,
                )
                print(f"\n[Checkpoint] saved → {ckpt_path}")

                # ── Option 1: generate matplotlib images ─────────────────
                # Writes ground_track, reward_curve, coverage_heatmap, and
                # composite all_ PNGs to output/images/.
                # Also served by the dashboard at /images/<file>.
                visualize_checkpoint(
                    out_dir=args.out,
                    step=i,
                    logs=dict(logs),
                    cone_angle_deg=args.cone,
                )
                plotter(logs)  # update the training curves after each checkpoint

    pbar.set_description(
        ", ".join([eval_str, cum_reward_str, stepcount_str, lr_str])
    )
    scheduler.step()

plotter(logs)  # save final plot at the end of training




# ─────────────────────────────────────────────────────────────────────────────
# Clean shutdown
# ─────────────────────────────────────────────────────────────────────────────
collector.shutdown()
env.close()          # sends "shutdown" to Earth-Gym server + closes socket
telemetry.close()    # flush and close telemetry.jsonl

if server_proc is not None:
    server_proc.stop()
