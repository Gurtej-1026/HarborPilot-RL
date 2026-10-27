# HarborPilot-RL

**Autonomous Ship Docking Under Wind and Currents**

HarborPilot-RL is a from-scratch reinforcement-learning mini-project. A ship learns to approach and dock at a designated berth in a custom 2D harbour simulator with circular obstacles, changing wind, and steady water currents.

> **Scope note:** This first version uses a simplified point-mass/heading model for controlled experiments. It is not a physically validated maritime simulator and is not intended for real-world vessel control.

## Research question

Can a DQN agent learn a docking policy that achieves a higher safe-docking rate, fewer collisions, and a better reward than a simple scripted controller under varied wind and current conditions?

## Current environment design

- **State (16 floats):** relative dock position; sine/cosine of heading; normalized speed and yaw rate; current and wind components; ship-frame forward/lateral offsets to both obstacles (validated observation version 3, with both offsets normalized by the world diagonal); active-obstacle stage; and time remaining. Experimental version 4 uses a separate lateral scale. Versions 1 and 2 remain available for comparison and older checkpoints.
- **Actions (9 discrete controls):** combinations of reverse/neutral/forward throttle and left/neutral/right rudder.
- **Reward:** progress toward the dock; a small fuel-use proxy; penalties for lateral deviation, poor alignment, and excessive speed near the berth; +120 for safe docking, -150 for collision/out-of-bounds, and -12 for timeout. Training defaults to reward version 3, which combines potential-based obstacle shaping with a direct proximity penalty after promising one-obstacle tests. Version 1 is the potential-shaping-only baseline; version 2 tests the direct penalty alone.
- **Disturbances:** episode-level current and correlated per-step wind changes. The default training script uses curriculum learning: no obstacles, then one, then both; validation always uses the full harbour. Exploration restarts and replay memory is cleared at stage transitions so stale samples from easier stages do not dominate learning.
- **Terminal success:** close to the dock, aligned with the berth, moving slowly, and with a low yaw rate.
- **Baseline:** a transparent scripted controller used for evaluation only. The learned DQN policy is the core project.

Full formulation and known simplifications are documented in [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md).

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick environment smoke test

From the repository root:

```bash
python - <<'PY'
from harborpilot_rl.environment import HarborPilotEnv

env = HarborPilotEnv(seed=7)
state, info = env.reset()
print("state shape:", state.shape, "initial distance:", round(info["distance"], 2))
for _ in range(5):
    state, reward, terminated, truncated, info = env.step(4)  # neutral controls
    print("reward:", round(reward, 3), "position:", info["position"])
    if terminated or truncated:
        break
PY
```

## Train the RL agent

A short run is useful to verify dependencies and the training loop; it is **not** sufficient evidence of a well-trained policy:

```bash
python -m harborpilot_rl.train --episodes 10 --output-dir outputs/smoke
```

The default curriculum allocates the first 20% of episodes to an open harbour, the next 30% to one obstacle, and the remaining episodes to both obstacles. Validation always tests on the full environment. For a full experiment, start with:

```bash
python -m harborpilot_rl.train --episodes 1000 --seed 7 --output-dir outputs/experiment_01
```

Training creates a CSV log, learning curves, and DQN checkpoints. Training outcomes can vary by seed, so we must not claim success before inspecting the evaluation results.

## Evaluate against the baseline

After training:

```bash
python -m harborpilot_rl.evaluate \
  --checkpoint outputs/experiment_01/harborpilot_dqn_best.pt \
  --episodes 50 \
  --seed 123 \
  --output-dir outputs/experiment_01/evaluation
```

Evaluation writes per-episode results, a baseline comparison figure, and representative trajectory plots.

## Two-obstacle risk-weight experiments (2026-10-27)

The development sweep compared obstacle-risk weights 4.0, 1.0, and 2.0 with two obstacles, wind/current disabled, observation v3, hybrid reward v3, a 320-step horizon, 1,000 episodes, training seed 17, and epsilon decay over 80,000 environment steps. The same development episode seeds (150000–150199) were used during tuning, so these are development results, not final test results.

- Weight 4.0: DQN docked in 0/200 episodes; 86 boundary exits and 114 timeouts.
- Weight 1.0: DQN docked in 149/200 episodes (74.5%); the remaining 51 episodes ended in obstacle collisions.
- Weight 2.0: DQN docked in 199/200 episodes (99.5%); the remaining episode ended in an obstacle collision.
- The scripted baseline docked in 200/200 episodes under all three configurations.

After tuning, the selected weight-2.0 checkpoint was evaluated once on episode seeds 300000–300199. It achieved 198/200 docks for training seed 17, but independently trained seeds 29 and 41 achieved only 14/200 and 52/200. Therefore, the strong seed-17 result does not establish that training is reliable across seeds. These evaluation seeds are consumed and must not be reused. See `docs/INDEPENDENT_VALIDATION_REPORT.md`, `docs/CHECKPOINT_SELECTION_AUDIT.md`, `docs/TRAINING_STABILITY_AUDIT.md`, and `docs/SEED_USAGE_LEDGER.md`.

## Tests

```bash
python -m pip install pytest
python -m pytest -q
```

## Main files

```text
harborpilot_rl/
  environment.py   Custom harbour simulator and trajectory plotting
  agent.py         DQN Q-network, replay buffer, target network, checkpointing
  baseline.py      Non-learning scripted comparison policy
  train.py         Training loop, logging, model saves, learning curves
  evaluate.py      Evaluation and baseline comparison
 docs/
  PROJECT_SPEC.md  State/action/reward design and experimental plan
 tests/
  test_environment.py
 outputs/          Generated models, logs, plots (not committed by default)
```

## Experimental discipline

1. Keep a fixed set of evaluation seeds separate from training seeds.
2. Compare DQN and baseline under the same evaluation seeds.
3. Report success rate, collision rate, return, episode length, and final distance.
4. Run multiple training seeds before drawing a strong conclusion.
5. Keep the implementation history and document major design decisions in Git commits.

## Academic integrity

The simulator, agent, and baseline in this repository are custom implementations for this project. Cite any external paper, library, or reference that informs the final report. Do not copy an external repository into the project.


Reward variants: version 1 uses potential-based obstacle shaping; version 2 tests a direct per-step proximity penalty; version 3 combines both and is the current training default. Use `--reward-version 1|2|3` to select one.

To reproduce the current one-obstacle diagnostic configuration (not the final full-harbour benchmark):

```bash
python -m harborpilot_rl.train --episodes 1000 --seed 17 --max-steps 320 --output-dir outputs/phase2_hybrid_reward_v3 --fixed-obstacles 1 --no-disturbances --reward-version 3 --obstacle-risk-weight 4.0 --epsilon-decay-steps 80000 --observation-version 3
python -m harborpilot_rl.evaluate --checkpoint outputs/phase2_hybrid_reward_v3/harborpilot_dqn_best.pt --episodes 200 --seed 100000 --output-dir outputs/phase2_hybrid_reward_v3/evaluation_200
```


`--dock-control-weight` scales the near-dock heading-alignment and speed penalties. The default `1.0` preserves the original scale; we are evaluating larger values for disturbed docking where failures have been caused by overshooting the berth.

## Canonical configuration for independent validation

The canonical project root is the directory containing this README. The current development candidate is stored at `models/harborpilot_dqn_two_obstacle_risk2_seed17.pt` (the original experiment checkpoint remains preserved under `outputs/`) and described in `docs/CANONICAL_PROJECT.md` and `configs/frozen_candidate_two_obstacle_risk2.json`. Follow `docs/REPRODUCIBILITY.md` for exact setup, test, smoke-evaluation, and training commands. It uses two obstacles, no wind/current, observation version 3, hybrid reward version 3, obstacle-risk weight 2.0, 1,000 episodes, training seed 17, and 80,000-step epsilon decay. This is a frozen development candidate. The risk-weight tuning set (episode seeds 150000–150199) is development data. Episode seeds 300000–300199 were subsequently used once for independent evaluation of training seeds 17, 29, and 41 and must not be reused for tuning. Episode seeds 10020–10099 were later used once to audit checkpoint-selection reliability and are now consumed development-validation data. See `docs/INDEPENDENT_VALIDATION_REPORT.md`, `docs/CHECKPOINT_SELECTION_AUDIT.md`, `docs/TRAINING_STABILITY_AUDIT.md`, and `docs/SEED_USAGE_LEDGER.md`. The final held-out benchmark was then run once on episode seeds 400000–400199: the frozen seed-17 DQN docked safely in 199/200 episodes (99.5%; one obstacle collision), while the scripted baseline docked in 200/200. Wind/current disturbances were disabled for that benchmark. These final-test seeds are consumed and must not be reused. See `docs/FINAL_TEST_REPORT.md` and the files in `results/`.
