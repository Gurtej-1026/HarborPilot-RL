# Reproducibility and runbook

This document defines the canonical source root and exact commands for tests, a short smoke evaluation, a fresh training reproduction, and future final evaluation. Run commands from the directory containing `README.md`.

## Canonical candidate

- Model: `models/harborpilot_dqn_two_obstacle_risk2_seed17.pt`
- Configuration: `configs/frozen_candidate_two_obstacle_risk2.json`
- Environment: two obstacles, no wind/current, maximum 320 steps
- Observation version: 3
- Reward version: 3 (hybrid shaping + obstacle-risk cost)
- Obstacle-risk distance: 100
- Obstacle-risk weight: 2.0
- Dock-control weight: 1.0
- DQN: 16 state features, 9 actions, gamma 0.99, learning rate 0.001, batch size 32, replay capacity 50,000, warmup 500 steps, target update every 1,000 gradient steps, hidden width 128
- Training provenance: candidate was selected from training seed 17; 1,000 episodes; fixed two-obstacle training; disturbances disabled; epsilon decay 80,000 environment steps; learn every 4 environment steps.

This is the selected and frozen submission candidate. Its final held-out result is recorded in `FINAL_TEST_REPORT.md`. Independent training seeds 29 and 41 under the same nominal configuration performed much worse, so training remains seed-sensitive. Historical source provenance is also limited: the original byte-for-byte source that generated all version-3 checkpoints was not found; see `SOURCE_RECOVERY.md`.

## 1. Environment setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest
```

## 2. Verify source and tests

```bash
python -m pytest -q
python -m harborpilot_rl.train --help
python -m harborpilot_rl.evaluate --help
```

The submission package was checked with `python -m pytest -q` (16 tests passed). Use `python -m pytest`, not the bare `pytest` entrypoint, because the latter did not place the project root on `sys.path` in the audit environment.

## 3. Smoke-evaluate the packaged checkpoint

The following short check verifies checkpoint loading and evaluation code; it is not a performance benchmark.

```bash
python -m harborpilot_rl.evaluate \
  --checkpoint models/harborpilot_dqn_two_obstacle_risk2_seed17.pt \
  --episodes 3 \
  --seed 930000 \
  --output-dir outputs/reproducibility_smoke
```

The evaluator derives episode seeds by adding 50,000 to the CLI seed, so this smoke check uses episodes 980000–980002. Its representative trajectory uses seed 990000. These are smoke-only seeds, not final-test data.

## 4. Fresh training reproduction command

This command reproduces the candidate's nominal configuration as a new run. It will **not** recreate the historical checkpoint byte-for-byte because the historical source implementation was not recovered exactly, and DQN training can vary. It creates a separate directory and does not overwrite the packaged candidate.

```bash
python -m harborpilot_rl.train \
  --episodes 1000 \
  --seed 17 \
  --max-steps 320 \
  --output-dir outputs/reproduction_risk2_seed17 \
  --no-curriculum \
  --fixed-obstacles 2 \
  --no-disturbances \
  --observation-version 3 \
  --reward-version 3 \
  --obstacle-risk-distance 100 \
  --obstacle-risk-weight 2.0 \
  --dock-control-weight 1.0 \
  --epsilon-decay-steps 80000 \
  --learn-every 4
```

Do not run this merely as a smoke test; it is a full 1,000-episode training experiment.

## 5. Development evaluation seeds (already consumed)

To reproduce the earlier development evaluation (not an independent final test):

```bash
python -m harborpilot_rl.evaluate \
  --checkpoint models/harborpilot_dqn_two_obstacle_risk2_seed17.pt \
  --episodes 200 \
  --seed 100000 \
  --output-dir outputs/reproduction_development_seed17
```

This generates episode seeds 150000–150199, which were used for risk-weight tuning. Do not report this as an untouched final evaluation.

## 6. Final held-out evaluation (already consumed)

The final benchmark was run once using:

```bash
python -m harborpilot_rl.evaluate \
  --checkpoint models/harborpilot_dqn_two_obstacle_risk2_seed17.pt \
  --episodes 200 \
  --seed 350000 \
  --output-dir outputs/final_test_seed400000
```

The evaluator maps this to episode seeds 400000–400199. The DQN achieved 199/200 safe docks (99.5%); the scripted baseline achieved 200/200 (100%). The DQN had one obstacle collision and no boundary exits. Do not reuse these seeds for tuning or a second confirmatory evaluation. See `docs/FINAL_TEST_REPORT.md` and the included descriptive-report PDF.

## Experiment hygiene

- Preserve the original experiment directories and candidate checkpoint.
- Make only one intentional change per experiment and record it in a new output directory.
- Never tune hyperparameters or select a candidate using a final-test seed set.
- Report performance across independent training seeds, including weak runs.
- A successful checkpoint does not by itself establish that the training procedure is reliable.
