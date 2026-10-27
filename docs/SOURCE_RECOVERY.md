# Source and checkpoint compatibility audit

**Date:** 2026-10-26

## What the audit found

The saved DQN checkpoints include environment metadata (observation version, reward version, disturbance flag, obstacle count, risk parameters, and for some runs the dock-control weight). However, the main `environment.py` source that was present at audit time only accepted observation versions 1–2 and reward versions 1–2. `train.py` did not expose the observation-version option or the dock-control multiplier described in the README. The test suite expected observation versions 1–4, reward version 3, and `dock_control_weight`, so it failed during collection because the corresponding constants were absent.

Available historical pieces were incomplete: `HarborPilot-RL-environment-before-lateral-scale.py` supports observation versions 1–3 and reward versions 1–2, but it does not contain the hybrid reward version 3 or dock-control multiplier. The exact historical source file that produced the v3/reward-v3 checkpoints was not found in the saved project files.

## Compatibility reconstruction

The current implementation has been reconstructed using the available historical source, checkpoint metadata, README/specification, and tests. It supports:

- Observation v1: legacy clipped world-frame obstacle offsets.
- Observation v2: axis-normalized world-frame obstacle offsets.
- Observation v3: ship-frame forward/lateral offsets normalized by the harbour diagonal.
- Observation v4: ship-frame forward offset normalized by harbour width, and lateral offset scaled by 65 units and clipped to [-1, 1].
- Reward v1: potential-based obstacle shaping.
- Reward v2: direct per-step obstacle-risk penalty.
- Reward v3: both potential shaping and direct obstacle-risk penalty.
- `dock_control_weight`: multiplier on near-dock heading and speed penalties; checkpoint metadata is used during evaluation.

**Important provenance limitation:** this is a compatibility reconstruction, not a byte-for-byte recovery of the missing historical source. Existing experimental CSVs and checkpoints have not been rewritten. Before treating historical scores as final report results, rerun evaluation under the reconstructed source and record the source revision/configuration used.

## Validation performed

- Unit tests: 11 passed.
- Training CLI exposes observation versions 1–4, reward versions 1–3, risk settings, and dock-control weight.
- A non-training smoke test loaded all 15 available `harborpilot_dqn_best.pt` checkpoints, rebuilt their environment configuration from metadata, and produced a valid action from each model.
- No training runs were started and no checkpoints were overwritten during this recovery step.

## Preserved files

Pre-repair copies of the README, project specification, experiment notes, DQN agent, environment, training/evaluation scripts, and environment tests were preserved outside the active project tree at `/mnt/data/HarborPilot-RL-pre-repair-snapshot/` in this working session. The snapshot is separate from the recovered project package.
