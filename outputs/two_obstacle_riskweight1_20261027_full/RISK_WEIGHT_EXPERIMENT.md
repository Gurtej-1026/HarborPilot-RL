# HarborPilot-RL: Two-obstacle risk-weight experiment

**Date:** 2026-10-27  
**Question:** Does reducing the direct obstacle-risk penalty weight from 4.0 to 1.0 make successful docking more learnable in the fixed two-obstacle environment?

## Controlled setup

- Active obstacles: 2
- Wind and current: disabled
- Episodes: 1,000 for the new weight-1.0 training run
- Training seed: 17
- Evaluation: same 200 episode seeds (150000–150199) for both saved DQN checkpoints and the scripted baseline
- Fixed settings: max 320 steps, observation v3, reward v3 (hybrid), risk distance 100, dock-control weight 1.0, DQN architecture and learning parameters, epsilon decay 80,000 environment steps, and learning every 4 steps
- Intended difference: `obstacle_risk_weight`, changed from 4.0 to 1.0 for the new training run

A small CLI option `--epsilon-decay-steps` was added in the experiment workspace because the recovered training script did not expose the schedule stored in the historical checkpoint. It controls a DQN configuration value and does not change the learning algorithm. The original weight-4.0 checkpoint was re-evaluated under the recovered source before this comparison; its results reproduced the saved obstacle-count study.

## Held-out results

| Risk weight | Policy | Docking | Obstacle collisions | Boundary exits | Timeouts | Mean steps |
|---:|---|---:|---:|---:|---:|---:|
| 4.0 | DQN | 0/200 (0%) | 0 | 86 | 114 | 215.3 |
| 1.0 | DQN | 149/200 (74.5%) | 51 | 0 | 0 | 169.0 |
| 4.0 | Scripted baseline | 200/200 (100%) | 0 | 0 | 0 | 260.1 |
| 1.0 | Scripted baseline | 200/200 (100%) | 0 | 0 | 0 | 260.1 |

The `collision` field in the existing evaluator counts both obstacle impacts and leaving the harbour bounds as terminal failures. The table separates these using the recorded `cause`, to avoid conflating the two.

## Interpretation

Reducing the penalty weight substantially improved docking: the DQN moved from 0% to 74.5% success on the shared test set. However, its remaining 25.5% failures were all actual obstacle collisions, whereas the weight-4.0 agent mainly left the harbour or timed out. This supports the hypothesis that the stronger risk penalty was discouraging useful routes, but weight 1.0 now under-penalizes hazardous proximity or otherwise fails to learn avoidance reliably enough.

The scripted baseline docked in all 200 episodes under both reward configurations. This shows the task is solvable by the known waypoint controller, but the learned policy remains less reliable.

**Reward-return caveat:** mean returns should not be compared directly between weights because changing the weight changes the reward scale and objective. Success and failure causes are more interpretable for this experiment.

**Limitations:** one training seed was used per learned policy. The result is a useful controlled exploratory comparison, not definitive evidence of the best risk weight. A follow-up could test an intermediate weight (for example 2.0) with the rest of the configuration fixed, then repeat across another training seed if promising.

## Files

- `control_risk4_eval/evaluation_results.csv`: original weight-4.0 checkpoint evaluated under the recovered source.
- `evaluation_200/evaluation_results.csv`: weight-1.0 checkpoint and baseline on the same 200 seeds.
- `risk_weight_comparison.csv`: aggregated metrics.
- `risk_weight_outcomes.png`: DQN outcome breakdown by weight.
- `training_log.csv`, `validation_log.csv`, and checkpoint files: training artifacts for weight 1.0.
