# HarborPilot-RL: Obstacle-Count Experiment

**Date:** 2026-10-27  
**Purpose:** Measure how the number of active obstacles affects learned docking performance before introducing wind/current disturbances.

## Experimental setup

- Conditions: 0, 1, and 2 active obstacles.
- Wind and current disturbances: disabled.
- Each condition uses a DQN trained for 1,000 episodes, max 320 steps/episode, seed 17, with no curriculum and random initialization.
- Shared DQN configuration: state dimension 16, 9 discrete actions, gamma 0.99, learning rate 0.001, batch size 32, replay capacity 50,000, warm-up 500 steps, target network update every 1,000 gradient updates, epsilon from 1.0 to 0.05 over 80,000 environment steps, hidden width 128.
- Shared environment configuration: observation version 3, hybrid reward version 3, obstacle-risk distance 100, risk weight 4.0, dock-control weight 1.0, max horizon 320.
- Evaluation: 200 episodes per condition, shared episode seeds 150000–150199. The DQN and scripted baseline each see the same seeds for each obstacle count.
- The one-obstacle model is the already trained checkpoint from `phase2_hybrid_reward_v3_clean`, whose metadata matches the same training configuration. Zero- and two-obstacle agents were trained for this study.

## Results

| Obstacles | Policy | Docked | Success | Collisions | Timeouts | Mean episode steps | Mean final distance |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | DQN | 200/200 | 100.0% | 0/200 | 0/200 | 167.6 | 18.0 |
| 0 | Scripted baseline | 200/200 | 100.0% | 0/200 | 0/200 | 260.1 | 19.7 |
| 1 | DQN | 191/200 | 95.5% | 0/200 | 9/200 | 222.9 | 13.8 |
| 1 | Scripted baseline | 200/200 | 100.0% | 0/200 | 0/200 | 260.1 | 19.7 |
| 2 | DQN | 0/200 | 0.0% | 86/200 | 114/200 | 215.3 | 121.7 |
| 2 | Scripted baseline | 200/200 | 100.0% | 0/200 | 0/200 | 260.1 | 19.7 |

## Failure details

- **0 obstacles:** DQN docked in all 200 episodes; no collisions or timeouts.
- **1 obstacle:** DQN docked in 191 episodes (95.5%); the remaining 9 timed out. It had no collisions or boundary exits.
- **2 obstacles:** DQN did not dock in any of the 200 episodes. It had 86 boundary exits (reported by the environment as collision/terminal failures) and 114 timeouts. There were no direct obstacle-contact events. The agent therefore failed to navigate/return safely in the two-obstacle condition rather than mostly striking the obstacles themselves.
- **Scripted baseline:** docked in all 200 episodes for all three conditions. It follows a fixed waypoint route; it is a hand-coded baseline, not a learned policy.

## Interpretation

Under this training budget and single training seed, DQN performance degrades sharply when trained from scratch with two active obstacles. The one-obstacle checkpoint generalizes well to the held-out seeds, while the two-obstacle checkpoint does not. This suggests the current policy/training setup has not learned a robust route for the two-obstacle layout.

This is an exploratory comparison, not a definitive claim that obstacle count alone caused the degradation: there is only one independently trained DQN per obstacle condition, and the two-obstacle task is a distinct learning problem. A follow-up could repeat the training with additional seeds, but first we should inspect the two-obstacle trajectories and training logs, as planned. We have not introduced disturbances, changed the reward, or changed the DQN algorithm in this experiment.

**Reward comparison caution:** mean return is not compared across different obstacle counts because the hybrid reward applies a proximity penalty for each active obstacle. Success rate, failure cause, episode length and final distance are more interpretable across these conditions.

## Files

- `combined_obstacle_count_summary.csv`: aggregated evaluation metrics.
- `obstacles_0/evaluation_200/evaluation_results.csv`, `obstacles_1/evaluation_200/evaluation_results.csv`, `obstacles_2/evaluation_200/evaluation_results.csv`: per-episode results.
- `obstacle_count_success.png`: safe docking success by obstacle count.
- `dqn_failure_outcomes.png`: DQN episode outcomes by obstacle count.
- Each trained condition also includes training logs, validation logs, learning curves and checkpoints.
