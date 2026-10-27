# HarborPilot-RL: Two-obstacle risk-weight comparison

## Purpose

Test whether changing only the direct obstacle-risk penalty weight affects DQN performance in the two-obstacle, no-disturbance environment.

## Fixed settings

- Obstacles: 2
- Wind/current disturbances: disabled
- Training episodes: 1,000 for each learned agent
- Training seed: 17
- Episode horizon: 320 steps
- Observation version: 3 (ship-frame obstacle features)
- Reward version: 3 (hybrid potential shaping + direct risk penalty)
- Obstacle-risk distance: 100
- Dock-control weight: 1.0
- Discount factor: 0.99
- DQN hidden dimension: 128
- Epsilon decay: 80,000 environment steps
- Test seeds: 150000–150199, shared by every weight and baseline

The 4.0 and 1.0 rows reuse prior experiments; weight 2.0 is the new experiment. Each newly trained agent starts from fresh weights. The selected checkpoint is the best validation checkpoint, not automatically the final training checkpoint.

## Results

| Risk weight | DQN safe docks | Success | Obstacle collisions | Boundary exits | Timeouts | Mean steps in successful episodes |
|---:|---:|---:|---:|---:|---:|---:|
| 4.0 | 0/200 | 0.0% | 0 | 86 | 114 | n/a (no successful episodes) |
| 1.0 | 149/200 | 74.5% | 51 | 0 | 0 | 208.0 |
| 2.0 | 199/200 | 99.5% | 1 | 0 | 0 | 232.9 |

The scripted baseline docked safely in 200/200 episodes for each evaluation condition.

## Interpretation

Weight 2.0 is the strongest of the three tested settings on this held-out seed set: 199/200 safe docks (99.5%) with one obstacle collision. Weight 1.0 had 149/200 safe docks (74.5%) and 51 obstacle collisions. Weight 4.0 had no successful docks, with 86 boundary exits and 114 timeouts.

This supports the hypothesis that the risk-weight setting is important in this environment. It does **not** prove that 2.0 is globally optimal: we tested one training seed per weight, so conclusions about training variability remain limited. The baseline is still 100% safe docking on these test seeds.

Returns are not compared across weights as the reward function itself changes with this parameter. The comparison emphasizes task outcomes.

## Lecture connection

The reward function and discounted return are part of the MDP/RL formulation in Lectures 20–23. The DQN Q-function approximates action values as covered in Lecture 24. Selecting a risk weight and comparing outcomes under a controlled experiment are project-specific engineering and evaluation choices.

## Reproducibility

The per-episode CSVs in the experiment folders include each method, episode, test seed, return, terminal outcome, steps and final docking measurements. This summary was derived from those files and checks that all three comparisons used exactly seeds 150000–150199.
