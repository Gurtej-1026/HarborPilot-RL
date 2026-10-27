# Final held-out test report

**Date:** 2026-10-27  
**Candidate:** frozen DQN checkpoint trained with seed 17  
**Environment:** two fixed obstacles; wind/current disturbances disabled; observation v3; reward v3; 320-step horizon.

## Protocol

The candidate was frozen before evaluation. The evaluator was run once with `--episodes 200 --seed 350000`, which maps to episode seeds `400000–400199`. DQN and the scripted baseline were evaluated on the same episodes. This range is now consumed and must not be used for tuning or another confirmatory claim.

## Results

| Metric | DQN | Scripted baseline |
|---|---:|---:|
| Safe docking | 199/200 (99.5%) | 200/200 (100%) |
| Collision | 1/200 (0.5%) | 0/200 (0%) |
| Boundary exits | 0/200 | 0/200 |
| Mean episodic return | 139.29 | -17.97 |
| Mean discounted return | 37.35 | -13.43 |
| Mean steps | 231.78 | 260.30 |
| Mean final distance | 15.01 | 19.73 |

The DQN performed strongly, but did not outperform the baseline in safe-docking success rate. It achieved higher mean return and fewer steps on average. The final benchmark is undisturbed and uses a fixed layout. Earlier evaluation on the same nominal configuration found large training-seed sensitivity: seeds 29 and 41 achieved 7% and 26% success on episode seeds `300000–300199`, while seed 17 achieved 99%. The selected checkpoint result should not be presented as evidence that training is robust across seeds.

## Artifacts

- `outputs/final_test_seed400000/evaluation_results.csv` — per-episode records for both methods.
- `outputs/final_test_seed400000/baseline_comparison.png` — metric comparison chart.
- `outputs/final_test_seed400000/dqn_trajectory.png` and `baseline_trajectory.png` — representative trajectories.
