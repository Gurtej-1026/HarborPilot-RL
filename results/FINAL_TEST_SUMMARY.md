# Final held-out benchmark summary

- Candidate: frozen DQN checkpoint trained with seed 17.
- Environment: two fixed obstacles; wind/current disturbances disabled; observation version 3; reward version 3; 320-step horizon.
- Episode seeds: 400000–400199, evaluated once after the candidate was frozen. Do not reuse these seeds for tuning.

| Metric | DQN | Scripted baseline |
|---|---:|---:|
| Safe docking | 199/200 (99.5%) | 200/200 (100%) |
| Obstacle collisions | 1/200 (0.5%) | 0/200 (0%) |
| Boundary exits | 0/200 | 0/200 |
| Mean episodic return | 139.29 | -17.97 |
| Mean discounted return | 37.35 | -13.43 |
| Mean steps | 231.78 | 260.30 |
| Mean final distance | 15.01 | 19.73 |

The DQN achieved a higher mean return and docked in fewer steps on average, but did not exceed the baseline's docking success rate. Performance varied substantially across training seeds in separate evaluation, so this is evidence for one selected policy, not for robust training.
