# Development Experiment Notes (Preliminary)

These are debugging experiments, not final project results. The simulator, reward and training schedule changed between runs; only report a result as final after reproducing it with a frozen configuration and a held-out test-seed set.

## What the early runs tell us

- The scripted waypoint baseline can dock reliably in this simplified environment, but it is handcrafted and is used only as a comparison policy.
- DQN learns the open-harbour docking task much more readily than the obstacle task.
- Adding obstacles creates a pronounced generalization challenge: a policy trained with one obstacle can approach or avoid the first hazard, but it may not adjust its route when the second hazard is introduced.
- Continued training can regress; the best full-harbour validation checkpoint should be selected rather than assuming the final checkpoint is the best.

## One reproducible preliminary comparison

For the best checkpoint produced during an 800-episode exploratory curriculum run, an evaluation over 50 seeds (seed base 1234) reported:

| Metric | DQN best checkpoint | Scripted baseline |
|---|---:|---:|
| Safe-docking success rate | 20% | 52% |
| Collision/out-of-bounds rate | 78% | 10% |
| Mean return | 41.12 | 305.76 |
| Mean steps | 107.2 | 262.0 |
| Mean final distance | 123.9 | 32.6 |

This result is **not good enough to claim that the RL agent outperforms the baseline**. It establishes a baseline for further development. The full-harbour evaluation must be repeated after the next training changes, with identical evaluation seeds across methods and multiple independent training seeds.

## Recommended next improvements

1. Keep epsilon exploration active after each curriculum transition; reset exploration and replay at stage changes so the new hazards can be learned.
2. Run longer full-obstacle training and monitor validation on held-out seeds.
3. Compare a direct-goal controller and the waypoint controller, clearly describing what each baseline can and cannot do.
4. Run at least three training seeds, report mean and standard deviation, and tune reward parameters only on validation seeds.
5. Consider randomizing the obstacle layout after the fixed-layout policy works, to test generalization rather than memorization.

## Latest curriculum-reset trial (incomplete)

A later run using the current curriculum reset logic was interrupted at episode 513, so it is not a complete 800-episode run. The best saved checkpoint evaluated on 20 held-out seeds (seed base 2345) achieved 10% safe-docking success and 80% collision/out-of-bounds, compared with 60% success and 10% collision/out-of-bounds for the scripted baseline on the same seeds. Because this is an incomplete, small-sample development evaluation, it should be treated as a warning that the RL policy still needs work, not as a final result.

## Additional fixed-one-obstacle experiments (historical artifacts)

The saved outputs also contain later diagnostic experiments. These are recorded here for traceability, but they are **not final project claims**: the exact source revision for observation v3 + hybrid reward v3 was missing during the source audit, so the compatibility implementation was reconstructed from available backups, metadata, docs, and tests. Re-evaluate before using these scores in a final report.

| Configuration | Disturbances | Episodes in stored test | DQN success | Scripted baseline success | Main observation |
|---|---:|---:|---:|---:|---|
| Observation v3 + reward v3, best saved checkpoint | Off | 200 each | 191/200 (95.5%) | 200/200 (100%) | DQN had 0 collisions; 9 timeouts |
| Observation v3 + reward v3, trained from scratch | On | 200 each | 44/200 (22%) | 94/200 (47%) | DQN often left harbour bounds |
| Observation v3 + reward v3, warm-started from no-disturbance checkpoint | On | 200 each | 36/200 (18%) | 94/200 (47%) | Warm start reduced collision/out-of-bounds rate relative to scratch training, but success remained low |
| Observation v4 + reward v1 | Off | 50 each | 8/50 (16%) | 50/50 (100%) | Feature scaling experiment regressed versus v3 |

The stored checkpoints preserve their key environment parameters in metadata. After source recovery, 15 `harborpilot_dqn_best.pt` files were smoke-tested by loading their network, reconstructing the saved environment configuration, and selecting one action. This is a compatibility check, not a full reproduction of the historical evaluation tables.


## Two-obstacle obstacle-risk weight experiment (2026-10-27)

**Hypothesis:** The direct obstacle-risk penalty at weight 4.0 accumulates over two active obstacles and may make useful routes unattractive.

**Controlled change:** The new run changed only `obstacle_risk_weight` from 4.0 to 1.0. It used a fresh DQN, 1,000 episodes, seed 17, fixed two-obstacle environment, no disturbances, observation v3, reward v3, risk distance 100, dock-control weight 1.0, and epsilon decay 80,000 environment steps. It was evaluated on the same 200 episode seeds as the re-evaluated weight-4.0 checkpoint.

**Results:** risk 4.0 had 0/200 docks, 86 boundary exits and 114 timeouts. Risk 1.0 had 149/200 docks (74.5%) and 51 obstacle collisions, with no boundary exits or timeouts. The scripted baseline docked in 200/200 episodes for both evaluation configurations.

**Interpretation:** Reducing the weight substantially improves docking but trades boundary exits/timeouts for direct obstacle collisions. This supports further testing of an intermediate weight, but does not prove that 1.0 is optimal. One training seed per learned policy limits the strength of the conclusion. Returns across weights are not directly comparable because changing the weight changes the reward function.

Detailed report: `outputs/two_obstacle_riskweight1_20261027_full/RISK_WEIGHT_EXPERIMENT.md`.
