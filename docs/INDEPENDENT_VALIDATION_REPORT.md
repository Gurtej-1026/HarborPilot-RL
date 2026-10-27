# Independent multi-seed validation report

**Date:** 2026-10-27  
**Task:** Two-obstacle harbour, wind/current disturbances disabled  
**Candidate setting:** observation v3, reward v3 (hybrid), obstacle-risk distance 100, obstacle-risk weight 2.0, dock-control weight 1.0, 320-step episode horizon.

## Purpose

After selecting obstacle-risk weight 2.0 during development, evaluate whether the strong result reproduces across fresh training seeds. The episode seed set 300000-300199 was reserved before these evaluations and does not appear in earlier saved CSVs.

## Method

- Training episodes: 1,000 per learned policy.
- DQN settings: state dimension 16, action dimension 9, gamma 0.99, learning rate 0.001, batch size 32, replay capacity 50,000, warmup 500 steps, target update every 1,000 steps, epsilon 1.0 to 0.05 over 80,000 environment steps, hidden dimension 128.
- Environment: exactly two obstacles, disturbances disabled, max 320 steps, no curriculum.
- Training seeds: 17, 29, 41.
- Checkpoint selection: fixed validation episode seeds 10000-10019; selection prioritizes success rate, then collision rate, discounted return and final distance.
- Test episode seeds: 300000-300199 for every checkpoint, using the same seeds for DQN and the scripted baseline.
- No training or hyperparameter tuning used the reserved test episodes before evaluation. The test set is now consumed and must not be reused for later tuning.

Seed 17 was the existing candidate selected during risk-weight development; seeds 29 and 41 were fresh training replicates. This distinction matters when interpreting the results.

## Results

| DQN training seed | Best validation success (20 episodes) | DQN test success | DQN obstacle collisions | DQN boundary exits | DQN timeouts | Baseline test success |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100% (episode 800) | 198/200 (99%) | 2 | 0 | 0 | 200/200 (100%) |
| 29 | 5% (episode 900) | 14/200 (7%) | 5 | 145 | 36 | 200/200 (100%) |
| 41 | 20% (episode 700) | 52/200 (26%) | 148 | 0 | 0 | 200/200 (100%) |

The baseline consistently completed the task. DQN performance differed sharply by training seed. Seed 17 generalized well to this independent episode set, while seeds 29 and 41 failed to learn similarly effective policies. The descriptive mean success rate across the three checkpoints is 44%, but it should **not** be presented as an unbiased performance estimate because seed 17 was the previously selected development candidate. The per-seed results and training-seed distinction are the main finding.

## Interpretation

The risk-weight-2.0 policy can solve the two-obstacle task, but the current DQN training process is highly seed-sensitive. The development result of 99.5% on episode seeds 150000-150199 was not itself enough to establish repeatable training. This independent evaluation reveals the gap between finding one strong checkpoint and reliably learning such a policy from different random initializations.

The weak seed-29 and seed-41 runs are not to be discarded. They are valid evidence about training instability. Their logs show that their best validation success rates were only 5% and 20%, respectively, which is consistent with their poor independent test performance.

## Limitations

- One candidate configuration and three training seeds were evaluated.
- The scripted baseline is designed for the fixed, known harbour geometry; it is a reference controller, not a learned policy.
- The environment is a simplified 2D simulator, and disturbances were disabled for this study.
- The historical code that originally produced all version-3 reward checkpoints was not recovered byte-for-byte; the current source is the documented compatibility reconstruction.
- The test episode seed set 300000-300199 has been consumed. It must not be reused to tune or confirm a changed agent. A future final test should use the reserved range 400000-400199 only after a new candidate is frozen.

## Conclusion

Do not move to wind/current disturbances yet. First investigate training instability using only existing training logs and the fixed validation set. If a change is later made, assess it on development/validation data, freeze the revised candidate, then use a fresh final-test seed set.
