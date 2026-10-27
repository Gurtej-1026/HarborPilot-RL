# Checkpoint-selection reliability audit

**Date:** 2026-10-27  
**Status:** Complete; evaluation only, no training or checkpoint changes.

## Question

Does selecting a checkpoint using 20 fixed validation episodes provide a useful estimate of how that checkpoint behaves on unseen episodes?

## Existing procedure and evidence

`harborpilot_rl/train.py` validates every 100 training episodes on the same episode seeds, `10000-10019`. It selects checkpoints lexicographically by docking success, collision rate, mean discounted return, and final distance. Because this compares up to ten checkpoints on the same 20 episodes, the selected validation score may be optimistic.

The previously saved validation logs and separate 200-episode evaluation reported:

| Training seed | Best score on 20 selection episodes | Result on episode seeds 300000-300199 |
|---:|---:|---:|
| 17 | 20/20 (100%) at episode 800 | 198/200 (99%) |
| 29 | 1/20 (5%) at episode 900 | 14/200 (7%) |
| 41 | 4/20 (20%) at episode 700 | 52/200 (26%) |

The 200-episode range is already consumed, so it was not reused in this audit.

## Additional held-out validation

The three existing selected checkpoints were evaluated without training on a non-overlapping set of 80 episodes with seeds `10020-10099`. Those seeds were not part of the checkpoint-selection loop in the existing run logs. DQN and the scripted baseline were evaluated on matching episode seeds. The evaluation was deterministic inference (`explore=False`); environment settings were loaded from each checkpoint's saved metadata.

| Training seed | DQN docks | DQN success rate | Exact 95% binomial confidence interval | DQN collisions | Scripted baseline docks |
|---:|---:|---:|---:|---:|---:|
| 17 | 80/80 | 100.0% | 95.5%-100.0% | 0 | 80/80 |
| 29 | 9/80 | 11.25% | 5.3%-20.3% | 46 | 80/80 |
| 41 | 24/80 | 30.0% | 20.3%-41.3% | 56 | 80/80 |

Confidence intervals use the exact Clopper-Pearson binomial interval and describe episode-level uncertainty for these particular trained checkpoints; they do not quantify uncertainty over new training runs.

Raw episode records are in `outputs/checkpoint_selection_audit_seed10020_10099.csv`.

## Conclusion

On this small set of three already-trained checkpoints, the ranking suggested by the 20-episode validation scores carried over to the 80 non-overlapping episodes: seed 17 remained strong, and seeds 29 and 41 remained weak. This is evidence against severe overfitting to the 20 checkpoint-selection episodes in these particular runs. It is not proof that 20 episodes are generally sufficient, because only three selected checkpoints were examined and validation was repeated across training.

The key unresolved problem remains **training-seed sensitivity**, not an observed checkpoint-ranking reversal. The stronger seed-17 result does not imply reliable learning across random initializations. Do not change the DQN update, reward, or frozen checkpoint on the basis of this audit alone. Do not use episode seeds `10020-10099` again for tuning or confirmation; they are now consumed development-validation data. Keep `400000-400199` reserved for a future final test after the candidate and training procedure are frozen.

No training was run, no checkpoint weights were modified, and the reserved final-test range was not touched.
