# Training-stability audit

**Date:** 2026-10-27  
**Scope:** Existing logs and checkpoints only; no training, hyperparameter tuning, or model-weight changes were performed.

## Questions

1. Do the three training runs use the same recorded configuration?
2. At what point does training behaviour diverge?
3. Do the logs identify a specific software defect or justify changing a hyperparameter?

## Configuration cross-check

The metadata saved inside the selected checkpoints for training seeds 17, 29, and 41 agree on the main DQN and environment settings: 16 state features, 9 actions, gamma 0.99, learning rate 0.001, batch size 32, replay capacity 50,000, warm-up 500 steps, target sync every 1,000 gradient updates, epsilon decaying from 1.0 to 0.05 over 80,000 environment steps, hidden width 128, two obstacles, disturbances disabled, observation version 3, reward version 3, obstacle-risk weight 2.0, dock-control weight 1.0, and 320 maximum steps. Each run records its own training seed and no warm-start checkpoint.

This cross-check makes an obvious saved-configuration mismatch an unlikely explanation for the result gap. It does not prove that the historical training source for seed 17 was byte-for-byte identical; the source-recovery caveat in `SOURCE_RECOVERY.md` still applies.

## Training-log evidence

Cause names below come from the per-episode training logs. The training policy acts with epsilon-greedy exploration, so its episode success rate is not directly equivalent to greedy checkpoint evaluation.

| Training seed | Docks in 1,000 training episodes | Main failure causes in training | Selected checkpoint | Key trajectory pattern |
|---:|---:|---|---:|---|
| 17 | 200 (20.0%) | 342 obstacle collisions, 262 boundary exits, 196 timeouts | Episode 800 | Training docking rose to 54% in episodes 701–800, then fell to 2% in episodes 901–1000. Greedy validation fell from 100% at episode 800 to 0% at episode 900. |
| 29 | 4 (0.4%) | 362 obstacle collisions, 365 boundary exits, 269 timeouts | Episode 900 | Docking was rare throughout; just 1% of episodes 801–900 and 2% of episodes 901–1000 docked. |
| 41 | 73 (7.3%) | 538 obstacle collisions, 258 boundary exits, 131 timeouts | Episode 700 | Docking rose to 26% in episodes 701–800, then fell to 9% and 3% over the next two 100-episode blocks; obstacle collisions remained common. |

The checkpoint selector is based on greedy evaluations of the same 20 validation episodes (seeds 10000–10019), prioritizing success rate, lower unsafe-termination rate, discounted return, then final distance. The saved validation records show why checkpointing matters: seed 17's selected checkpoint at episode 800 was much better than its later policy. For seed 29, the best recorded validation success was 5%; for seed 41 it was 20%.

## Additional held-out checkpoint evidence

The separate checkpoint-selection audit evaluated the already-selected checkpoints on seeds 10020–10099, which were not used by the 20-episode checkpoint-selection loop. Results were:

| Training seed | Docks on 80 held-out episodes | Failure causes |
|---:|---:|---|
| 17 | 80/80 (100%) | No failures |
| 29 | 9/80 (11.25%) | 44 boundary exits, 2 obstacle collisions, 25 timeouts |
| 41 | 24/80 (30%) | 56 obstacle collisions |

The scripted baseline docked in all 80 episodes for each training-seed comparison. This supports the conclusion that the weakness is policy-dependent rather than an unavoidable property of these particular episodes. The held-out set is now consumed development-validation data and must not be reused for tuning or confirmation.

**Terminology note:** In the environment's `info` dictionary and training CSV, the boolean `collision` is set for both an obstacle collision and leaving harbour bounds. The string `cause` distinguishes `obstacle_collision` from `left_harbour_bounds`; summaries should preserve that distinction.

## What the evidence does—and does not—show

- **Strong evidence:** DQN learning is highly sensitive to the training seed under the recorded configuration. The failure mode also differs: seed 29 frequently exits bounds; seed 41 frequently hits obstacles; seed 17 learns a strong policy but later training updates degrade the policy, which the best-checkpoint mechanism avoids packaging as the candidate.
- **Not diagnostic by itself:** Mean training loss falls late in all three runs, including the weak runs. A lower SmoothL1 TD loss does not guarantee successful docking, and these CSVs do not contain Q-value statistics, gradient norms by update, action distributions, or replay composition needed to identify a more specific mechanism.
- **No demonstrated single-line bug:** Inspection of the DQN update, terminal masking, target-network synchronization, and checkpoint score ordering did not reveal an obvious defect that explains the different outcomes by itself. This is not a proof that the full implementation is defect-free.
- **No hyperparameter change is justified by these logs alone:** Lower learning rate, slower epsilon decay, or a different target-update interval are plausible experimental hypotheses, not conclusions. Picking one without a controlled comparison would be guesswork and could compromise the current working candidate.

## Decision and next steps

1. Preserve the seed-17 best checkpoint and current source unchanged for now. Do not use the last-episode checkpoint simply because it is newer.
2. In the report, state the success of the selected seed-17 policy separately from training repeatability. The current evidence supports a strong selected policy in the tested fixed, disturbance-free harbour; it does not support claiming that the DQN reliably learns that policy from arbitrary seeds.
3. Do not use episode seeds 10000–10099 or 300000–300199 for further tuning or confirmation. Seeds 400000–400199 remain reserved for a final evaluation only after the candidate and its evaluation procedure are formally frozen.
4. If there is time for one more experiment, pre-register one hypothesis and change one training factor only, compare across multiple training seeds using development validation data, and keep the current checkpoint as the rollback candidate. Do not launch another broad reward-weight sweep.

## Reproducibility boundary

This audit was computed from the saved training and validation CSVs, checkpoint metadata, and the existing held-out checkpoint-selection audit. No new episode was simulated. No training was started, no model checkpoint was modified, and no reserved final-test seed was used.
