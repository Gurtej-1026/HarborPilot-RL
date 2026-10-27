# HarborPilot-RL seed usage ledger

Updated: 2026-10-27

Seed roles are intentionally separated. Never reuse a test episode seed set for reward tuning, policy selection, debugging, or a later confirmatory claim. The final benchmark range 400000–400199 has now been consumed.

| Episode seeds | Role | Status | Notes |
|---|---|---|---|
| `10000-10019` | Training-time checkpoint validation | Used repeatedly | `train.py` evaluates checkpoints on this fixed 20-episode set. It is not a final test set. |
| `10020-10099` | Held-out checkpoint-selection audit | **Consumed once on 2026-10-27** | Evaluated the three already-selected checkpoints on 80 episode seeds not used in the original selection loop. Development-validation evidence only; do not reuse for tuning or confirmation. See `docs/CHECKPOINT_SELECTION_AUDIT.md`. |
| `150000-150199` | Development/tuning evaluation | Consumed | Reused to compare obstacle-risk weights 4.0, 1.0, and 2.0. Treat these results as development evidence. |
| `300000-300199` | Independent evaluation of training seeds 17, 29, and 41 | **Consumed once on 2026-10-27** | These results are the independent evaluation reported in `outputs/independent_validation_20261027/`. Do not reuse for tuning or a second confirmation. |
| `400000-400199` | Final held-out benchmark | **Consumed once on 2026-10-27** | Frozen seed-17 candidate vs scripted baseline, 200 episodes each. DQN 199/200 success (99.5%), one obstacle collision; baseline 200/200 success. No wind/current disturbances. See `docs/FINAL_TEST_REPORT.md`. Do not reuse. |
| `980000-980002` | Reproducibility smoke evaluation | Used once | Three short episodes to confirm the packaged candidate checkpoint loads and the evaluation pipeline runs; not a performance claim. The trajectory plotting seed was `990000`. |

Training RNG seeds:

- `17`: risk-weight-2.0 candidate selected during development; not a fresh independent training replicate.
- `29`: first fresh replicate using the frozen risk-weight-2.0 configuration.
- `41`: second fresh replicate using the same frozen configuration.

The model checkpoint selector uses the fixed validation episodes above, prioritizing docking success, then collision rate, discounted return and final distance. All final-test results should be stated alongside the exact training seed and checkpoint used.
