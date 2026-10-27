# Stage 1 canonical source and reproducibility audit

**Status:** complete for source organization, candidate artifact, configuration, tests, and smoke evaluation. No RL algorithm or environment behaviour was changed during this stage.

## Canonical project

- Root: this directory (contains `README.md`).
- Development candidate: `models/harborpilot_dqn_two_obstacle_risk2_seed17.pt`.
- Human-readable configuration: `configs/frozen_candidate_two_obstacle_risk2.json`.
- Original experiment checkpoint remains in `outputs/two_obstacle_riskweight2_20261027/harborpilot_dqn_best.pt` for provenance.
- Older directories and ZIPs are historical snapshots. Use this project tree for new work.

## Checks performed

- Extracted the complete `HarborPilot-RL-CANONICAL.zip` into a clean working tree.
- Confirmed `python -m pytest -q` passes all 16 tests (11 environment tests and 5 DQN-agent regression tests).
- Confirmed train/evaluate CLI help works.
- Confirmed frozen configuration matches the candidate checkpoint's saved environment metadata and DQN parameters.
- Ran a three-episode smoke evaluation to confirm checkpoint loading and the end-to-end evaluation pipeline. It used episode seeds 980000–980002 and trajectory seed 990000; these are recorded as smoke-only in `docs/SEED_USAGE_LEDGER.md`.
- Verified the selected candidate path is under `models/`, so it is not excluded by the `outputs/*` rule in `.gitignore`.
- Refreshed the file manifest and SHA-256 checksum after intentional documentation/config/model-copy changes.

## Smoke-evaluation result

The three smoke episodes yielded 100% docking success for both the packaged DQN candidate and scripted baseline. This is a pipeline smoke check only, not a performance claim or an independent evaluation.

## Known limitations

- The candidate's independent evaluation (seeds 300000–300199) is consumed. Seeds 400000–400199 remain reserved for a future final evaluation after a revised candidate is frozen.
- The DQN training process remains seed-sensitive: the existing training seeds 29 and 41 performed poorly compared with seed 17.
- The exact historical source that generated all version-3/reward-version-3 checkpoints was not recovered byte-for-byte; see `docs/SOURCE_RECOVERY.md`.
- A Git repository and genuine commit history have not yet been created. That is a later submission-preparation task; do not fabricate historical commits.
