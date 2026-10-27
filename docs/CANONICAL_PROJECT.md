# Canonical project and frozen candidate

**Canonical working copy:** this directory (`HarborPilot-RL-CANONICAL`).  Use this copy as the single source of truth for future work. Other `/mnt/data/HarborPilot-RL*` folders and ZIPs are historical snapshots/experiment exports; do not edit them as part of ongoing development.

## Current candidate (not yet final)

The current development candidate is copied to `models/harborpilot_dqn_two_obstacle_risk2_seed17.pt` (original experiment artifact: `outputs/two_obstacle_riskweight2_20261027/harborpilot_dqn_best.pt`), trained with two obstacles, disturbances disabled, observation version 3, hybrid reward version 3, obstacle-risk weight 2.0, risk distance 100, dock-control weight 1.0, max episode length 320, 1,000 training episodes, training seed 17, and epsilon decay 80,000 environment steps. The exact DQN parameters and evaluation seed policy are recorded in `configs/frozen_candidate_two_obstacle_risk2.json`.

This freezes the candidate configuration for the next validation step; it does **not** claim the candidate is final or optimal. Two additional training seeds (29 and 41) performed poorly under the same nominal configuration, so reliable learning is not yet established.

## Reproducibility cautions

- The 200 development episode seeds 150000-150199 were reused during risk-weight tuning. Do not report them as independent final-test results.
- Episode seeds 300000-300199 were used once for independent evaluation of training seeds 17, 29, and 41 on 2026-10-27. Do not reuse them for tuning or another confirmation. See `docs/SEED_USAGE_LEDGER.md`.
- Independent training seeds 29 and 41 were run under the frozen risk-weight-2.0 configuration. Their held-out success was 7% and 26%, compared with 99% for the development-selected seed-17 checkpoint. Training-seed sensitivity is now a central unresolved issue.
- The training script uses fixed validation seeds 10000-10019 for checkpoint selection. The additional seeds 10020-10099 were used once to audit whether the selected checkpoints generalized beyond that 20-episode selection set. For the three saved checkpoints, their performance ranking carried over, but this does not resolve training-seed sensitivity; see `docs/CHECKPOINT_SELECTION_AUDIT.md`. The additional seeds are now consumed development-validation data, not final-test data.
- The original historical source that generated all version-3 reward checkpoints was not recovered byte-for-byte. The present source is the documented compatibility reconstruction; keep this provenance caveat in reports.
- Checkpoint files do not save optimizer state or replay memory, so they are suitable for inference and weight warm-starting, but not exact continuation of an interrupted training process.

## Integrity

`configs/frozen_candidate_two_obstacle_risk2.json` is the human-readable configuration snapshot. `docs/CANONICAL_MANIFEST.json` records hashes of the project files in this frozen package. Recreate the manifest after intentional changes to source/configuration; document any change to the candidate in the experiment notes.

## Independent validation outcome and next work

The independent evaluation described in `docs/INDEPENDENT_VALIDATION_REPORT.md` and the additional checkpoint-selection audit in `docs/CHECKPOINT_SELECTION_AUDIT.md` are complete. Seed 17 achieved 198/200 docks on the new episode set, but fresh training seeds 29 and 41 achieved only 14/200 and 52/200. Do not proceed to wind/current disturbances yet. The training-stability audit is documented in `docs/TRAINING_STABILITY_AUDIT.md`. The existing logs confirm substantial training-seed sensitivity but do not identify a specific code defect or justify one hyperparameter change on their own. Preserve the seed-17 best checkpoint for now; if further experimentation is attempted, change one factor at a time and compare multiple training seeds on development/validation data. Episode seeds 300000-300199 and 10020-10099 are consumed; episode seeds 400000-400199 remain reserved for final evaluation after the candidate and evaluation procedure are frozen.
