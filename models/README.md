# Trained model artifact

`harborpilot_dqn_two_obstacle_risk2_seed17.pt` is the saved best-validation checkpoint from the two-obstacle, no-disturbance development run using training seed 17, observation version 3, hybrid reward version 3, and obstacle-risk weight 2.0.

It is included here because `outputs/` is ignored by `.gitignore`; putting this selected checkpoint in `models/` makes it straightforward to include the trained agent in the eventual source repository, as the assignment requires.

Important caveats:
- This is a development-selected checkpoint, not proof that the training procedure is reliable across seeds.
- Under the same nominal training configuration, seeds 29 and 41 performed much worse.
- The test seeds 300000–300199 are consumed. Do not use them to tune this checkpoint.
- The exact historical source that created all version-3 checkpoints was not recovered byte-for-byte. See `docs/SOURCE_RECOVERY.md`.
