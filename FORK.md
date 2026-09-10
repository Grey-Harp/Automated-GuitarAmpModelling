# Maintained reference trainer

Based on GuitarML/Automated-GuitarAmpModelling at
`a0b05caa9066b094de5122222cca3dcb61ec48bd`, with a pinned CoreAudioML fork.
The first maintained revision transfers an already-used patched trainer into
ordinary source control. It does not change the network, loss, data split,
alignment, or accepted early-stopping behavior.

Changes carried forward:

- Scheduler construction compatible with the tested PyTorch runtime.
- Existing relative improvement factor of 0.999 for checkpoint/patience decisions.
  TrainTrack still updates its best value on any decrease. This is preserved
  behavior, not a newly designed stopping rule.
- Float32 preprocessing accumulators and parameter channels.
- Seed application after JSON configuration, including seed zero.
- TensorBoard events scoped to the model result directory.
- Explicit initial-weight, RNG, effective-configuration and optimizer receipts.
- Checkpoint lifecycle calls after the two training-statistics save boundaries.

Ordinary CLI training uses the upstream arguments. Set an explicit seed in the
config or CLI when required. No cloud integration is imported by default.

Strict reference mode requires all of `PET_EXPECTED_CONFIG`,
`PET_PROVENANCE_DIR`, `PET_RUN_ID`, and `PET_REFERENCE_SPEC`. The specification
supplies `reproducibility` and `code_settings`; the expected config contains all
resolved argparse values. `PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG` must be
set before interpreter startup. Receipts are sealed before the first epoch.

An optional `TRAINER_CHECKPOINT_CALLBACK=module:function` receives `save_path`
after checkpoint/statistics writes. Cloud orchestration supplies this callback
and remains outside this repository. Callback failure propagates to the caller.

Known boundaries retained from the accepted version: validation/test sequences
must be at least one chunk; the training frame is hardcoded to 22050 samples;
`segment_length`, `data_location`, `iter_num`, and `num_blocks` do not control
those parts of this trainer. Training state is not fully resumed from weights.
CPU reproducibility tests do not guarantee identical GPU results.

Tests from repository root (with submodule initialized and dependencies present):

    python -B -m unittest discover -s tests
    python -B -m unittest discover -s CoreAudioML/tests

Licenses and upstream attribution are retained in LICENSE.md and README.md.
