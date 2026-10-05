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
those parts of this trainer. Weight-only loading does not restore training state; explicit continuation is described below.
CPU reproducibility tests do not guarantee identical GPU results.

Tests from repository root (with submodule initialized and dependencies present):

    python -B -m unittest discover -s tests -p test_reference_training.py
    python -B -m unittest discover -s CoreAudioML/tests

Licenses and upstream attribution are retained in LICENSE.md and README.md.

## Full-state continuation

`dist_model_recnet.py` now writes `training_state.pt` and a hash manifest
`training_state.json` after each completed epoch. These contain current model,
Adam, ReduceLROnPlateau, TrainTrack, patience and Python/NumPy/torch CPU/CUDA
RNG states. Recurrent hidden state must be reset at this boundary. The manifest
binds the model exports and statistics to the tensor state. Initial provenance
is retained, not regenerated on continuation.

Set `PET_RESUME_DIR` to a verified immutable checkpoint directory and use a new
empty output directory. The original logical `PET_RUN_ID`, numerical arguments,
all six prepared data files, source code and runtime settings must match.
Resume uses `torch.load(..., weights_only=True)` after hash verification.
Old weight-only snapshots are deliberately rejected. Finished results cannot
start new training; a checkpoint at a training-complete boundary can finish its
pending final tests without running another epoch.

`PET_PAUSE_AFTER_EPOCH` or `PET_PAUSE_AFTER_SECONDS` requests a pause at an
epoch boundary; exit code 75 means paused, not completed. Limits must be positive
integers. No final tests are performed at a pause. The cloud callback must copy
both training-state files, model exports, stats and provenance before declaring
an attempt recoverable. Hard VM loss can still discard work after the last
confirmed remote snapshot. CPU equivalence does not certify a CUDA continuation.

The older statement above about incomplete weight-only resume applies to old
snapshots and the legacy `load_model` path, not this explicit full-state path.

The `test_training_resume.py` integration suite runs from the containing Pet-Sim
checkout: it uses its frozen specification, Colab snapshot callback and the
previous maintained trainer commit as a numerical control. Run it through
`Colab/verify_continuation.py` with an explicit report output path.

## Fixed-terminal paired warm start

`gain_balance.py` implements a separate DS1 gain-balanced experiment. It reads
only hash-pinned initial weights, creates fresh Adam state, and keeps C/T model
and RNG states separate. A saved CPU permutation plan is shared by both arms;
C then T execute each epoch. The legacy files and finalized-resume prohibition
are unchanged. C calls the pinned LossWrapper. T filters the whole update
before grouping samples by Gain, including transitions inside a segment.

Snapshots contain both arms, optimizer/RNG/history and the plan. Terminal
exports require completion of both fixed endpoints; validation cannot select
weights or change training order. The containing project owns data identity,
authorization, resource limits and remote snapshot handling. No cloud or data
paths are embedded in this module. CPU controls include independent finite
differences, real legacy Adam updates, C/C identity, and paired recovery.

    python -B -m unittest discover -s tests -p test_gain_balance.py -v

CPU equivalence does not certify CUDA equivalence, speed or sound quality.
