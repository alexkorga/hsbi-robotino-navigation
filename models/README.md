# Trained models

Training scripts write model `.zip` files here. CNN policies use SB3 archives;
the GRU policy uses a custom PyTorch checkpoint with its architecture and
weights. Generated experiment models and evaluation files are ignored by Git.

Reviewed model candidates and their reports are kept in `deliverable/`.
Intermediate phase checkpoints are written under `runtime/`.

Profiled local-target models also have a matching `.profile.json` sidecar. It
records LiDAR binning, action timing, local-target geometry, and deterministic
tracking gains. Keep the zip and sidecar together; runtime loading deliberately
fails when their contract cannot be verified.

Set `MODEL_PATH` in `run_simulation.py`, `run_shadow.py`, or `run_live.py` to
the relative path of the artifact that should perform navigation. Profiled
models use either two-value local targets or a three-value target with learned
rotation, depending on the profile. The runtime checks the saved spaces and
sidecar before inference. See [TRAINING.md](../TRAINING.md).
