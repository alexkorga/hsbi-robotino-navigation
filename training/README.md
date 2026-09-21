# Training scripts

Start with [TRAINING.md](../TRAINING.md) for the current experiments, commands,
evaluation, and deployment steps. Architecture widths, curriculum, rewards,
PPO settings, and the training loop remain visible in each `train_*.py` script;
the network implementations live in `src/robotino_fleet/learning/`.

`common.py` contains shared environment, path, device, progress, and
parameter-count helpers; it does not define an experiment or run training.
