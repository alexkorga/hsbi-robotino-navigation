# Train and Evaluate Navigation Models

[Overview](README.md) · [Setup](SETUP.md) · [Run](RUN.md) · [API](API.md) · Training · [Architecture](ARCHITECTURE.md) · [Future work](FUTURE_WORK.md)

Training and evaluation run offline in the same factory simulation used by the
navigation runtime. Complete [Setup](SETUP.md), then run the commands below
from the repository root.

## Train

The default deployed model is a shared GRU actor-critic. This complete example
trains its base model, then fine-tunes it on corner-navigation scenarios:

```text
uv run python training/train_shared_multi_robot_gru.py --device auto --worlds 4 --seed 42 --output models/shared_multi_robot_gru.zip
uv run python training/train_shared_multi_robot_gru_corners.py --device auto --worlds 4 --seed 42 --source models/shared_multi_robot_gru.zip --output models/shared_multi_robot_gru_corners.zip
```

The second command reads the model produced by the first. For the alternative
four-frame CNN using SB3 PPO, run:

```text
uv run python training/train_shared_multi_robot_cnn_temporal_wide.py --device auto --worlds 4 --seed 42 --output models/shared_multi_robot_cnn_temporal_wide.zip
uv run python training/train_shared_multi_robot_cnn_temporal_wide_corners.py --device auto --worlds 4 --seed 42 --source models/shared_multi_robot_cnn_temporal_wide.zip --output models/shared_multi_robot_cnn_temporal_wide_corners.zip
```

These commands use four worlds as an example; adjust `--worlds` to your
machine as described below. `--device auto` selects an available training
device. Each script writes the named model and a matching `.profile.json`.

### Choose the number of simulation worlds

`--worlds` sets the number of independent simulation workers. Each world in
these experiments contains three Robotinos. The scripts default to two worlds
and run workers in separate processes.

Choose this value for the CPU running the simulation, even when model training
uses a GPU. Start with the default, increase toward the number of physical CPU
cores, and test counts up to the number of logical threads if helpful. A count
below either number may be fastest on a processor with stronger individual
cores or limited memory bandwidth. Compare the progress bar's agent steps per
second after startup, using the same experiment phase, and keep the count that
gives the best sustained throughput. More worlds are not automatically faster;
each worker also consumes memory.

Use `--device cuda:0` to select a supported NVIDIA GPU explicitly. Checkpoints
and TensorBoard logs go under `runtime/`. Each script's `--help` lists its
additional options.

## Evaluate

Evaluation is a separate step. This command evaluates the fine-tuned GRU from
the training example in 100 three-Robotino episodes:

```text
uv run python training/evaluate_shared_multi_robot.py --model models/shared_multi_robot_gru_corners.zip --device cpu --episodes 100 --robots-per-world 3 --worlds 4 --seed 100000
```

The multi-Robotino evaluator reports success, collisions, timeouts, and path
results, and saves a `.multi_evaluation.json` beside the model. To evaluate
the fine-tuned CNN instead, change `--model` to
`models/shared_multi_robot_cnn_temporal_wide_corners.zip`. The
[model comparison](models/deliverable/README.md) summarizes the retained
artifacts.

## Develop a new architecture

Keep the training experiment in a `training/train_*.py` script: define its
curriculum, rewards, network sizes, PPO settings, and training loop there.
Put reusable network classes and feature extractors in
`src/robotino_fleet/learning/`, so the saved model can import them again during
evaluation and runtime inference. The GRU implementation in `gru_policy.py`
and the CNN extractor in `extractors.py` are examples.

Keep the saved `.profile.json` aligned with the model's observations, actions,
and timing. If the new architecture uses the existing policy contract, train
and load it through the current controller. For a new observation format,
action meaning, or model artifact format, update the corresponding builders
and loaders in `learning/`, the runtime controller in `navigation/`, and its
selection in `bootstrap.py`. Evaluate the saved artifact in a fresh process
before using it in simulation or with physical Robotinos; see
[Architecture](ARCHITECTURE.md#where-to-extend-the-project).

## Use a trained model

Keep the model `.zip` and its `.profile.json` together. Set `MODEL_PATH` in
`run_simulation.py` to inspect the model through the API or optional dashboard.
Set the corresponding path in `run_shadow.py` and `run_live.py` for laboratory
operation. The runtime checks the saved observation and action contract when
loading a model. See [Run](RUN.md#physical-operation) for physical operation
and [Architecture](ARCHITECTURE.md#where-to-extend-the-project) for adding a
new policy contract.
