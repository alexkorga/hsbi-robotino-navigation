# Project Setup

[Overview](README.md) · Setup · [Run](RUN.md) · [API](API.md) · [Training](TRAINING.md) · [Architecture](ARCHITECTURE.md) · [Future work](FUTURE_WORK.md)

These instructions prepare the navigation runtime and training tools on
Windows, macOS, or Linux. The dashboard has a separate, optional setup step.
Runtime launch commands are in [Run](RUN.md).

## Prerequisites

Install [Git](https://git-scm.com/downloads) and
[uv](https://docs.astral.sh/uv/getting-started/installation/) for the Python
runtime. The project uses Python 3.11 or newer, which `uv` can install and
manage.

For the optional dashboard, also install
[Node.js](https://nodejs.org/en/download) `20.19.x` or `22.12` or newer and
[pnpm](https://pnpm.io/installation).

Confirm the required tools are available:

```text
git --version
uv --version
```

If using the dashboard, also check:

```text
node --version
pnpm --version
```

All remaining commands in this document are run from the repository root
unless a step explicitly changes into `frontend`.

## First-time Python setup

Create the local virtual environment and install the exact dependencies from
`uv.lock`:

```text
uv sync --locked
```

There is no need to activate `.venv`. Commands prefixed with `uv run` use it
automatically.

Windows and Linux install the configured CUDA-capable PyTorch package, which
also supports CPU inference. macOS resolves its supported PyTorch package from
PyPI. A GPU is useful for training but is not required to run the backend or
dashboard.

## Dashboard setup (optional)

Install the locked frontend dependencies and produce one complete build:

```text
cd frontend
pnpm install --frozen-lockfile
pnpm build
cd ..
```

The build verifies TypeScript, React, Three.js, and Vite. The Python backend
and its API run independently of this step.

## Verify the installation

Run the backend test suite:

```text
uv run python -m unittest discover -s tests -v
```

If using or developing the dashboard, run its tests and production build:

```text
cd frontend
pnpm test
pnpm build
cd ..
```

Continue with the launch commands in [Run](RUN.md).

## Updating an existing checkout

After pulling changes, synchronize the Python dependencies again:

```text
uv sync --locked
```

If using the dashboard, also synchronize its dependencies:

```text
cd frontend
pnpm install --frozen-lockfile
cd ..
```

Neither `.venv` nor `frontend/node_modules` belongs in Git. Each computer
creates its own local copies.

## Troubleshooting

### `vite` is not found or a native frontend binding is missing

Reinstall the frontend dependencies for the current operating system and CPU:

```text
cd frontend
pnpm install --force
pnpm build
```

Do not copy `frontend/node_modules` between Windows, macOS, Linux, x64, and
ARM64 computers.

### Windows blocks a PyTorch DLL

If Python reports `WinError 4551` for a DLL under
`.venv\Lib\site-packages\torch\lib`, Windows application control has blocked a
downloaded dependency. From PowerShell in the repository root, unblock the
installed PyTorch files and retry the command:

```powershell
Get-ChildItem .venv\Lib\site-packages\torch\lib -Recurse | Unblock-File
```

Only do this for the project environment created from the tracked `uv.lock`.

### `uv sync --locked` reports that the lock file is outdated

Do not regenerate the lock file merely to bypass the error. Confirm that
`pyproject.toml` and `uv.lock` came from the same revision. Dependency updates
should deliberately update and commit both files together.
