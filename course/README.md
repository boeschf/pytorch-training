# CSCS PyTorch Course

Participant material for the three-day PyTorch course on Alps.

The course starts with tensors and training loops, progresses through CNNs and transformers, and ends with distributed large-model training.

## Choose a runtime profile

The repository has three isolated Python 3.12 environments:

| Profile | Environment | PyTorch source | Use |
| --- | --- | --- | --- |
| `cpu` | `.venv-cpu` | locked CPU wheel from the PyTorch index | default laptop and workstation path |
| `cuda` | `.venv-cuda` | locked CUDA 12.8 wheel from the PyTorch index | a generic NVIDIA workstation |
| `alps-gh200` | `.venv-alps-gh200` | pinned course image, not `uv` | GH200 exercises on Alps |

Do not reuse one environment for another profile. The setup command checks both
the profile marker and who owns the PyTorch installation.

## Laptop or workstation setup

The portable CPU profile is the default:

```bash
./environment/create-venv.sh cpu
source .venv-cpu/bin/activate
course doctor
course kernel install
```

On a workstation with a compatible NVIDIA driver, replace `cpu` with `cuda`:

```bash
./environment/create-venv.sh cuda
source .venv-cuda/bin/activate
course doctor
course kernel install
```

The CUDA profile verifies the locked CUDA build even when no GPU is currently
visible. A command that needs a GPU still fails clearly if CUDA is unavailable.

## Bash completion

From any activated course profile, load the packaged static completion script:

```bash
source <(course completion bash)
```

The script completes commands, nested commands, options, fixed choices, and
output paths. Add the same line to shell initialization that runs after the
selected course profile is activated.

## Alps GH200 setup

Create the Alps environment inside the pinned course image so it inherits that
image's PyTorch, CUDA, NCCL, and network stack:

```bash
./environment/run-alps.sh \
  --account=<account> --partition=debug \
  --nodes=1 --ntasks=1 --gpus-per-node=1 --time=00:10:00 \
  ./environment/create-venv.sh alps-gh200
```

The environment contains only a small course-source bootstrap. It never installs
or shadows the image's PyTorch. Activate it only inside the same image, for
example:

```bash
./environment/run-alps.sh \
  --account=<account> --partition=debug \
  --nodes=1 --ntasks=1 --gpus-per-node=1 --time=00:10:00 \
  bash -lc 'source .venv-alps-gh200/bin/activate && course doctor && course kernel install'
```

## Start JupyterLab

From an activated profile:

```bash
jupyter lab
```

Installed kernels are named **CSCS PyTorch Course — CPU**, **— CUDA**, and
**— Alps GH200**. They can coexist. The checked-in notebook selects CPU by
default; choose another course kernel when the lesson requires it.

## Beginner preparation and Day 1

The complete beginner path runs on the portable CPU profile without a cluster or dataset download:

```bash
course prep tensor-device --device cpu
course prep python-numpy
course prep tensors-autograd --device cpu
course prep data-loader
course train mlp --device cpu
course train cnn --device cpu
course diagnose evaluation
```

Each exercise states its outcome, prerequisites, expected runtime, and resources. Complete its bounded gaps, then use the self-check:

```bash
course exercise mlp --device cpu
course exercise cnn --device cpu
course exercise debugging-evaluation
```

Preparation exercises run directly from their lesson directories. Every exercise has an adjacent executable `solution.py`.

## Course material

Material is organized by lesson rather than artifact type:

```text
lessons/
├── prep/
│   ├── python_numpy/
│   ├── tensor_device/
│   ├── tensors_autograd/
│   └── datasets_loaders/
└── day1/
    ├── mlp/
    ├── cnn/
    └── debugging_evaluation/
```

Topic directories keep executable behavior, exercises, solutions, paired notebooks, and slides together. The required Day 1 MLP and CNN paths are deterministic and bounded for CPU execution.

## Command help

```bash
course --help
course doctor --help
```

Instructor timings and delivery notes are in [`INSTRUCTOR.md`](INSTRUCTOR.md). Developer, slide-authoring, Alps runtime, and migration instructions are in [`DEVELOPMENT.md`](DEVELOPMENT.md).
