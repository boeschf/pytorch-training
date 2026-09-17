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

## First runnable slice

From any activated profile:

```bash
course prep tensor-device --device cpu
course train mlp --device cpu
```

Then complete the five explicit optimization operations in
[`lessons/day1/mlp/exercise.py`](lessons/day1/mlp/exercise.py) and run its
self-check:

```bash
course exercise mlp --device cpu
```

The complete answer is in the adjacent `solution.py`; the shared training
implementation, notebook, and slides are in the same directory.

## Course material

Material is organized by lesson rather than artifact type:

```text
lessons/
├── prep/
│   └── tensor_device/
└── day1/
    └── mlp/
        ├── training.py
        ├── exercise.py
        ├── solution.py
        ├── notebook.py
        ├── notebook.ipynb
        └── slides.md
```

The Phase 1 golden slice covers the preparation diagnostic and deterministic Day 1 MLP. Later lessons appear as their runnable material is added.

## Command help

```bash
course --help
course doctor --help
```

Instructor, developer, slide-authoring, Alps runtime, and migration instructions are in [`DEVELOPMENT.md`](DEVELOPMENT.md).
