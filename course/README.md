# CSCS PyTorch Course

Participant material for the three-day PyTorch course on Alps.

The course starts with tensors and training loops, progresses through CNNs and transformers, and ends with distributed large-model training.

## One-time setup

From this directory:

```bash
./environment/create-venv.sh
uv run course doctor
uv run course kernel install
```

The first command creates the locked Python 3.12 environment. PyTorch, CUDA, NCCL, and the Alps network stack come from the canonical course container rather than from `uv`.

## Start JupyterLab

```bash
uv run jupyter lab
```

Select the **CSCS PyTorch Course** kernel when opening a notebook.

## First runnable slice

Inside the canonical course image:

```bash
export PYTHONPATH="$PWD/src"
python -m pytorch_course.cli prep tensor-device --device cpu
python -m pytorch_course.cli train mlp --device cpu
```

Then complete the five explicit optimization operations in [`exercises/day1/mlp.py`](exercises/day1/mlp.py). The reference path, thin notebook, and slides all use the implementation under `src/pytorch_course/foundations/`.

A workstation without PyTorch can still build slides, edit material, and run the authoring tools. PyTorch exercises and notebooks require the canonical image.

## Course material

The participant-facing material will be organized as:

```text
exercises/prep/   pre-course Python, NumPy, tensor, and environment preparation
exercises/day1/   tensors, autograd, explicit training loops, MLPs, and CNNs
exercises/day2/   robust training, attention, transformers, and first distributed run
exercises/day3/   large-model parallelism, MoE, profiling, and kernels
notebooks/        guided explanations and analysis
slides/           presentation source
```

The Phase 1 golden slice now covers the preparation diagnostic and a deterministic Day 1 MLP. Later sections appear as their runnable material is added.

## Command help

```bash
uv run course --help
uv run course doctor --help
```

Instructor, developer, slide-authoring, Alps runtime, and migration instructions are in [`DEVELOPMENT.md`](DEVELOPMENT.md).
