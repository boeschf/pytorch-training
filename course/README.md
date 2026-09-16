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

Directories appear as their runnable material is added. The clean-slate rebuild is currently at the environment and course-structure stage.

## Command help

```bash
uv run course --help
uv run course doctor --help
```

Instructor, developer, slide-authoring, Alps runtime, and migration instructions are in [`DEVELOPMENT.md`](DEVELOPMENT.md).
