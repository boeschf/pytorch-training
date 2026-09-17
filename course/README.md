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
export PYTHONPATH="$PWD:$PWD/src"
python -m pytorch_course.cli prep tensor-device --device cpu
python -m pytorch_course.cli train mlp --device cpu
```

Then complete the five explicit optimization operations in [`lessons/day1/mlp/exercise.py`](lessons/day1/mlp/exercise.py). Its reference implementation, solution, notebook, and slides are in the same directory.

A workstation without PyTorch can still build slides, edit material, and run the authoring tools. PyTorch exercises and notebooks require the canonical image.

## Course material

Material is organized by lesson rather than artifact type:

```text
lessons/
├── prep/
│   └── tensor_device/
└── day1/
    └── mlp/
        ├── reference.py
        ├── exercise.py
        ├── solution.py
        ├── notebook.py
        ├── notebook.ipynb
        └── slides.md
```

The Phase 1 golden slice covers the preparation diagnostic and deterministic Day 1 MLP. Later lessons appear as their runnable material is added.

## Command help

```bash
uv run course --help
uv run course doctor --help
```

Instructor, developer, slide-authoring, Alps runtime, and migration instructions are in [`DEVELOPMENT.md`](DEVELOPMENT.md).
