# CSCS PyTorch Course

Clean-slate implementation of the complete three-day course, including pre-course beginner preparation and large-scale model training on Alps.

The legacy `../notebooks/` and `../slides/` trees remain operational while material is moved or copied into this directory in history-preserving commits.

## Canonical environment

The PyTorch/CUDA/NCCL/network stack comes from the image pinned in [`environment/image.env`](environment/image.env). The image is pinned by both readable tag and immutable manifest digest.

Course-owned Python tools are installed with `uv` into `.venv`. The environment intentionally does not declare `torch` as a dependency: inside the canonical container, `.venv` exposes the image-provided system packages.

Pull the exact image for local inspection:

```bash
./environment/pull-image.sh
```

On Alps, launch the image through the checked-in EDF. The Alps extended image requires its entrypoint, PMIx, and the image-provided network libraries:

```bash
./environment/run-alps.sh \
  --account=<account> \
  --partition=debug \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=1 \
  ./environment/smoke-gpu.sh
```

Pass `--environment` to `srun`, not as an `#SBATCH` directive. The EDF follows the [CSCS Alps extended image guidance](https://docs.cscs.ch/software/alps-extended-images/).

```bash
./environment/create-venv.sh
uv run course-doctor
```

A GPU allocation should pass `environment/smoke-gpu.sh`. Set `COURSE_MINIMUM_GPUS=4` for the full-node Phase 0 check.

The full-node check additionally initializes NCCL and performs an all-reduce across four local ranks:

```bash
./environment/run-alps.sh \
  --account=<account> \
  --partition=debug \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=4 \
  ./environment/smoke-node.sh
```

Install the Jupyter kernel after creating the environment:

```bash
./environment/install-kernel.sh
```

## Slides

Slidev has a separate locked Node environment. Node is installed locally through `nodeenv`; no system Node installation is required.

```bash
./environment/create-slide-env.sh
./environment/run-slides.sh run build
```

The static build is written to `build/slides/`.

## Intended structure

```text
course/
├── environment/       pinned runtime and bootstrap commands
├── src/               importable reference implementations
├── configs/           beginner, single-GPU, and distributed configurations
├── exercises/         preparation and day-specific participant work
├── solutions/         verified reference solutions
├── notebooks/         educational and analysis views
├── slides/            preparation and three-day decks
├── jobs/              bounded single- and multi-node launchers
├── data/              manifests and preparation code, never bulk datasets
├── reports/           report generators and small reference summaries
├── tests/             behavioral correctness checks
└── tools/             release and authoring utilities
```

Directories are created when they acquire real content; empty placeholder directories are not committed.

## Source of truth

- Executable behavior lives in importable Python modules or explicit exercise programs.
- Day 1 training loops remain visible rather than hidden behind a generic trainer.
- Notebooks explain, exercise, and visualize tested behavior.
- Slides cite exact files and commands instead of maintaining divergent implementations.
- Generated datasets, checkpoints, notebook outputs, traces, and reports stay outside Git.

See [`OUTLINE.md`](OUTLINE.md) for the rough learning spine and [`CONVENTIONS.md`](CONVENTIONS.md) for authoring and migration rules.
