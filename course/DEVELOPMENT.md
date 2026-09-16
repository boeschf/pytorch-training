# Instructor and Developer Guide

This document covers course authoring, validation, slides, Alps execution, and migration from the legacy material. Participant setup remains in [`README.md`](README.md).

## Environment model

The canonical image is recorded by tag and immutable manifest digest in [`environment/image.env`](environment/image.env). The checked-in [`environment/alps.toml`](environment/alps.toml) follows the [CSCS Alps extended image guidance](https://docs.cscs.ch/software/alps-extended-images/):

- execute the image entrypoint;
- use PMIx;
- disable the host CXI hook so the image network libraries take precedence;
- pass `--network=disable_rdzv_get`;
- pass the EDF to `srun`, not `sbatch`.

The Python project intentionally does not declare `torch`. `environment/create-venv.sh` creates `.venv` with `--system-site-packages`, allowing the canonical container's PyTorch/CUDA/NCCL installation to remain authoritative. `uv` owns only course tooling such as Jupyter, Ruff, and pytest.

The bootstrap script remains necessary because `uv` has no `pyproject.toml` setting equivalent to `uv venv --system-site-packages`. After that one-time bootstrap, common workflows use the unified `course` command.

## Common commands

```bash
# One-time Python setup
./environment/create-venv.sh

# Environment reports
uv run course doctor
uv run course doctor --json

# Participant kernel
uv run course kernel install

# Slides
uv run course slides setup
uv run course slides build
uv run course slides dev
uv run course slides preview
uv run course slides audit

# One-node image and NCCL verification
uv run course alps-smoke --account=<account> --gpus=4
```

Run `uv run course --help` for the complete command surface. Scripts under `environment/` are implementation details and remain directly callable for debugging.

## Node and npm policy

Slide authoring uses:

```text
Node.js 24.21.0 LTS (Krypton)
npm 12.0.2
Slidev 53.0.0
```

Node 26 is newer but is not an LTS release. Node 24.21.0 is therefore intentional rather than stale. `nodeenv` installs Node locally under `.nodeenv`; the bootstrap then updates the bundled npm to the separately pinned npm 12.0.2.

The lockfile contains security overrides for vulnerable transitive versions selected by current Slidev dependencies:

- `lodash-es` 4.18.1;
- `dompurify` 3.4.15;
- `image-size` 2.0.4.

Run `uv run course slides audit` after dependency updates. Do not run `npm audit fix --force`: its current recommendation downgrades the direct Slidev dependency and can silently change the authoring contract.

`image-size` crosses the major-version range currently declared by PptxGenJS. The HTML Slidev build is verified, but PPTX export is not a supported course workflow and must be re-evaluated before it is enabled.

## Slide output

`uv run course slides build` writes the static application to `build/slides/`.

Slidev is a client-rendered single-page application. Its generated `index.html` intentionally contains only application mount elements such as:

```html
<div id="app"></div>
```

JavaScript renders the slides at runtime. Opening the file directly or inspecting only its body will therefore appear empty. Serve the build over HTTP instead:

```bash
uv run course slides preview
```

For live authoring:

```bash
uv run course slides dev
```

The eventual deployment must set an explicit Slidev base path if it is hosted below a URL prefix rather than at the web root.

## Alps smoke checks

The concise interface is:

```bash
uv run course alps-smoke \
  --account=<account> \
  --partition=debug \
  --gpus=4 \
  --time=00:05:00
```

This checks:

- the pinned image metadata and expected PyTorch build;
- Python, CUDA, NCCL, and visible GH200 devices;
- a four-rank NCCL all-reduce.

The implementation stores the EDF image cache under `course/.edf_imagestore/`, which is ignored by Git. This avoids relying on a potentially read-only `${SCRATCH}/.edf_imagestore` parent.

## Intended repository structure

```text
course/
├── environment/       pinned runtime and low-level launch scripts
├── src/               importable reference implementations and CLI
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

Directories are committed only when they contain real material.

## Source-of-truth rules

- Executable behavior lives in importable Python modules or explicit exercise programs.
- Day 1 training loops remain visible rather than hidden behind a generic trainer.
- Notebooks explain, exercise, and visualize tested behavior.
- Slides cite exact files and commands instead of maintaining divergent implementations.
- Generated datasets, checkpoints, notebook outputs, traces, and reports stay outside Git.

See [`OUTLINE.md`](OUTLINE.md) for the learning spine and [`CONVENTIONS.md`](CONVENTIONS.md) for naming, authoring, and Git migration rules.

## Git-native migration

For same-repository material:

1. use a dedicated `git mv` commit when the old path can disappear;
2. otherwise commit an exact copy while the old course remains operational;
3. adapt the destination only in later commits;
4. verify copy lineage with `git log --follow -C --find-copies-harder` and `git blame -C -C`.

Git history does not replace third-party copyright, licence, citation, or upstream-version notices.
