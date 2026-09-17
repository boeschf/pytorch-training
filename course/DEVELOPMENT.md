# Instructor and Developer Guide

This document covers course authoring, validation, slides, Alps execution, and migration from the legacy material. Participant setup remains in [`README.md`](README.md); delivery timings and intervention notes are in [`INSTRUCTOR.md`](INSTRUCTOR.md).

## Environment model

The canonical image is recorded by tag and immutable manifest digest in
[`environment/image.env`](environment/image.env). The checked-in
[`environment/alps.toml`](environment/alps.toml) follows the
[CSCS Alps extended image guidance](https://docs.cscs.ch/software/alps-extended-images/):

- execute the image entrypoint;
- use PMIx;
- disable the host CXI hook so the image network libraries take precedence;
- pass `--network=disable_rdzv_get`;
- pass the EDF to `srun`, not `sbatch`.

One `pyproject.toml` and one `uv.lock` define three mutually isolated profiles:

| Profile | Directory | PyTorch owner | Contract |
| --- | --- | --- | --- |
| `cpu` | `.venv-cpu` | `uv`, PyTorch CPU index | CPU-only build inside the venv |
| `cuda` | `.venv-cuda` | `uv`, PyTorch CUDA 12.8 index | CUDA 12.8 build inside the venv |
| `alps-gh200` | `.venv-alps-gh200` | canonical image | image build outside the venv, with distributed and NCCL |

The `cpu` and `cuda` optional dependencies are declared as conflicting extras,
so a resolver cannot combine their PyTorch builds. Both use the same locked
course, notebook, and development dependencies. `environment/create-venv.sh`
sets `UV_PROJECT_ENVIRONMENT` explicitly; plain `uv run` is avoided because it
would otherwise select the conventional `.venv` path.

The Alps profile is intentionally different. It creates a Python 3.12 venv with
`--system-site-packages` inside the image, then adds only the repository root and
`src/` through a `.pth` file. It does not invoke `uv` or `pip`, so it cannot
shadow the pinned image's PyTorch, CUDA, NCCL, or network libraries. The venv is
not portable: use `.venv-alps-gh200` only inside that image.

Every environment records a `.course-profile` marker. `course doctor` validates
the corresponding ownership and build invariants automatically. Passing
`--require-profile` additionally rejects an accidentally activated environment.

After changing Python dependencies, run `uv lock` once and recreate both local
profiles through `create-venv.sh`. Never run `uv sync` against the Alps profile.

## Common commands

Start with the portable authoring profile:

```bash
./environment/create-venv.sh cpu
source .venv-cpu/bin/activate
source <(course completion bash)

# Environment reports
course doctor
course doctor --json

# Profile-specific participant kernel
course kernel install

# Beginner preparation and Day 1
course prep tensor-device --device cpu
course prep python-numpy
course prep tensors-autograd --device cpu
course prep data-loader
course train mlp --device cpu --json
course train cnn --device cpu --json
course diagnose evaluation
# After completing the adjacent exercise files
course exercise mlp --device cpu
course exercise cnn --device cpu
course exercise debugging-evaluation
# Complete reference dry run
python tools/smoke_phase2.py --device cpu

# Slides
course slides setup
course slides build
course slides dev
course slides preview
course slides audit
course slides export

# One-node image and NCCL verification; requires .venv-alps-gh200
course alps-smoke --account=<account> --gpus=4
```

Run `course --help` for the complete command surface. Scripts under
`environment/` are implementation details and remain directly callable for
debugging.

The Bash completion script is generated from `create_parser()` and packaged with
the CLI, so runtime environments do not need `shtab`. Regenerate and verify it
after changing the command surface:

```bash
python tools/generate_completion.py
python tools/generate_completion.py --check
```

To author against a generic NVIDIA runtime, create and activate `cuda` instead.
Create the Alps profile in a bounded image allocation:

```bash
./environment/run-alps.sh \
  --account=<account> --partition=debug \
  --nodes=1 --ntasks=1 --gpus-per-node=1 --time=00:10:00 \
  ./environment/create-venv.sh alps-gh200
```

Kernel installation always derives its name from the active marker. The three
stable kernel IDs are `cscs-pytorch-course-cpu`,
`cscs-pytorch-course-cuda`, and `cscs-pytorch-course-alps-gh200`.

## Notebook and slide integration

Notebooks and slides remain separate educational views: notebooks support exploration and retained outputs, while slides support presentation pacing and concise explanation. Embedding a live Jupyter interface in Slidev was rejected because it rendered poorly, coupled the deck to a running notebook server, and did not remove source drift.

Each directory under `lessons/` keeps one topic's executable behavior, participant exercise, solution, notebook, and slides together where those artifacts support the learning objective:

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

The MLP and CNN `training.py` modules own shared data, model, evaluation, and training paths. Paired notebooks, the CLI, exercise harnesses, smoke scenarios, and slides call or quote those implementations rather than maintaining parallel versions. Every exercise has an executable adjacent `solution.py`.

Synchronize participant notebooks after editing their adjacent percent-format sources:

```bash
jupytext --sync lessons/prep/*/notebook.py lessons/day1/*/notebook.py
```

For an occasional live demonstration, open the notebook beside the deck rather than embedding Jupyter in a slide.

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

Run `course slides audit` after dependency updates. Do not run `npm audit fix --force`: its current recommendation downgrades the direct Slidev dependency and can silently change the authoring contract.

`image-size` crosses the major-version range currently declared by PptxGenJS. The HTML Slidev build is verified, but PPTX export is not a supported course workflow and must be re-evaluated before it is enabled.

## Slide output

`course slides build` writes the static application to `build/slides/`.

Slidev is a client-rendered single-page application. Its generated `index.html` intentionally contains only application mount elements such as:

```html
<div id="app"></div>
```

JavaScript renders the slides at runtime. Opening the file directly or inspecting only its body will therefore appear empty. Serve the build over HTTP instead:

```bash
course slides preview
```

For live authoring:

```bash
course slides dev
```

PDF export uses the pinned `playwright-chromium` development dependency:

```bash
course slides export
```

The PDF is written to `build/pytorch-course.pdf`. The first `course slides setup` downloads the matching Chromium build, so it requires network access and takes longer than later clean installs. npm is allowed to run only the install script for the exact pinned Playwright Chromium version; that script installs the export browser.

The eventual deployment must set an explicit Slidev base path if it is hosted below a URL prefix rather than at the web root.

## Alps smoke checks

The concise interface is:

```bash
course alps-smoke \
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
├── lessons/           topic-local code, exercises, solutions, notebooks, and slides
│   ├── prep/
│   ├── day1/
│   ├── day2/
│   └── day3/
├── src/               course CLI and non-lesson infrastructure
├── slides/            Slidev toolchain, entry deck, and lesson-tree bridge
├── environment/       pinned runtime and low-level launch scripts
├── configs/           shared runtime configurations
├── jobs/              bounded single- and multi-node launchers
├── data/              manifests and preparation code, never bulk datasets
├── reports/           report generators and small reference summaries
├── tests/             behavioral correctness checks
└── tools/             release and authoring utilities
```

New participant material belongs in a lesson directory. Cross-lesson infrastructure belongs under `src/pytorch_course/`; do not move teaching code there merely to make it importable.

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
