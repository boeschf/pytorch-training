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

On a workstation without PyTorch, this environment still supports slide and notebook authoring, Jupytext synchronization, linting, and documentation work. Executing PyTorch cells or training commands requires the canonical image.

## Common commands

```bash
# One-time Python setup
./environment/create-venv.sh

# Environment reports
uv run course doctor
uv run course doctor --json

# Participant kernel
uv run course kernel install

# Golden vertical slice
uv run course prep tensor-device --device cpu
uv run course train mlp --device cpu --json

# Slides
uv run course slides setup
uv run course slides build
uv run course slides dev
uv run course slides preview
uv run course slides audit
uv run course slides export

# One-node image and NCCL verification
uv run course alps-smoke --account=<account> --gpus=4
```

Run `uv run course --help` for the complete command surface. Scripts under `environment/` are implementation details and remain directly callable for debugging.

The `uv run` training commands assume the selected Python environment can import PyTorch. The canonical image already contains the runtime but not `uv`; execute the same CLI there with:

```bash
export PYTHONPATH="$PWD:$PWD/src"
python -m pytorch_course.cli prep tensor-device --device cpu
python -m pytorch_course.cli train mlp --device cuda --json
```

## Notebook and slide integration

Notebooks and slides remain separate educational views: notebooks support exploration and retained outputs, while slides support presentation pacing and concise explanation. Embedding a live Jupyter interface in Slidev was rejected because it rendered poorly, coupled the deck to a running notebook server, and did not remove source drift.

Each directory under `lessons/` keeps one topic's implementation, exercise, solution, notebook, and slides together. For the MLP lesson:

```text
lessons/day1/mlp/
├── reference.py
├── exercise.py
├── solution.py
├── notebook.py
├── notebook.ipynb
└── slides.md
```

`reference.py` is the executable source of truth. The paired notebook imports it, the solution and smoke scenarios call it, and `slides.md` imports marked regions directly from it. `slides/lessons` is a relative symlink into the lesson tree because Slidev restricts imported Markdown and snippets to its project root.

Synchronize the participant notebook after editing its adjacent percent-format source:

```bash
uv run jupytext --sync lessons/day1/mlp/notebook.py
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

PDF export uses the pinned `playwright-chromium` development dependency:

```bash
uv run course slides export
```

The PDF is written to `build/pytorch-course.pdf`. The first `uv run course slides setup` downloads the matching Chromium build, so it requires network access and takes longer than later clean installs. npm is allowed to run only the install script for the exact pinned Playwright Chromium version; that script installs the export browser.

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
