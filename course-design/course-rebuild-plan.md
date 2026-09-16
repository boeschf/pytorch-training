# Three-Day PyTorch Course Rebuild Plan

Status: Phases 0 and 1 implemented and verified on 2026-09-16; Phase 2 is next

Related detailed design: [`day-3-large-scale-models.md`](day-3-large-scale-models.md)

## Decision

Start from a new top-level `course/` directory and leave the current `slides/` and `notebooks/` trees unchanged while material is assessed and ported.

Use a code-first process, but not a code-only process:

1. define the three-day learning spine and executable contracts;
2. build one end-to-end runnable reference path;
3. make notebooks and exercises thin educational views of tested code;
4. write slides against behavior that already runs;
5. port old material only when it supports an explicit learning objective.

This prevents slides from describing APIs or commands that later change. A short curriculum skeleton must come first so the implementation serves the course rather than determining it accidentally.

The rebuild covers the complete course: pre-course beginner preparation, Days 1 and 2, and the large-scale-model material on Day 3.

## Constraints and principles

- Beginner-friendly entry point; familiarity with Python is assumed, prior PyTorch or ML knowledge is not.
- One conceptual progression from tensors and gradients to distributed transformer/MoE training.
- The code participants execute is the source of truth.
- Do not hide the training loop behind a framework before participants have implemented and understood it.
- Reuse small utilities only when the abstraction removes repetition without hiding the concept being taught.
- Keep the existing course operational until the new course has passed a complete dry run.
- No wholesale copying of the old tree. Port one coherent artifact at a time in Git history-preserving commits.
- Generated datasets, checkpoints, notebook outputs, traces, and benchmark artifacts do not belong in Git.
- Every exercise must have a bounded runtime and a non-GPU or recorded-data fallback where practical.

## Canonical runtime

Use the following Alps extended image as the base runtime:

```text
jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/pytorch-cuda:26.07-py3-alps7-dev-14ac42497583f60a1dd156230f00af8f
```

It provides NVIDIA PyTorch together with the Alps-specific network stack. Before implementation, resolve and record its immutable image digest; the tag remains the readable alias.

Use Podman for local image inspection and smoke runs. Use the supported Alps container launch path for Slurm exercises. Do not build another PyTorch/CUDA/NCCL stack on top of the image unless a demonstrated missing capability requires it.

Use `uv` for course-owned Python packages:

- create a virtual environment with access to the image's system site packages;
- never install or replace `torch`, CUDA, NCCL, or network-stack packages through `uv`;
- pin Jupyter, analysis, data, and authoring dependencies in `uv.lock`;
- separate optional dependency groups such as `notebooks`, `analysis`, and `dev`;
- run a startup check that records the actual PyTorch, CUDA, NCCL, GPU, and network-facing environment.

Slidev remains a Node project with its own pinned package lock. `uv` manages Jupyter and Python tooling, not Slidev's JavaScript dependencies.

The first environment proof must cover:

1. import and version report on the login/development host;
2. CPU execution of the smallest exercise;
3. one-GPU container execution;
4. four-GPU NCCL execution on one Alps node;
5. a bounded profiler trace;
6. Jupyter kernel registration using the same Python environment;
7. Slidev development build and static export.

## Proposed directory structure

```text
course/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── environment/
│   ├── image.env
│   ├── alps.toml
│   ├── create-venv.sh
│   └── doctor.py
├── src/pytorch_course/
│   ├── foundations/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── distributed/
│   ├── observability/
│   └── cli.py
├── configs/
│   ├── beginner/
│   ├── single_gpu/
│   └── distributed/
├── exercises/
│   ├── prep/
│   ├── day1/
│   ├── day2/
│   └── day3/
├── solutions/
├── notebooks/
│   ├── prep/
│   ├── day1/
│   ├── day2/
│   └── day3/
├── slides/
│   ├── prep/
│   ├── day1/
│   ├── day2/
│   ├── day3/
│   └── shared/
├── jobs/
│   ├── single_gpu/
│   └── distributed/
├── data/
│   ├── manifests/
│   └── preparation/
├── reports/
├── tests/
└── tools/
```

`data/` contains manifests and preparation code, not the prepared corpora. `reports/` contains report-generation code and small reference summaries, not full traces.

Avoid a universal trainer abstraction. Day 1 should show an explicit training loop. Reusable model, data, distributed, and observability components can emerge in Days 2 and 3 after the underlying operations are visible.

## Git-native authorship and history

Use Git history as the primary authorship record for material already in this repository. A separate course-wide attribution manifest is unnecessary when lineage can be retained through disciplined moves or copies.

Git stores snapshots rather than explicit rename/copy metadata, so history preservation depends on similarity detection and commit structure.

Rules:

- For a one-to-one relocation whose old path can disappear, use `git mv`.
- Make the move a dedicated commit with no content edits. Adapt the file in later commits.
- When the old course must remain operational, make an exact copy into `course/` and commit only that copy. Adapt it in a later commit.
- For copied files, verify lineage with `git log --follow -C --find-copies-harder -- <path>` and `git blame -C -C <path>`; the default history view may not follow copies.
- When changing representation, such as `.ipynb` to Jupytext `.py`, first move or copy the original unchanged, then convert it in a separate commit.
- Avoid combining unrelated ports in one commit. Small, source-preserving commits improve rename/copy detection and review.
- Mention original source paths in port commit messages, especially when one source is split or several sources are combined.
- Retain embedded copyright, creator metadata, citations, and licence notices.
- Git history is not a replacement for third-party licence obligations. Preserve notices and source/version references for externally adapted code, figures, datasets, and text.
- Do not delete the old source tree until the complete course is accepted. Individual `git mv` operations are appropriate only when removing that file cannot break the old course.

The migration check is therefore history-based rather than manifest-based: every reused same-repository artifact must have an inspectable move/copy boundary before adaptation.

## Three-day learning spine

This is deliberately a rough outline. Exact durations and some optional sections remain open until the executable path is timed.

### Pre-course preparation

Purpose: remove environment and Python/PyTorch mechanics from Day 1 without excluding beginners.

- course access, repository, Jupyter, terminal, and Slurm orientation;
- Python/NumPy shape and indexing refresher;
- tensors, dtype, shape, device, and basic operations;
- CPU versus GPU execution;
- environment doctor and a five-minute diagnostic exercise;
- optional deeper Python material for participants who need it.

The preparation must be self-checking and runnable without an Alps allocation wherever possible.

### Day 1: from tensors to a trained model

- supervised-learning vocabulary: samples, features, targets, parameters, loss;
- tensor shapes and broadcasting in model code;
- modules, parameters, forward passes, and autograd;
- explicit optimization loop: zero gradients, forward, loss, backward, step;
- datasets, splits, batching, and deterministic evaluation;
- debugging shapes, devices, gradients, and train/eval mode;
- MLP exercise, followed by a compact CNN/image-classification exercise;
- interpreting loss curves, accuracy, overfitting, and validation behavior.

Participants should finish Day 1 able to write and diagnose a small training loop without copying a framework template blindly.

### Day 2: robust training and the bridge to scale

- optimizers, learning rates, regularization, normalization, and initialization;
- mixed precision and memory/performance basics;
- checkpointing and reproducibility;
- transfer learning or fine-tuning as an applied exercise;
- tokens, embeddings, attention, transformer blocks, and next-token prediction;
- single-GPU profiling and the first structured run report;
- data parallelism and a first DDP exercise;
- optional FSDP2 introduction if timing permits.

Participants should finish Day 2 with a working decoder-only language-model path and a concrete reason for distributed execution.

### Day 3: large models on Alps

Use the detailed Day 3 design as the working specification:

- scaling arithmetic and parallelism dimensions;
- DDP and FSDP2 recap;
- tensor and sequence/context parallelism;
- pipeline parallelism;
- MoE architecture, routing, expert parallelism, and all-to-all;
- composable device meshes;
- measurement, profiler traces, communication overlap, and imbalance;
- Triton and optional Mojo kernel comparison;
- evidence-based end-to-end strategy selection.

Use the labelled Dolma course corpus, semantic checkpoints, systems configurations, and observability design described in `day-3-large-scale-models.md`.

## Executable architecture and progression

Build the implementation in layers that match the learning progression.

### Layer A: foundations

Small CPU/GPU examples with explicit operations:

- tensor/device exercise;
- linear regression or two-class classification;
- MLP with an explicit training loop;
- deterministic evaluation and checkpoint round trip.

These examples optimize clarity, not reuse.

### Layer B: reusable single-GPU components

Introduce only after Layer A:

- configuration loading;
- dataset and batch contracts;
- CNN and decoder-only transformer components;
- reusable evaluation;
- mixed precision;
- run manifests and JSON Lines metrics;
- checkpointing.

### Layer C: distributed strategies

Wrap the same transformer/MoE workload with independently selectable strategies:

- single rank reference;
- DDP;
- FSDP2;
- TP;
- PP;
- MoE/EP;
- selected compositions through `DeviceMesh`.

Every strategy must preserve a common batch, loss, correctness, metric, and checkpoint contract. Strategy-specific code should remain visible rather than hidden behind a large framework.

### Layer D: educational views

Notebooks, exercises, slides, and reports consume the executable layers:

- notebooks explain and visualize; they do not contain the only working implementation;
- exercise code contains deliberate, bounded gaps;
- solutions reuse the reference implementation or are generated from a common source;
- slides import tested snippets or cite exact files and commands;
- recorded metrics and traces back resource-dependent explanations.

Prefer text-based notebook sources, such as Jupytext percent-format files, for review and Git blame. Generate participant-facing `.ipynb` files as release artifacts after verifying that the Jupyter workflow is comfortable for beginners.

## Phased implementation

### Phase 0: contract and clean skeleton

Create `course/` with:

- the rough three-day outline;
- pinned image reference and digest;
- `pyproject.toml`, `uv.lock`, and environment bootstrap;
- environment doctor;
- Slidev skeleton;
- format, naming, configuration, and artifact conventions.

Acceptance:

- the old course remains untouched;
- the environment doctor runs on the development host and inside the container;
- a Jupyter kernel starts from the course environment;
- an empty Slidev deck builds;
- one sample move or copy retains inspectable Git lineage before adaptation.

### Phase 1: golden vertical slice

Build one small, complete path before broad migration:

1. pre-course tensor/device diagnostic;
2. Day 1 MLP training exercise;
3. reference solution;
4. deterministic evaluation;
5. machine-readable metrics;
6. one thin notebook;
7. a small set of slides that execute the same commands.

Acceptance:

- CPU and one-GPU smoke runs succeed;
- the notebook executes from a clean kernel;
- the solution produces a deterministic learning signal;
- the slide build succeeds;
- any reused same-repository material has an inspectable move/copy commit.

This slice validates the directory structure, environment, exercise/solution convention, notebook authoring approach, and code-to-slide workflow before more material is ported.

### Phase 2: complete the beginner path

Add the remaining preparation and Day 1 material:

- Python/NumPy refresher;
- tensors and autograd;
- datasets and loaders;
- MLP and CNN exercises;
- debugging and evaluation;
- instructor notes and expected timings.

Port only material that is accurate and pedagogically useful. Rebuild stale examples rather than preserving their accidental structure.

Acceptance:

- a beginner can complete the preparation without cluster assistance;
- every Day 1 exercise has a verified solution and bounded runtime;
- the Day 1 sequence passes an instructor dry run.

### Phase 3: training and transformer bridge

Add Day 2:

- optimizer and regularization experiments;
- mixed precision and checkpointing;
- transfer-learning/fine-tuning exercise;
- attention and transformer construction;
- small language-model training/evaluation;
- first profiler report;
- DDP introduction.

This phase establishes the model and data contracts later used on Day 3.

Acceptance:

- the transformer produces matching reference outputs before and after checkpoint reload;
- the semantic configuration shows a meaningful loss/generation signal;
- one-rank and DDP runs agree within defined numerical tolerance;
- profiler output and structured reports are generated automatically.

### Phase 4: large-scale core

Implement the Day 3 reference harness:

- FSDP2, TP, PP, MoE, and EP;
- composed meshes;
- standard run modes: benchmark, profile, inspect, evaluate;
- per-rank metrics and multi-rank traces;
- Slurm jobs for one and two nodes;
- Triton kernel exercise and optional Mojo comparison;
- prepared traces and metric bundles for fallback use.

Acceptance:

- each strategy matches the single-rank reference on a small deterministic case;
- one-node runs succeed in the canonical container;
- the planned two-node composition path is tested or explicitly replaced by a recorded trace fallback;
- benchmark and profile modes are demonstrably separated;
- standard reports are produced without manual log scraping.

### Phase 5: material expansion and selective porting

With the code stable enough to teach:

- fill in slides for preparation and all three days;
- port diagrams, explanations, and exercises selectively;
- add instructor notes, exercise timing, expected outputs, and failure hints;
- keep visible placeholders only in sections explicitly outside the current rough-draft milestone;
- preserve third-party notices and source/version references where Git history alone is insufficient.

Slides should follow the executable sequence. Do not port a slide merely because it existed in the previous deck.

### Phase 6: integration and pilot

- execute every notebook from a clean kernel;
- build and export all slide decks;
- run preparation and Day 1 on a clean participant account;
- run Day 2 on one GPU and one node;
- run Day 3 on one node and the planned multi-node allocation;
- measure actual exercise duration, profiler overhead, trace size, and queue overhead;
- conduct instructor dry runs and revise the timetable;
- freeze release images, locks, datasets, checkpoints, and checksums.

The old course becomes removable only after this phase, a history-lineage check, and a third-party notice review.

## Rough-draft milestone

The first rough draft is not the complete course. It should include:

- the new directory and reproducible environment;
- the complete pre-course and Day 1 skeleton;
- one finished golden vertical slice;
- Day 2 and Day 3 section outlines;
- a runnable single-GPU transformer path;
- placeholders identifying missing lessons by learning objective, owner, prerequisite, and expected duration;
- a migration inventory marking old artifacts as port, adapt, reference, or retire.

A placeholder may exist in the rough plan or slide outline. Runnable exercises must not contain fake implementations or unmarked missing behavior.

## Migration workflow

For each old artifact:

1. identify the learning objective it supports;
2. inspect its path-specific Git authors and external sources;
3. classify it as `port`, `adapt`, `reference`, or `retire`;
4. move it in a dedicated commit, or make and commit an exact copy if the old course still needs it;
5. verify rename/copy detection before editing;
6. update it to the new executable contracts in a later commit;
7. run the associated exercise or build;
8. review pedagogy, accessibility, citations, and third-party notices.

Suggested initial classifications:

- MLP and autograd notebooks: adapt into the Day 1 explicit-loop path;
- CIFAR-10 CNN exercise: adapt after the golden slice;
- training/regularization material: adapt for Day 2;
- transformer attention and feed-forward notebooks: adapt into the decoder-only model progression;
- distributed launcher, logging, and plotting code: reference and selectively port; do not copy obsolete FSDP1 or timing behavior;
- distributed slide diagrams: port selectively after terminology and Alps topology are corrected;
- Hugging Face task pipelines and BERT/SQuAD material: optional applied modules, not part of the scaling core;
- existing Slidev theme and generic components: likely direct moves or exact-copy commits after licence and external-source review.

## First implementation increment

The first coding increment should be deliberately small:

1. create `course/` and demonstrate one history-preserving sample port;
2. pin the container by tag and digest;
3. create the `uv` environment without replacing image-provided PyTorch;
4. implement `environment/doctor.py`;
5. add a CPU/single-GPU tensor diagnostic;
6. add a minimal explicit MLP training/evaluation program;
7. write one smoke command and one behavioral correctness check;
8. create one notebook view and one short Slidev section from that code;
9. run the complete slice in the canonical image.

Do not start by porting the full slide deck or all notebooks. The first slice should expose flaws in the environment, package boundaries, notebook workflow, Git migration convention, and beginner-facing API while changes remain cheap.

## Decisions to resolve during the first increment

- final course title and dates;
- exact participant prerequisites and preparation deadline;
- whether generated `.ipynb` files are committed or release-only artifacts;
- canonical Alps container launch configuration and image digest;
- supported Jupyter entry path for participant accounts;
- dataset and checkpoint storage locations;
- participant/instructor separation for solutions;
- expected team size and one-/two-node allocation;
- scope of optional BERT/SQuAD, inference-pipeline, agent, and Mojo material.
