# Rough Course Outline

Sections and durations remain provisional until the executable exercises have been timed.

## Runtime progression

- Preparation and Day 1 require only the portable `cpu` profile; `cuda` and `alps-gh200` may accelerate the same material.
- Day 2 uses `cuda` for GPU-specific training and profiling. Conceptual work and every required exercise retain a bounded CPU path or recorded evidence.
- Day 3 uses `alps-gh200` for GH200, NCCL, topology, and multi-GPU or multi-node work. Local profiles support preparation and analysis, not substitutes for measured Alps runs.
- Each lesson states its minimum profile and hardware before its commands. A visible GPU is a lesson requirement, separate from selecting a CUDA-capable profile.

## Preparation

Outcome: participants can enter Day 1 with a working environment and manipulate tensors confidently.

- account, repository, Jupyter, terminal, and Slurm orientation;
- focused Python and NumPy refresher;
- tensor shapes, indexing, dtype, and device;
- CPU versus GPU execution;
- environment diagnostic and self-check.

## Day 1 — From tensors to a trained model

Outcome: participants can implement, evaluate, and diagnose a small training loop.

- supervised-learning vocabulary;
- modules, parameters, forward passes, and autograd;
- explicit optimization loop;
- datasets, splits, batching, and deterministic evaluation;
- device, shape, gradient, and train/eval debugging;
- MLP exercise;
- compact CNN/image-classification exercise;
- loss curves, validation behavior, and overfitting.

## Day 2 — Robust training and the bridge to scale

Outcome: participants have a working decoder-only language-model path and understand why it must be distributed.

- optimizers, learning rates, regularization, initialization, and normalization;
- mixed precision, checkpointing, and reproducibility;
- transfer learning or fine-tuning;
- tokens, embeddings, attention, and transformer blocks;
- next-token training and evaluation;
- single-GPU profiling and structured reports;
- DDP and an optional FSDP2 introduction.

## Day 3 — Large models on Alps

Outcome: participants can select and defend a parallel training strategy using correctness and performance evidence.

- scaling arithmetic and Alps topology;
- DDP and FSDP2 recap;
- tensor, sequence/context, and pipeline parallelism;
- MoE routing and expert parallelism;
- composable device meshes;
- profiling, communication overlap, and imbalance;
- Triton and optional Mojo kernel comparison;
- end-to-end strategy challenge.

The detailed Day 3 design remains in `../course-design/day-3-large-scale-models.md`.
