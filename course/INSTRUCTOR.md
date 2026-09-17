# Phase 2 Instructor Guide

The beginner path is designed for the portable `cpu` profile. No preparation or Day 1 activity requires a cluster, visible GPU, network download, or pre-populated dataset cache.

## Preparation before the course

| Lesson | Participant time | Completion evidence |
| --- | ---: | --- |
| Python and NumPy refresher | 35 min | `course prep python-numpy` reports `PASS` |
| Tensors and autograd | 45 min | `course prep tensors-autograd --device cpu` reports the analytical gradient and `PASS` |
| Datasets and DataLoaders | 40 min | `course prep data-loader` reports a complete deterministic epoch and `PASS` |

Ask participants to run the adjacent `exercise.py` before consulting `solution.py`. The `course prep` commands execute the completed reference paths, so they distinguish an environment problem from an unfinished participant exercise.

## Day 1 schedule

| Segment | Time | Instructor checkpoint |
| --- | ---: | --- |
| Environment and preparation recap | 20 min | CPU profile active; all three preparation commands pass |
| Supervised-learning and autograd bridge | 30 min | participants can name features, logits, targets, loss, and gradients |
| Dataset, split, and batching walkthrough | 35 min | first batch has expected shape and dtype |
| MLP model and explicit optimization step | 50 min | pre-training baseline recorded before updates |
| Break | 15 min | |
| MLP exercise | 35 min | held-out accuracy at least 95% |
| Images, convolution, pooling, and shape flow | 30 min | participants derive the 64-element flattened size |
| CNN exercise | 55 min | offline run finishes in under one minute and reaches at least 95% |
| Break | 15 min | |
| Debugging and deterministic evaluation | 40 min | batch repair removes every reported invariant violation |
| Day 1 dry run and recap | 25 min | `python tools/smoke_phase2.py --device cpu` prints `Phase 2 cpu: PASS` |

Total scheduled time: 5 hours 50 minutes including breaks. Leave remaining workshop time for setup variance, questions, and notebook exploration.

## Teaching cues

- Ask for shape, dtype, and device before discussing values.
- Keep the MLP and CNN optimization loops visible. Name every state transition: `train`, zero gradients, forward, loss, backward, update, `eval`, no gradients.
- Measure a pre-training baseline. A final metric without a baseline cannot establish a learning signal.
- Treat training loss and held-out metrics as different evidence.
- Use the synthetic stripe images for the required CNN path. The legacy CIFAR-10 material supplied the pedagogical structure, but its download and runtime are unsuitable for the bounded exercise.
- If a GPU is available, repeat one completed command with `--device cuda` only after the CPU result passes. GPU access is enrichment, not a prerequisite.

## Intervention order

When a participant is blocked, inspect in this order:

1. active profile and `course doctor`;
2. tensor shape and batch alignment;
3. feature and target dtypes;
4. model and tensor devices;
5. `train()` versus `eval()` state;
6. whether gradients are cleared, produced, and consumed;
7. only then learning rate or architecture.

Do not bypass an unfinished exercise by editing its harness. Compare with the adjacent `solution.py`, then restore the participant file and make the smallest conceptual correction.

## Instructor dry run

From a fresh CPU profile:

```bash
source .venv-cpu/bin/activate
course doctor --require-profile cpu
python tools/smoke_phase2.py --device cpu
jupytext --sync lessons/prep/*/notebook.py lessons/day1/*/notebook.py
course slides build
```

The smoke script executes preparation, both completed Day 1 models, and the debugging diagnostic. Notebook execution and the slide build remain separate checks because they exercise presentation surfaces rather than the importable training paths.
