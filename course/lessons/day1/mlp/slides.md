# Learning path

By the end of this lesson, you can explain and implement:

```text
tensors → dataset → model → logits → loss → gradients → update → evaluation
```

- every tensor has a shape, dtype, and device;
- the MLP maps two input features to two class scores;
- the optimization loop changes parameters from evidence;
- held-out accuracy checks whether learning generalizes.

---

# Select a device explicitly

<<< @/lessons/prep/tensor_device/reference.py#device-selection

`auto` is convenient for examples. Exercises that compare CPU and GPU behavior should request the device explicitly.

---

# One device invariant

```python
features = split.train_features.to(device)
targets = split.train_targets.to(device)
model = MLP().to(device)
logits = model(features)
```

<Admonition title="Required" color="sky-light">
The model parameters and every tensor used by an operation must be on compatible devices.
</Admonition>

A mismatch fails before training; it is not a performance detail.

---

# The supervised-learning contract

For $N$ samples and two classes:

```text
features   [N, 2]   floating-point coordinates
parameters          weights and biases learned by the model
logits     [N, 2]   one raw score per class
targets    [N]      integer class indices, 0 or 1
loss       []       one scalar measuring prediction error
```

Training changes the parameters. The features and targets remain evidence.

---

# Generate a deterministic offline problem

<<< @/lessons/day1/mlp/reference.py#xor-generation

- fixed generator seed;
- four balanced clusters;
- deterministic shuffle;
- no network or dataset cache.

---

# Why one linear layer is insufficient

```text
class 0          class 1
(-1, -1)         (-1,  1)
( 1,  1)         ( 1, -1)
```

No single straight line separates the classes.

The hidden layer transforms the coordinates; `ReLU` introduces the nonlinearity needed to represent two separating boundaries.

---

# The complete model

<<< @/lessons/day1/mlp/reference.py#mlp-model

Shape flow:

```text
[N, 2] → Linear → [N, 16] → ReLU → [N, 16] → Linear → [N, 2]
```

---

# Logits first, probabilities later

`MLP.forward` returns raw logits. It does **not** apply softmax.

```python
logits = model(features)  # [N, 2]
loss = loss_function(logits, targets)  # scalar
predictions = logits.argmax(dim=1)  # [N]
```

`CrossEntropyLoss` combines log-softmax and negative log-likelihood internally. Applying softmax in the model would duplicate part of that operation and weaken numerical stability.

---

# Choose the learning components

<<< @/lessons/day1/mlp/reference.py#training-components

- the model owns trainable parameters;
- cross-entropy measures classification error;
- SGD applies parameter updates using gradients;
- the learning rate controls update size.

---

# The explicit optimization loop

<<< @/lessons/day1/mlp/reference.py#explicit-training-loop

The order is part of the algorithm. Each iteration uses the current parameters and leaves gradients ready for exactly one update.

---

# What each operation changes

| Operation | Observable effect |
| --- | --- |
| `zero_grad` | clears gradients left by the previous iteration |
| `model(features)` | builds a computation graph and produces logits |
| `loss_function(...)` | reduces prediction errors to one scalar |
| `loss.backward()` | accumulates gradients in each parameter |
| `optimizer.step()` | updates parameters using those gradients |

Omitting `zero_grad` silently accumulates gradients across iterations.

---

# Evaluation is a different state

<<< @/lessons/day1/mlp/reference.py#evaluation

- `model.eval()` selects evaluation behavior;
- `torch.no_grad()` avoids building a gradient graph;
- loss measures error magnitude;
- accuracy measures the fraction of correct classes.

Evaluation uses held-out samples that did not drive parameter updates.

---

# A deterministic learning signal

```bash
course train mlp --device cpu --seed 7 --json
```

Reference CPU result:

```text
training loss       0.765172 → 0.176132
evaluation accuracy 1.000000
status              pass
```

The contract requires final loss below 35% of initial loss and evaluation accuracy of at least 95%.

---

# One lesson, all related artifacts

```text
lessons/day1/mlp/
├── reference.py   complete executable behavior
├── exercise.py    bounded participant gap
├── solution.py    complete exercise answer
├── notebook.py    reviewable Jupytext source
├── notebook.ipynb participant notebook
└── slides.md      presentation view
```

The notebook and slides use `reference.py`. The solution repeats only the five operations participants must implement, so the answer remains directly inspectable.

---
layout: center
---

# Implement one optimization step

Open `lessons/day1/mlp/exercise.py`.

Replace one `NotImplementedError` with the five ordered operations, then run:

```bash
course exercise mlp --device cpu
```

Require:

```text
evaluation accuracy ≥ 95%
Status: PASS
```
