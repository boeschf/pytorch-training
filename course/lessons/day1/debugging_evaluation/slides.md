# Debugging and evaluation

Day 1 · 40 minutes · CPU only · offline

Adapted from `slides/src/3.1-more_on_training/section-slides.md`, with a narrower Day 1 focus on observable failures and held-out evidence.

---

# Inspect invariants before hyperparameters

For one batch, print and verify:

```text
features  [batch, channels, height, width]  floating point
labels    [batch]                           torch.int64
model and features                          same device
training loop                               model.train()
evaluation                                  model.eval() + no gradients
```

A learning-rate change cannot repair a shape, dtype, device, or state error.

---

# Make errors actionable

`classification_issues` reports every visible contract violation at once:

```text
features must have shape [batch, channels, height, width]
features must use a floating-point dtype
CrossEntropyLoss targets must use torch.int64
```

```bash
course exercise debugging-evaluation
```

The exercise repairs a deliberately broken batch, then runs a real forward evaluation.

---

# Training loss is not evaluation

<img src="/legacy/debugging/loss.png" style="height: 300px; margin: auto" alt="Loss curve illustration from the legacy training slides" />

- optimization loss answers whether updates fit training evidence;
- held-out loss and accuracy answer whether the result generalizes;
- always record a pre-training baseline;
- compare the same deterministic split across runs.

---

# Read the two curves together

```text
train ↓, eval ↓       learning and generalization
train ↓, eval ↑       overfitting or split mismatch
both flat             no update, bad scale, or insufficient model
erratic                unstable updates or nondeterministic input order
```

Inspect incorrect examples before adding model complexity.

---

# Evaluation state is explicit

```python
model.eval()
with torch.no_grad():
    logits = model(features)
    predictions = logits.argmax(dim=1)
```

`model.eval()` changes layer behavior. `torch.no_grad()` changes autograd recording. They solve different problems and are both required here.

---

# Day 1 completion evidence

```bash
course prep python-numpy
course prep tensors-autograd --device cpu
course prep data-loader
course train mlp --device cpu --json
course train cnn --device cpu --json
course diagnose evaluation
```

Each command is bounded, deterministic, and independent of a cluster or dataset download.
