# A compact CNN, end to end

Day 1 · 55 minutes · CPU or one GPU · offline

Adapted from the original CNN slides and CIFAR-10 exercise. The bounded version removes downloads and preserves the same convolutional learning path.

---

# The offline classification problem

<<< @/lessons/day1/cnn/training.py#stripe-images

- class 0: noisy vertical stripe;
- class 1: noisy horizontal stripe;
- fixed seed and balanced labels;
- 192 training and 64 held-out images by default.

The data are deliberately simple so the exercise tests architecture and training mechanics, not network access.

---

# Construct the model

<<< @/lessons/day1/cnn/solution.py#cnn-model-solution

```text
[N, 1, 8, 8] → conv → [N, 4, 8, 8]
              → pool → [N, 4, 4, 4]
              → flatten → [N, 64]
              → linear → [N, 2]
```

The final layer returns logits. `CrossEntropyLoss` owns the softmax transformation.

---

# Parameter sharing changes the model

A dense layer learns a separate weight for every input-output pair. A convolution reuses one small kernel across positions.

```text
Conv2d weights = out_channels × in_channels × kernel_height × kernel_width
```

Local connectivity and shared weights encode an image-specific inductive bias.

---

# Pooling trades detail for invariance

`MaxPool2d(2)` halves each spatial dimension and keeps the strongest response in each $2\times2$ region.

```text
8 × 8 → 4 × 4
```

The operation has no trainable parameters, but it changes the input size required by the linear classifier.

---

# The explicit mini-batch loop

<<< @/lessons/day1/cnn/training.py#cnn-training-loop

The MLP and CNN use the same state transitions:

```text
train → zero gradients → forward → loss → backward → update
eval  → no gradients   → forward → held-out metrics
```

The CNN adds batches and image-shaped inputs; it does not change the optimization contract.

---

# Bounded exercise

Open `lessons/day1/cnn/exercise.py` and implement `build_cnn`.

```bash
course exercise cnn --device cpu
course train cnn --device cpu --json
```

Expected CPU runtime: under one minute. Success requires at least 95% held-out accuracy and a substantial evaluation-loss decrease.
