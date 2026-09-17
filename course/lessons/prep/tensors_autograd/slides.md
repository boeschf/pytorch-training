# Tensors and autograd

Preparation · 45 minutes · CPU only; optional GPU · offline

Adapted from the original `slides/src/1.2-pytorch-overview/section-slides.md` overview.

---

# A tensor has values and metadata

```python
values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
values.shape   # torch.Size([2, 2])
values.dtype   # torch.float32
values.device  # device(type='cpu')
```

Every operation must respect shape, dtype, and device.

---

# Indexing follows the NumPy model

```python
values[0]       # first row
values[:, 1]    # second column
values[:1, :]   # first row, rank preserved
```

`values[0]` has shape `[2]`; `values[:1]` has shape `[1, 2]`. That difference changes later broadcasting.

---

# Device movement is explicit

<<< @/lessons/prep/tensor_device/reference.py#device-selection

```python
values = values.to(device)
model = model.to(device)
```

An operation cannot combine CPU and CUDA tensors.

---

# Autograd records a graph

```python
inputs = torch.tensor([1.0, 2.0], requires_grad=True)
loss = (inputs - targets).square().sum()
loss.backward()
print(inputs.grad)
```

For $L = \sum_i(x_i-y_i)^2$, autograd should produce $\partial L/\partial x_i = 2(x_i-y_i)$.

---

# Gradient lifecycle

- forward operations build the current graph;
- `backward()` accumulates gradients in leaf tensors;
- optimizers read parameter gradients;
- training clears old gradients before the next backward pass;
- evaluation disables gradient recording with `torch.no_grad()` or `torch.inference_mode()`.

Use `.detach()` only when a value must leave the graph intentionally.

---

# Bounded exercise

Open `lessons/prep/tensors_autograd/exercise.py` and implement one scalar squared-error objective.

```bash
python lessons/prep/tensors_autograd/exercise.py
course prep tensors-autograd --device cpu
```

The solution is verified against the analytical gradient, not merely checked for a nonzero result.
