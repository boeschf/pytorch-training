# Python and NumPy refresher

Preparation · 35 minutes · CPU only · offline

By the end, you can read array shapes, slice samples and features, reduce along an axis, and use broadcasting instead of a Python loop.

---

# Samples and features

```text
values.shape == (samples, features)
values[0]          one sample
values[:, 0]       one feature across all samples
values[:4, :2]     a batch and feature subset
```

Shape is part of the program's contract, not incidental metadata.

---

# Reductions need an axis

```python
column_means = values.mean(axis=0)  # shape: (features,)
row_means = values.mean(axis=1)     # shape: (samples,)
```

Ask which dimension disappears. For feature normalization, reduce the sample dimension.

---

# Broadcasting replaces the loop

```python
means = values.mean(axis=0)
deviations = values.std(axis=0)
normalized = (values - means) / deviations
```

```text
(samples, features)
          (features)  → broadcast across samples
```

The expression returns a new array; the input stays unchanged.

---

# Bounded exercise

Open `lessons/prep/python_numpy/exercise.py` and implement `standardize_columns`.

```bash
python lessons/prep/python_numpy/exercise.py
course prep python-numpy
```

The self-check requires zero column means, unit column standard deviations, the original shape, and an unmodified input.
