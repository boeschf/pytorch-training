# Datasets and DataLoaders

Preparation · 40 minutes · CPU only · offline

Adapted from the Dataset/DataLoader section of the original PyTorch overview.

---

# Separate one sample from one epoch

```text
Dataset[index]      → one (features, target) pair
DataLoader(dataset) → batches covering an epoch
```

The dataset owns indexing. The loader owns batch size, shuffle order, and worker policy.

---

# The Dataset protocol

```python
class PairDataset(Dataset):
    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]
```

Keep paired tensors aligned: every index must identify the same sample in both.

---

# A deterministic loader

```python
generator = torch.Generator().manual_seed(7)
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    generator=generator,
    num_workers=0,
)
```

Seed the loader's generator independently. The portable preparation path uses no worker processes.

---

# Inspect before training

For every first batch, check:

```text
feature shape   [batch, features]
target shape    [batch]
feature dtype   floating point
target dtype    integer class index for classification
sample count    complete epoch, including a smaller final batch
```

A loader bug is easier to fix before an optimizer obscures it.

---

# Bounded exercise

Open `lessons/prep/datasets_loaders/exercise.py` and implement `__len__` and `__getitem__`.

```bash
python lessons/prep/datasets_loaders/exercise.py
course prep data-loader
```

The check verifies indexing, batching, complete coverage, and repeatable shuffling.
