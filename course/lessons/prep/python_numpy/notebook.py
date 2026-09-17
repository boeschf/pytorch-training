# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: "CSCS PyTorch Course \u2014 CPU"
#     language: python
#     name: cscs-pytorch-course-cpu
# ---

# %% [markdown]
# # Python and NumPy refresher
#
# Goal: inspect a small table, use axis-aware reductions, and standardize each
# feature with broadcasting. This preparation path is CPU-only and offline.

# %%
import numpy as np

from lessons.prep.python_numpy.reference import make_measurements
from lessons.prep.python_numpy.solution import python_numpy_report, standardize_columns

# %%
measurements = make_measurements(samples=12, seed=7)
print("shape:", measurements.shape)
print("first row:", measurements[0])
print("column means:", measurements.mean(axis=0))

# %% [markdown]
# A slice such as `measurements[:, 1]` selects one feature across every sample.
# `axis=0` reduces the sample dimension and retains one statistic per feature.

# %%
means = measurements.mean(axis=0)
deviations = measurements.std(axis=0)
normalized = standardize_columns(measurements)
print("broadcast operands:", measurements.shape, means.shape, deviations.shape)
print("normalized means:", normalized.mean(axis=0))
print("normalized standard deviations:", normalized.std(axis=0))
assert np.allclose(normalized.mean(axis=0), 0.0)
assert np.allclose(normalized.std(axis=0), 1.0)

# %%
report = python_numpy_report()
print(report)
assert report["status"] == "pass"
