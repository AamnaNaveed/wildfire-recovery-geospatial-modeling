"""
Minimal unit tests for index calculations.

These tests verify that NDVI and NBR formulas are implemented correctly
on synthetic inputs. They do not test the full pipeline.
"""

import numpy as np
import pytest


def ndvi(red, nir):
    return (nir - red) / (nir + red)


def nbr(nir, swir):
    return (nir - swir) / (nir + swir)


def test_ndvi_dense_vegetation():
    red = np.array([0.05], dtype="float32")
    nir = np.array([0.50], dtype="float32")
    result = ndvi(red, nir)
    assert 0.8 < result[0] < 0.95


def test_ndvi_bare_soil():
    red = np.array([0.30], dtype="float32")
    nir = np.array([0.35], dtype="float32")
    result = ndvi(red, nir)
    assert 0.0 < result[0] < 0.15


def test_nbr_vegetation():
    nir = np.array([0.40], dtype="float32")
    swir = np.array([0.10], dtype="float32")
    result = nbr(nir, swir)
    assert 0.5 < result[0] < 0.7


def test_nbr_burned():
    nir = np.array([0.20], dtype="float32")
    swir = np.array([0.35], dtype="float32")
    result = nbr(nir, swir)
    assert result[0] < 0.0


def test_dnbr_positive_for_burn():
    pre_nbr = 0.6
    post_nbr = 0.2
    dnbr = pre_nbr - post_nbr
    assert dnbr > 0.0


def test_dnbr_range():
    # dNBR should be bounded between -2 and 2 in the raw formula,
    # but our pipeline clips to [-1, 1].
    dnbr_raw = 1.5
    dnbr_clipped = np.clip(dnbr_raw, -1.0, 1.0)
    assert dnbr_clipped == 1.0