import numpy as np
import pytest

from src.phaseII_full_paired_v2 import top_p_set


def test_top_p_keeps_minimal_prefix_and_includes_exact_equality():
    idx, q = top_p_set(np.array([0.5, 0.3, 0.2]), 0.8)
    assert idx.tolist() == [0, 1]
    assert np.allclose(q, [0.5 / 0.8, 0.3 / 0.8])


def test_top_p_keeps_first_crossing_token():
    idx, q = top_p_set(np.array([0.6, 0.25, 0.15]), 0.7)
    assert idx.tolist() == [0, 1]
    assert np.isclose(q.sum(), 1.0)


def test_top_p_one_keeps_full_support():
    idx, q = top_p_set(np.array([0.4, 0.3, 0.2, 0.1]), 1.0)
    assert len(idx) == 4
    assert np.allclose(q, [0.4, 0.3, 0.2, 0.1])


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.1, np.nan])
def test_top_p_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="top_p"):
        top_p_set(np.array([0.6, 0.4]), threshold)


def test_top_p_rejects_unnormalized_probabilities():
    with pytest.raises(ValueError, match="normalized"):
        top_p_set(np.array([0.6, 0.3]), 0.8)
