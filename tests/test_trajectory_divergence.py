import numpy as np
import pytest

from src.phaseII_full_paired_v2 import trajectory_divergence


def test_identical_histories_have_no_divergence():
    tokens = np.array([1, 2, 3, 4])
    assert trajectory_divergence(tokens, tokens, 3) == (True, -1, 0)


def test_divergence_at_current_step_does_not_change_pre_step_history():
    donor = np.array([1, 2, 8, 4])
    recipient = np.array([1, 2, 3, 4])
    assert trajectory_divergence(donor, recipient, 2) == (True, 2, 0)


def test_divergence_age_counts_steps_after_first_mismatch():
    donor = np.array([1, 8, 3, 4, 5])
    recipient = np.array([1, 2, 3, 4, 5])
    assert trajectory_divergence(donor, recipient, 4) == (False, 1, 3)


def test_divergence_rejects_mismatched_shapes_and_invalid_step():
    with pytest.raises(ValueError, match="identical shapes"):
        trajectory_divergence(np.array([1, 2]), np.array([1]), 1)
    with pytest.raises(ValueError, match="step"):
        trajectory_divergence(np.array([1, 2]), np.array([1, 2]), 3)
