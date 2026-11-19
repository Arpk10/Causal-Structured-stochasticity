import numpy as np
import pytest

from src.phaseII_full_paired_v2 import paired_state_diagnostics


def test_identical_pre_sampling_states_and_logits_have_zero_distance():
    state = np.array([[1.0, -2.0], [0.5, 0.25]])
    logits = np.array([0.1, -0.2, 1.5])
    result = paired_state_diagnostics(state, state.copy(), logits, logits.copy())
    assert result["prepatch_state_l2"] == 0.0
    assert result["prepatch_state_max_abs"] == 0.0
    assert result["prepatch_logit_max_abs"] == 0.0
    assert result["prepatch_js"] == pytest.approx(0.0, abs=1e-14)


def test_state_diagnostics_detect_a_real_difference():
    donor_state = np.array([[1.0, 0.0]])
    recipient_state = np.array([[0.0, 0.0]])
    donor_logits = np.array([2.0, 0.0])
    recipient_logits = np.array([0.0, 2.0])
    result = paired_state_diagnostics(
        donor_state, recipient_state, donor_logits, recipient_logits
    )
    assert result["prepatch_state_l2"] == pytest.approx(1.0)
    assert result["prepatch_state_max_abs"] == pytest.approx(1.0)
    assert result["prepatch_logit_max_abs"] == pytest.approx(2.0)
    assert result["prepatch_js"] > 0.0


def test_state_diagnostics_reject_shape_mismatch():
    with pytest.raises(ValueError, match="identical shapes"):
        paired_state_diagnostics(
            np.zeros((2, 2)), np.zeros((2, 3)), np.zeros(4), np.zeros(4)
        )
