import numpy as np

from data_provider.masking import (
    generate_mask_bank,
    mixed_test_mask,
    test_block_mask,
    training_mask,
    validation_mask,
)


def test_training_mask_matches_reported_bounds() -> None:
    sample = training_mask(96, 5, np.random.default_rng(1))
    assert 10 <= sample.block_length <= 48
    assert 0.1 <= sample.point_missing_rate <= 0.3
    block = sample.observed[
        sample.block_start : sample.block_start + sample.block_length
    ]
    assert not block.any()


def test_validation_mask_has_final_shared_block() -> None:
    sample = validation_mask(96, 3, np.random.default_rng(2))
    assert sample.block_start == 46
    assert not sample.observed[-50:].any()


def test_block_mask_only_masks_history_suffix() -> None:
    sample = test_block_mask(720, 4, 200)
    assert sample.observed[:520].all()
    assert not sample.observed[520:].any()


def test_mixed_mask_does_not_resample_inside_block() -> None:
    sample = mixed_test_mask(96, 2, 20, 0.25, np.random.default_rng(3))
    assert not sample.observed[-20:].any()


def test_mask_bank_is_reproducible_per_seed() -> None:
    first = generate_mask_bank("validation", 4, 96, 2, seed=3)
    second = generate_mask_bank("validation", 4, 96, 2, seed=3)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left.observed, right.observed)


def test_mask_bank_rejects_unknown_mode() -> None:
    try:
        generate_mask_bank("test", 1, 96, 2, seed=1)
    except ValueError as error:
        assert "mode" in str(error)
    else:
        raise AssertionError("Unknown mask-bank mode should fail.")
