import numpy as np

from data_provider.dataloader import WindowedTimeSeriesDataset, build_dataloader
from data_provider.preprocessing import StandardScaler


def test_window_shapes_and_unmasked_target() -> None:
    values = np.arange(80, dtype=np.float32).reshape(40, 2)
    mask = np.ones((8, 2), dtype=bool)
    mask[-3:, :] = False
    dataset = WindowedTimeSeriesDataset(
        values,
        input_length=8,
        prediction_length=4,
        mask_factory=lambda _index, _features: mask,
    )
    sample = dataset[0]
    assert sample["history"].shape == (8, 2)
    assert sample["observed_mask"].shape == (8, 2)
    assert sample["target"].shape == (4, 2)
    assert np.all(sample["history"].numpy()[-3:] == 0.0)
    np.testing.assert_array_equal(sample["target"].numpy(), values[8:12])


def test_observed_zero_is_not_missing() -> None:
    values = np.zeros((20, 2), dtype=np.float32)
    dataset = WindowedTimeSeriesDataset(values, 8, 4)
    sample = dataset[0]
    assert sample["observed_mask"].bool().all()


def test_seeded_shuffle_is_reproducible() -> None:
    values = np.arange(120, dtype=np.float32).reshape(60, 2)
    dataset = WindowedTimeSeriesDataset(values, 8, 4)
    first = next(iter(build_dataloader(dataset, 4, shuffle=True, seed=3)))
    second = next(iter(build_dataloader(dataset, 4, shuffle=True, seed=3)))
    assert np.array_equal(first["history"].numpy(), second["history"].numpy())


def test_scaler_uses_supplied_training_values() -> None:
    train = np.array([[0.0], [2.0]], dtype=np.float32)
    test = np.array([[100.0]], dtype=np.float32)
    scaler = StandardScaler().fit(train)
    assert float(scaler.transform(test)[0, 0]) == 99.0
