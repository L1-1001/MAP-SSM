from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(relative_path: str) -> dict:
    return yaml.safe_load((ROOT / relative_path).read_text(encoding="utf-8"))


def test_model_config_matches_reported_hyperparameters() -> None:
    config = load_yaml("configs/model.yaml")
    model = config["model"]
    assert model == {
        "name": "MAP-SSM",
        "layers": 2,
        "model_dimension": 128,
        "state_dimension": 16,
        "prototype_bank_size": 32,
        "prototype_query_dimension": 32,
        "dropout": 0.1,
    }
    assert config["discretization"] == {
        "method": "zero_order_hold",
        "minimum_step": 0.0001,
        "maximum_step": 0.1,
        "missing_step": 0.0001,
    }


def test_protocol_config_matches_public_missingness_description() -> None:
    config = load_yaml("configs/protocol.yaml")
    assert config["training_missingness"]["block_length_min"] == 10
    assert config["training_missingness"]["block_length_max"] == {96: 48, 720: 250}
    assert config["training_missingness"]["point_missing_rate_uniform"] == [0.1, 0.3]
    assert config["validation_missingness"]["final_block_length"] == 50
    assert config["validation_missingness"]["remaining_point_missing_rate"] == 0.15
    assert config["matched_accuracy_seeds"] == [1, 2, 3]
    assert config["profiling_seeds"] == [2024, 2025, 2026]


def test_dataset_configs_use_reported_split_families() -> None:
    ett_names = ("etth1", "etth2", "ettm1", "ettm2")
    other_names = ("weather", "electricity", "traffic", "exchange", "solar")
    for name in ett_names:
        assert load_yaml(f"configs/data/{name}.yaml")["split_ratios"] == [0.6, 0.2, 0.2]
    for name in other_names:
        assert load_yaml(f"configs/data/{name}.yaml")["split_ratios"] == [0.7, 0.1, 0.2]
