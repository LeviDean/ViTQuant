from vitquant.quant.qconfig import QConfig, TensorQConfig, qconfig_from_dict


def test_defaults_match_spec():
    qc = QConfig()
    assert qc.weight.symmetric and qc.weight.per_channel and qc.weight.bits == 8
    assert not qc.activation.symmetric and not qc.activation.per_channel


def test_from_dict():
    qc = qconfig_from_dict({
        "weight": {"bits": 8, "symmetric": True, "per_channel": True, "observer": "minmax"},
        "activation": {"bits": 8, "symmetric": False, "per_channel": False, "observer": "moving_avg"},
    })
    assert qc.weight.per_channel is True
    assert qc.activation.observer == "moving_avg"


def test_frozen():
    import dataclasses, pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        QConfig().weight.bits = 4


def test_load_config_yaml(tmp_path):
    from pathlib import Path
    from vitquant.utils.config import load_config

    p = tmp_path / "c.yaml"
    p.write_text("model:\n  name: deit_tiny_patch16_224\ndevice: auto\n")
    cfg = load_config(p)
    assert cfg["model"]["name"] == "deit_tiny_patch16_224"


def test_shipped_configs_parse():
    from pathlib import Path
    from vitquant.utils.config import load_config
    from vitquant.quant.qconfig import qconfig_from_dict

    for name in ("deit_tiny", "vit_base"):
        cfg = load_config(Path("configs") / f"{name}.yaml")
        for key in ("model", "data", "quant", "eval", "benchmark", "device", "output_dir"):
            assert key in cfg, f"{name}.yaml missing '{key}'"
        qconfig_from_dict(cfg["quant"])  # must build without error
