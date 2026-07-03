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
