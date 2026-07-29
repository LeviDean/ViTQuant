from dataclasses import dataclass, field


@dataclass(frozen=True)
class TensorQConfig:
    """Quantization config for one tensor kind (weight or activation)."""
    bits: int = 8
    symmetric: bool = True
    per_channel: bool = False
    ch_axis: int = 0
    observer: str = "minmax"  # minmax | moving_avg | percentile
    percentile: float = 0.999  # only used by PercentileObserver
    # "int": uniform integer lattice (default). "fp4": E2M1 floating-point grid
    # (bits must be 4; inherently symmetric, zero_point unused; AdaRound n/a).
    fmt: str = "int"


@dataclass(frozen=True)
class QConfig:
    """Default per spec: weight per-channel symmetric, activation per-tensor asymmetric."""
    weight: TensorQConfig = field(
        default_factory=lambda: TensorQConfig(symmetric=True, per_channel=True))
    activation: TensorQConfig = field(
        default_factory=lambda: TensorQConfig(symmetric=False, per_channel=False,
                                              observer="moving_avg"))


def qconfig_from_dict(d: dict) -> QConfig:
    return QConfig(weight=TensorQConfig(**d["weight"]),
                   activation=TensorQConfig(**d["activation"]))
