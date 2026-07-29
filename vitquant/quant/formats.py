"""Non-uniform quantization grids (beyond the default integer lattice).

FP4 (E2M1): 1 sign + 2 exponent + 1 mantissa bit — 15 distinct values with
magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6}. The grid is denser near zero and
sparser toward the clip point, which matches bell-shaped weight/activation
distributions better than INT4's uniform lattice and is the element format
of NVFP4/MXFP4-style block floating point. Here it is simulated the same way
as the integer fake-quant: x -> scale * nearest_fp4(x / scale), with the
scale calibrated so the observed amax maps to the largest magnitude (6).

Being sign-symmetric with no zero offset, FP4 is inherently symmetric; the
asymmetric/zero-point machinery does not apply. AdaRound's floor+offset
parameterization assumes a uniform step and is skipped for FP4 tensors."""
import torch

FP4_E2M1 = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
                         0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_FP4_MIDPOINTS = (FP4_E2M1[1:] + FP4_E2M1[:-1]) / 2
FP4_AMAX = 6.0


def fp4_round(y: torch.Tensor) -> torch.Tensor:
    """Map each element of y to the nearest FP4 (E2M1) grid value. Values
    beyond +-6 clamp to the end of the grid."""
    idx = torch.bucketize(y, _FP4_MIDPOINTS.to(y.device, y.dtype))
    return FP4_E2M1.to(y.device, y.dtype)[idx]
