import timm
import torch
from timm.models.vision_transformer import Attention

from vitquant.quant.convert import convert_vit
from vitquant.quant.fake_quant import FakeQuantize
from vitquant.quant.modules import QuantAttention, QuantConv2d, QuantLinear
from vitquant.quant.qconfig import QConfig


def _tiny_vit():
    torch.manual_seed(0)
    return timm.create_model("deit_tiny_patch16_224", pretrained=False).eval()


def test_quant_attention_fp32_equivalent():
    torch.manual_seed(0)
    attn = Attention(dim=64, num_heads=4).eval()
    qattn = QuantAttention.from_float(attn, QConfig()).eval()
    x = torch.randn(2, 5, 64)
    with torch.no_grad():
        assert torch.allclose(qattn(x), attn(x), atol=1e-5)


def test_convert_replaces_modules():
    model = convert_vit(_tiny_vit(), QConfig())
    types = [type(m) for m in model.modules()]
    assert QuantAttention in types
    assert QuantConv2d in types  # patch embed
    assert QuantLinear in types  # mlp
    assert not any(isinstance(m, Attention) for m in model.modules())


def test_convert_skips_head():
    model = convert_vit(_tiny_vit(), QConfig())
    assert type(model.head) is torch.nn.Linear  # classifier stays fp32


def test_converted_model_fp32_equivalent():
    ref = _tiny_vit()
    torch.manual_seed(0)
    qmodel = convert_vit(timm.create_model("deit_tiny_patch16_224", pretrained=False),
                         QConfig()).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(qmodel(x), ref(x), atol=1e-4)


def test_converted_model_has_fake_quant():
    qmodel = convert_vit(_tiny_vit(), QConfig())
    n = sum(1 for m in qmodel.modules() if isinstance(m, FakeQuantize))
    assert n > 50  # 12 blocks x (qkv+proj+2 mlp) x 2 + matmuls + patch embed


def test_converted_module_names_stable():
    # Task 11 groups FakeQuantize modules by these name prefixes — pin the contract.
    names = dict(convert_vit(_tiny_vit(), QConfig()).named_modules())
    assert "patch_embed.proj.input_fq" in names
    assert "blocks.0.attn.qkv.input_fq" in names
    assert "blocks.0.attn.qk_matmul.a_fq" in names
    assert "blocks.0.mlp.fc1.weight_fq" in names


def test_quant_attention_mask_path_matches_timm():
    torch.manual_seed(0)
    attn = Attention(dim=64, num_heads=4).eval()
    qattn = QuantAttention.from_float(attn, QConfig()).eval()
    x = torch.randn(2, 5, 64)
    mask = torch.randn(2, 1, 5, 5)
    with torch.no_grad():
        assert torch.allclose(qattn(x, attn_mask=mask), attn(x, attn_mask=mask),
                              atol=1e-5)
