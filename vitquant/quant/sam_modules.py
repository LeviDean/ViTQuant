import torch
from torch import nn
from torch.nn import functional as F

from vitquant.quant.modules import QuantLinear, QuantMatMul
from vitquant.quant.qconfig import QConfig


class QuantSamAttention(nn.Module):
    """SAM vision encoder's windowed attention with relative position
    embeddings, rewritten with explicit matmuls so q@k^T and attn@v can be
    fake-quantized. qkv/proj become QuantLinear. The relative-position-
    embedding addition (pre-softmax) and softmax itself stay fp32 - same
    design decision as the classification ViT's QuantAttention."""

    @classmethod
    def from_float(cls, attn: nn.Module, qconfig: QConfig) -> "QuantSamAttention":
        new = cls.__new__(cls)
        nn.Module.__init__(new)
        new.num_attention_heads = attn.num_attention_heads
        new.scale = attn.scale
        new.dropout = attn.dropout  # a plain float on SamVisionAttention too, not an nn.Dropout
        new.use_rel_pos = attn.use_rel_pos
        if new.use_rel_pos:
            new.rel_pos_h = attn.rel_pos_h
            new.rel_pos_w = attn.rel_pos_w
        new.qkv = QuantLinear.from_float(attn.qkv, qconfig)
        new.proj = QuantLinear.from_float(attn.proj, qconfig)
        new.qk_matmul = QuantMatMul(qconfig)
        new.av_matmul = QuantMatMul(qconfig)
        return new

    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).transpose(1, 2),
            size=max_rel_dist, mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
        return rel_pos_resized[relative_coords.long()]

    def get_decomposed_rel_pos(self, query, rel_pos_h, rel_pos_w, q_size, k_size):
        query_height, query_width = q_size
        key_height, key_width = k_size
        relative_position_height = self.get_rel_pos(query_height, key_height, rel_pos_h)
        relative_position_width = self.get_rel_pos(query_width, key_width, rel_pos_w)
        batch_size, _, dim = query.shape
        reshaped_query = query.reshape(batch_size, query_height, query_width, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", reshaped_query, relative_position_height)
        rel_w = torch.einsum("bhwc,wkc->bhwk", reshaped_query, relative_position_width)
        decomposed_rel_pos = rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        return decomposed_rel_pos

    def forward(self, hidden_states: torch.Tensor, output_attentions=None):
        batch_size, height, width, _ = hidden_states.shape
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, height * width, 3, self.num_attention_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.reshape(3, batch_size * self.num_attention_heads, height * width, -1).unbind(0)
        attn_weights = self.qk_matmul(query * self.scale, key.transpose(-2, -1))
        if self.use_rel_pos:
            decomposed_rel_pos = self.get_decomposed_rel_pos(
                query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width))
            decomposed_rel_pos = decomposed_rel_pos.reshape_as(attn_weights)
            attn_weights = attn_weights + decomposed_rel_pos
        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = self.av_matmul(attn_probs, value).reshape(batch_size, self.num_attention_heads, height, width, -1)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)
        attn_output = self.proj(attn_output)
        return attn_output, attn_weights
