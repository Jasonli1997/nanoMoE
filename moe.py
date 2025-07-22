"""https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modular_qwen3_moe.py
https://github.com/kyegomez/SwitchTransformers/blob/main/switch_transformers/model.py"""

import math
from dataclasses import dataclass

import torch
from torch import nn

from gpt import GPTConfig


@dataclass
class MoEConfig(GPTConfig):
    topk = 2
    n_experts = 4
    capacity_factor = 1
    bias = False


# TODO: kv-cache
# TODO: Grouped attention
# TODO: add normalization (Qwen3RMSNorm and LayerNorm)
class CausalMultiHeadedAttention(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = config.n_head
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.atten_dropout = nn.Dropout(p=config.dropout)  # 0.1 in the original paper
        self.z_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        B, T, n_embd = x.size()
        tril = torch.tril(torch.ones(T, T))

        Q = self.q_proj(x)  # (B, T, n_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.reshape(B, T, self.n_head, -1).transpose(1, 2)
        K = K.reshape(B, T, self.n_head, -1).transpose(1, 2)
        V = V.reshape(B, T, self.n_head, -1).transpose(1, 2)

        atten = Q @ K.transpose(-1, -2) / math.sqrt(n_embd // self.n_head)
        causal_atten = atten.masked_fill(tril == 0, float("-inf"))

        final_atten = causal_atten.softmax(dim=-1)
        final_atten = self.atten_dropout(final_atten)
        Z = final_atten @ V  # (B, num_heads, T, head_dim)

        Z = Z.transpose(1, 2).reshape(B, T, n_embd)
        output = self.z_proj(Z)
        # TODO: nanogpt has another dropout here

        return output


# TODO: Qwen3 uses bilinear transformations instead of affine before applying non-linearity
class MLP(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.n_embd * 4, bias=config.bias)
        self.gelu = nn.GELU()  # TODO: Difference between GELU and RELU
        self.proj = nn.Linear(config.n_embd * 4, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.proj(self.gelu(self.fc(x))))


class TokenRouter(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.topk = config.topk
        self.gate = nn.Linear(config.n_embd, config.n_experts)

    def forward(self, x):
        logits = self.gate(x)  # (B, T, num_experts)

        top_k_logits, top_k_indices = logits.topk(self.topk, dim=-1)

        mask = torch.full_like(logits, float("-inf"))
        sparse_logits = mask.scatter(-1, top_k_indices, top_k_logits)
        sparse_probs = sparse_logits.softmax(-1)

        return top_k_indices, sparse_probs


class MoEBlockLayer(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.router = TokenRouter(config=config)
        self.experts = nn.ModuleList(
            [MLP(config=config) for _ in range(config.n_experts)]
        )
        self.topk = config.topk
        self.num_experts = config.n_experts
        self.capacity_factor = config.capacity_factor

    def forward(self, x):
        B, T, D = x.size()
        expert_capacity = int((B * T * self.capacity_factor) / self.num_experts)
        top_k_indices, sparse_probs = self.router(x)  # (B, T, topk), (B, T, n_experts)

        # Reshape for easier processing
        flat_x = x.view(-1, D)  # (B*T, D)
        flat_probs = sparse_probs.view(-1, self.num_experts)  # (B*T, n_experts)
        flat_top_k_indices = top_k_indices.view(-1, self.topk)  # (B*T, topk)
        output = torch.zeros_like(flat_x)

        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            expert_mask = (flat_top_k_indices == i).any(dim=-1)  # (B*T,)
            token_indices = torch.nonzero(expert_mask, as_tuple=False).squeeze(-1)

            if token_indices.numel() == 0:
                continue

            # Apply capacity constraint by selecting highest probability tokens
            if token_indices.numel() > expert_capacity:
                expert_probs = flat_probs[token_indices, i]
                _, sorted_indices = expert_probs.sort(descending=True)
                selected_indices = token_indices[sorted_indices[:expert_capacity]]

            else:
                selected_indices = token_indices

            # Run selected tokens through expert with associated weight
            routed_tokens = flat_x[selected_indices, :]
            selected_probs = flat_probs[selected_indices, i].unsqueeze(1)
            expert_output = expert(routed_tokens) * selected_probs

            output[selected_indices] += expert_output

        return output.view(B, T, D)
