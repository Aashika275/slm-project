# ============================================================
# CODEX-SLM MODEL
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. RMS NORMALIZATION
# ============================================================

class RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):

        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(hidden_size)
        )

    def forward(self, x):

        rms = torch.rsqrt(
            x.pow(2).mean(
                dim=-1,
                keepdim=True
            ) + self.eps
        )

        return (
            x * rms
        ) * self.weight


# ============================================================
# 2. ROTARY POSITIONAL EMBEDDING
# ============================================================

class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_dim,
        max_seq_len=512,
        base=10000
    ):

        super().__init__()

        self.head_dim = head_dim

        inv_freq = 1.0 / (
            base ** (
                torch.arange(
                    0,
                    head_dim,
                    2
                ).float()
                / head_dim
            )
        )

        positions = torch.arange(
            max_seq_len
        ).float()

        freqs = torch.outer(
            positions,
            inv_freq
        )

        self.register_buffer(
            "cos_cached",
            freqs.cos(),
            persistent=False
        )

        self.register_buffer(
            "sin_cached",
            freqs.sin(),
            persistent=False
        )

    def forward(self, x):

        seq_len = x.size(-2)

        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        return cos, sin


def rotate_half(x):

    x1 = x[..., ::2]

    x2 = x[..., 1::2]

    return torch.stack(
        (-x2, x1),
        dim=-1
    ).flatten(-2)


def apply_rotary(
    x,
    cos,
    sin
):

    # x shape:
    # [batch, heads, sequence, head_dim]

    cos = torch.stack(
        [cos, cos],
        dim=-1
    ).flatten(-2)

    sin = torch.stack(
        [sin, sin],
        dim=-1
    ).flatten(-2)

    cos = cos.unsqueeze(0).unsqueeze(0)

    sin = sin.unsqueeze(0).unsqueeze(0)

    return (
        x * cos
        + rotate_half(x) * sin
    )


# ============================================================
# 3. CAUSAL SELF ATTENTION
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_heads,
        max_seq_len
    ):

        super().__init__()

        if hidden_size % num_heads != 0:

            raise ValueError(
                "hidden_size must be "
                "divisible by num_heads"
            )

        self.hidden_size = hidden_size

        self.num_heads = num_heads

        self.head_dim = (
            hidden_size // num_heads
        )

        self.q_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False
        )

        self.k_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False
        )

        self.v_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False
        )

        self.out_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False
        )

        self.rotary = RotaryEmbedding(
            self.head_dim,
            max_seq_len
        )

        # ----------------------------------------------------
        # Causal mask
        #
        # Token at position i can only see:
        # positions 0 ... i
        # ----------------------------------------------------

        mask = torch.tril(
            torch.ones(
                max_seq_len,
                max_seq_len
            )
        )

        self.register_buffer(
            "mask",
            mask,
            persistent=False
        )

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        # ----------------------------------------------------
        # Q, K, V
        # ----------------------------------------------------

        q = self.q_proj(x)

        k = self.k_proj(x)

        v = self.v_proj(x)

        # ----------------------------------------------------
        # Reshape
        # ----------------------------------------------------

        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # ----------------------------------------------------
        # Rotary embeddings
        # ----------------------------------------------------

        cos, sin = self.rotary(q)

        q = apply_rotary(
            q,
            cos,
            sin
        )

        k = apply_rotary(
            k,
            cos,
            sin
        )

        # ----------------------------------------------------
        # Attention scores
        # ----------------------------------------------------

        attention_scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(
            self.head_dim
        )

        # ----------------------------------------------------
        # Causal mask
        # ----------------------------------------------------

        causal_mask = self.mask[
            :seq_len,
            :seq_len
        ]

        attention_scores = attention_scores.masked_fill(
            causal_mask == 0,
            float("-inf")
        )

        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        attention_weights = F.softmax(
            attention_scores,
            dim=-1
        )

        # ----------------------------------------------------
        # Weighted values
        # ----------------------------------------------------

        attention_output = (
            attention_weights @ v
        )

        # ----------------------------------------------------
        # Restore shape
        # ----------------------------------------------------

        attention_output = attention_output.transpose(
            1,
            2
        ).contiguous()

        attention_output = attention_output.view(
            batch_size,
            seq_len,
            self.hidden_size
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        return self.out_proj(
            attention_output
        )


# ============================================================
# 4. SWIGLU FEED FORWARD NETWORK
# ============================================================

class SwiGLU(nn.Module):

    def __init__(
        self,
        hidden_size,
        intermediate_size
    ):

        super().__init__()

        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False
        )

        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False
        )

        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False
        )

    def forward(self, x):

        gate = F.silu(
            self.gate_proj(x)
        )

        up = self.up_proj(x)

        return self.down_proj(
            gate * up
        )


# ============================================================
# 5. TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        max_seq_len
    ):

        super().__init__()

        self.norm1 = RMSNorm(
            hidden_size
        )

        self.attention = CausalSelfAttention(
            hidden_size,
            num_heads,
            max_seq_len
        )

        self.norm2 = RMSNorm(
            hidden_size
        )

        self.feed_forward = SwiGLU(
            hidden_size,
            intermediate_size
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Attention residual connection
        # ----------------------------------------------------

        x = x + self.attention(
            self.norm1(x)
        )

        # ----------------------------------------------------
        # Feed-forward residual connection
        # ----------------------------------------------------

        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x


# ============================================================
# 6. CODEX-SLM
# ============================================================

class CODEXSLM(nn.Module):

    def __init__(
        self,
        vocab_size=8000,
        max_seq_len=512,
        hidden_size=256,
        num_heads=4,
        num_layers=4,
        intermediate_size=1024
    ):

        super().__init__()

        self.vocab_size = vocab_size

        self.max_seq_len = max_seq_len

        # ----------------------------------------------------
        # Token embedding
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            hidden_size
        )

        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_heads,
                    intermediate_size,
                    max_seq_len
                )
                for _ in range(num_layers)
            ]
        )

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        self.final_norm = RMSNorm(
            hidden_size
        )

        # ----------------------------------------------------
        # Language modeling head
        # ----------------------------------------------------

        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False
        )

        # ----------------------------------------------------
        # Tie input and output embeddings
        # ----------------------------------------------------

        self.lm_head.weight = (
            self.token_embedding.weight
        )

    def forward(self, input_ids):

        # ----------------------------------------------------
        # Input shape
        #
        # [batch_size, sequence_length]
        # ----------------------------------------------------

        batch_size, seq_len = input_ids.shape

        # ----------------------------------------------------
        # Sequence length validation
        # ----------------------------------------------------

        if seq_len > self.max_seq_len:

            raise ValueError(
                f"Sequence length {seq_len} "
                f"exceeds maximum "
                f"{self.max_seq_len}"
            )

        # ----------------------------------------------------
        # Token embeddings
        # ----------------------------------------------------

        x = self.token_embedding(
            input_ids
        )

        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        for block in self.blocks:

            x = block(x)

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        x = self.final_norm(x)

        # ----------------------------------------------------
        # Vocabulary logits
        #
        # Shape:
        # [batch_size, sequence_length, vocab_size]
        # ----------------------------------------------------

        logits = self.lm_head(x)

        # ----------------------------------------------------
        # IMPORTANT:
        # Return logits
        # ----------------------------------------------------

        return logits