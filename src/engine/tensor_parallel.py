"""
Tensor parallelism (Megatron-LM style), simulated on a single device.

WHAT TENSOR PARALLELISM IS
--------------------------
TP shards the WEIGHT MATRICES of a layer across N devices so each device does
1/N of the matmul, then combines results. Two complementary shardings (Shoeybi
et al. 2019, "Megatron-LM"):

  COLUMN-parallel linear  y = W x, W: (out, in)
      Split W along its OUTPUT dim (rows) into N pieces. Device r computes
      y_r = W[r-rows] x -> the r-th slice of y. No communication; the output is
      sharded across devices.

  ROW-parallel linear     y = W x, W: (out, in)
      Split W along its INPUT dim (columns). The INPUT is already sharded (it's
      the column-parallel output), so device r holds x_r and W[:, r-cols] and
      computes a PARTIAL output y_r = W[:, r-cols] x_r. The full output is
      sum_r y_r -- an ALL-REDUCE across devices.

The magic pairing: a column-parallel layer feeding a row-parallel layer needs
exactly ONE all-reduce (at the row-parallel output) per such pair, because
    W2 (W1 x) = sum_r W2[:, r-cols] (W1[r-rows] x).
Attention (qkv column-parallel, output row-parallel) and the MLP (fc1 column,
fc2 row) are each one such pair -> two all-reduces per transformer layer.

THE SIMULATION (and the real path)
----------------------------------
A real deployment runs ONE rank PER PROCESS on its own GPU; RowParallelLinear's
`all_reduce_sum` then sums partials across processes (nccl), and that all-reduce
overlapping real per-GPU compute is where the speedup comes from.

Here, on a single device, we SIMULATE N ranks IN ONE PROCESS: we instantiate the
per-rank shards and sum their row-parallel partials ourselves (see
tp_model.TensorParallelLlamaModel and the benchmark). `all_reduce_sum` still goes
through torch.distributed when a process group is initialised (so the gloo/nccl
code path is exercised and correct); with no group it is the identity, because in
the in-process simulation the summation is done explicitly across rank objects.
On a single device this is NOT faster than dense -- it does the same FLOPs
serially -- it reproduces the MATH and the communication structure. Real speedup
requires N physical GPUs.

To extend to real multi-GPU: launch N processes (torchrun), build ONE rank per
process via from_linear(..., rank=dist.get_rank()), put each on its own GPU, and
delete the in-process rank loop -- RowParallelLinear.all_reduce_sum already does
the cross-process combine.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.engine.attention import apply_rope, build_rope_cache


@dataclass
class TensorParallelConfig:
    """Describes the TP group this process participates in.

    world_size : number of TP ranks (devices) the weights are sharded across.
    rank       : this process's rank in [0, world_size).
    device_ids : optional explicit device ordinals per rank (real multi-GPU);
                 None in the single-device simulation.
    """
    world_size: int = 1
    rank: int = 0
    device_ids: list[int] | None = None

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("world_size must be >= 1")
        if not (0 <= self.rank < self.world_size):
            raise ValueError(f"rank {self.rank} out of range for world_size {self.world_size}")


def all_reduce_sum(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Sum `tensor` across the TP ranks of `group`, in place, returning it.

    Backed by torch.distributed (gloo on CPU, nccl on GPU) when a process group
    is initialised. With no initialised group -- the in-process simulation, or
    world_size 1 -- it is the identity: there is exactly one rank's contribution
    here, and the simulation sums across rank OBJECTS itself.
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor


def _even_split(total: int, world_size: int, what: str) -> int:
    if total % world_size != 0:
        raise ValueError(
            f"{what}={total} not divisible by world_size={world_size}; "
            "tensor parallelism needs an even shard per rank"
        )
    return total // world_size


# ---------------------------------------------------------------------------
# Column- and row-parallel linears.
# ---------------------------------------------------------------------------


class ColumnParallelLinear(nn.Module):
    """Linear whose weight is sharded along the OUTPUT dim; output is sharded.

    This rank holds weight rows [out/N * rank : out/N * (rank+1)], so forward(x)
    returns this rank's slice of the full output -- no communication.
    """

    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features          # FULL output dim
        self.world_size = world_size
        self.rank = rank
        self.local_out = _even_split(out_features, world_size, "out_features")
        # Bias-less, matching LLaMA's projections.
        self.weight = nn.Parameter(torch.empty(self.local_out, in_features))

    @classmethod
    def from_linear(cls, linear: nn.Linear, world_size: int, rank: int) -> "ColumnParallelLinear":
        """Build rank `rank`'s column shard from a dense nn.Linear."""
        out_f, in_f = linear.weight.shape
        m = cls(in_f, out_f, world_size, rank)
        local_out = m.local_out
        with torch.no_grad():
            row0 = rank * local_out
            m.weight.copy_(linear.weight[row0:row0 + local_out, :])
        return m.to(linear.weight.device).to(linear.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (..., in) -> (..., out/N): this rank's slice of the output.
        return F.linear(x, self.weight)


class RowParallelLinear(nn.Module):
    """Linear whose weight is sharded along the INPUT dim; output is all-reduced.

    The input is expected to already be sharded (the column-parallel output), so
    this rank holds weight columns [in/N * rank : in/N * (rank+1)] and input
    slice x_r. forward computes the PARTIAL output W[:, r-cols] x_r and reduces
    it across ranks (all_reduce_sum) to recover the full output.
    """

    def __init__(self, in_features: int, out_features: int, world_size: int,
                 rank: int, group=None) -> None:
        super().__init__()
        self.in_features = in_features            # FULL input dim
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.group = group
        self.local_in = _even_split(in_features, world_size, "in_features")
        self.weight = nn.Parameter(torch.empty(out_features, self.local_in))

    @classmethod
    def from_linear(cls, linear: nn.Linear, world_size: int, rank: int,
                    group=None) -> "RowParallelLinear":
        """Build rank `rank`'s row shard from a dense nn.Linear."""
        out_f, in_f = linear.weight.shape
        m = cls(in_f, out_f, world_size, rank, group=group)
        local_in = m.local_in
        with torch.no_grad():
            col0 = rank * local_in
            m.weight.copy_(linear.weight[:, col0:col0 + local_in])
        return m.to(linear.weight.device).to(linear.weight.dtype)

    def forward(self, x_shard: torch.Tensor) -> torch.Tensor:
        # Partial output from this rank's input+weight slice, then reduce.
        partial = F.linear(x_shard, self.weight)
        return all_reduce_sum(partial, self.group)


# ---------------------------------------------------------------------------
# Tensor-parallel attention (one rank's head shard).
# ---------------------------------------------------------------------------


class TensorParallelAttention(nn.Module):
    """One rank of a head-sharded attention block.

    Each rank owns num_heads/N query heads and num_kv_heads/N KV heads. Q/K/V are
    column-parallel (head shard); the output projection is row-parallel (it sums
    each rank's head contribution). forward returns this rank's PARTIAL output;
    summing partials across ranks equals dense multi-head attention exactly,
    because o_proj(concat_h head_h) = sum_r o_proj[:, r-cols] (concat of rank r's
    heads).

    This is the no-KV-cache (prefill) forward -- enough to demonstrate and test
    the sharding math and to drive the benchmark. Wiring the paged KV cache and
    the batched-decode path through TP is the documented next step.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        world_size: int,
        rank: int,
        rope_base: float = 10000.0,
        group=None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.world_size = world_size
        self.rank = rank
        self.local_num_heads = _even_split(num_heads, world_size, "num_heads")
        self.local_num_kv_heads = _even_split(num_kv_heads, world_size, "num_kv_heads")
        self.group_size = self.local_num_heads // self.local_num_kv_heads

        # Placeholders; real shards are installed by from_attention. Sized so a
        # standalone instance is still usable (e.g. tests building from scratch).
        self.q_proj = ColumnParallelLinear(hidden_size, num_heads * head_dim, world_size, rank)
        self.k_proj = ColumnParallelLinear(hidden_size, num_kv_heads * head_dim, world_size, rank)
        self.v_proj = ColumnParallelLinear(hidden_size, num_kv_heads * head_dim, world_size, rank)
        self.o_proj = RowParallelLinear(num_heads * head_dim, hidden_size, world_size, rank, group=group)

        cos, sin = build_rope_cache(head_dim, max_seq_len, base=rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @classmethod
    def from_attention(cls, attn, world_size: int, rank: int, group=None) -> "TensorParallelAttention":
        """Shard a dense MultiHeadAttention into rank `rank`."""
        m = cls(
            hidden_size=attn.hidden_size,
            num_heads=attn.num_heads,
            num_kv_heads=attn.num_kv_heads,
            head_dim=attn.head_dim,
            max_seq_len=attn.rope_cos.shape[0],
            world_size=world_size,
            rank=rank,
            group=group,
        )
        m.q_proj = ColumnParallelLinear.from_linear(attn.q_proj, world_size, rank)
        m.k_proj = ColumnParallelLinear.from_linear(attn.k_proj, world_size, rank)
        m.v_proj = ColumnParallelLinear.from_linear(attn.v_proj, world_size, rank)
        m.o_proj = RowParallelLinear.from_linear(attn.o_proj, world_size, rank, group=group)
        # Reuse the source's RoPE table verbatim (identical for every rank).
        m.rope_cos = attn.rope_cos
        m.rope_sin = attn.rope_sin
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.local_num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.local_num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.local_num_kv_heads, self.head_dim).transpose(1, 2)

        cos = self.rope_cos[:S].to(dtype=q.dtype)
        sin = self.rope_sin[:S].to(dtype=q.dtype)
        q, k = apply_rope(q, k, cos, sin)

        # GQA: replicate each local KV head to its local query-head group.
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, lnq, S, D)
        attn = attn.transpose(1, 2).contiguous().view(
            B, S, self.local_num_heads * self.head_dim
        )
        # Row-parallel output projection -> this rank's partial (reduced if a
        # process group is live; summed across rank objects in the simulation).
        return self.o_proj(attn)


# ---------------------------------------------------------------------------
# Tensor-parallel MLP (one rank's intermediate shard).
# ---------------------------------------------------------------------------


class TensorParallelMLP(nn.Module):
    """One rank of a SwiGLU MLP: gate/up column-parallel, down row-parallel.

    SwiGLU shards cleanly because `silu(gate) * up` is elementwise, so each
    intermediate dim is independent: rank r owns intermediate columns
    [inter/N * r : inter/N * (r+1)] of gate and up, computes that slice of the
    hidden, and down (row-parallel) sums each rank's contribution. forward
    returns this rank's PARTIAL output; the sum across ranks equals the dense MLP.
    """

    def __init__(self, hidden_size: int, intermediate_size: int,
                 world_size: int, rank: int, group=None) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, world_size, rank)
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, world_size, rank)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, world_size, rank, group=group)

    @classmethod
    def from_mlp(cls, mlp, world_size: int, rank: int, group=None) -> "TensorParallelMLP":
        """Shard a dense SwiGLUMLP into rank `rank`."""
        hidden = mlp.gate_proj.in_features
        inter = mlp.gate_proj.out_features
        m = cls(hidden, inter, world_size, rank, group=group)
        m.gate_proj = ColumnParallelLinear.from_linear(mlp.gate_proj, world_size, rank)
        m.up_proj = ColumnParallelLinear.from_linear(mlp.up_proj, world_size, rank)
        m.down_proj = RowParallelLinear.from_linear(mlp.down_proj, world_size, rank, group=group)
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)        # (.., inter/N)
        up = self.up_proj(x)            # (.., inter/N)
        hidden = F.silu(gate) * up      # elementwise -> stays sharded
        return self.down_proj(hidden)   # row-parallel partial (+ all_reduce)
