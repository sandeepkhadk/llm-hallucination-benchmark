"""
Space-complexity measurement utilities (Methodology III-A theoretical
framework and III-C metric logging): peak VRAM, KV-cache footprint, and
vector-store memory footprint.
"""
from __future__ import annotations

from typing import Optional

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def reset_peak_vram() -> None:
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb() -> float:
    """Peak VRAM via torch.cuda.max_memory_allocated() (Methodology III-C)."""
    if _HAS_TORCH and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def estimate_kv_cache_mb(
    batch_size: int, seq_len: int, num_layers: int, num_heads: int, head_dim: int, dtype_bytes: int = 2
) -> float:
    """Formula-based KV-cache estimate matching the paper's theoretical bound
    O(N * L * d_model): 2 (K and V) * layers * heads * head_dim * seq_len *
    batch * bytes-per-element."""
    total_bytes = 2 * num_layers * num_heads * head_dim * seq_len * batch_size * dtype_bytes
    return total_bytes / (1024 ** 2)


def vector_store_memory_mb(num_vectors: int, embedding_dim: int, dtype_bytes: int = 4) -> float:
    """Vector-store memory footprint, V * d_embed * bytes (Methodology
    III-C): 4 bytes/dim for FP32 embeddings, 2 bytes/dim for FP16."""
    return (num_vectors * embedding_dim * dtype_bytes) / (1024 ** 2)


def try_get_vllm_kv_cache_usage(llm) -> Optional[dict]:
    """Best-effort introspection of vLLM's internal block manager for
    ground-truth KV-cache usage (Methodology III-D, PagedAttention [7]).
    vLLM's internal APIs change across versions, so this returns None if
    introspection fails; fall back to `estimate_kv_cache_mb` in that case."""
    try:
        engine = llm.llm_engine
        cache_config = engine.cache_config
        num_gpu_blocks = cache_config.num_gpu_blocks
        block_size = cache_config.block_size
        free_blocks = engine.scheduler.block_manager.get_num_free_gpu_blocks()
        used_blocks = num_gpu_blocks - free_blocks
        return dict(num_gpu_blocks=num_gpu_blocks, used_blocks=used_blocks, block_size=block_size)
    except Exception:  # noqa: BLE001
        return None
