import torch
try:
    import sparse_mask_lib
except ImportError:
    print("Error: sparse_mask_lib not found. Please run 'python setup.py install'")

def prepare_sparse_mask(
    topk_indices: torch.Tensor,  # [total_q, topk] int32
    total_q: int,
    max_seqlen_k: int,
    max_k_blocks: int,
    kBlockN: int,
    kBlockM: int,
    reordered: bool = False,
) -> torch.Tensor:
    """
    Python wrapper that allocates memory and calls the CUDA kernel.
    
    Args:
        topk_indices: [total_q, topk] tensor of int32, containing the top-k key indices for each query
        total_q: Total number of query tokens
        max_seqlen_k: Maximum sequence length of keys
        max_k_blocks: Maximum number of key blocks
        kBlockN: Key tile size (N dimension)
        kBlockM: Query tile size (M dimension)
        reordered: If True, use the optimized layout for GMMA access pattern.
                   - False (default): Original layout with sequential bit storage
                     word0: cols [0-31], word1: cols [32-63], etc.
                   - True: Reordered layout grouped by quad_lane for optimal GMMA access
                     word0: quad_lane 0 cols, word1: quad_lane 1 cols, etc.
                     This allows each thread to read only 1 int32 instead of 4.
    
    Returns:
        fine_mask: [total_q, max_k_blocks, num_int32_per_block] tensor of int32
    """
    device = topk_indices.device
    
    num_int32_per_block = (kBlockN + 31) // 32
    
    # (max_k_blocks * num_int32_per_block * sizeof(int)) must be multiple of 128
    # i.e., (max_k_blocks * num_int32_per_block) must be multiple of 32
    if (max_k_blocks * num_int32_per_block) % 32 != 0:
        raise ValueError(f"(max_k_blocks * num_int32_per_block * sizeof(int)) must be multiple of 128, got max_k_blocks={max_k_blocks}, num_int32_per_block={num_int32_per_block}, product={max_k_blocks * num_int32_per_block}")
    
    fine_mask = torch.zeros(
        (total_q, max_k_blocks, num_int32_per_block),
        dtype=torch.int32,
        device=device
    )
    
    sparse_mask_lib.prepare_sparse_mask(
        topk_indices,
        fine_mask,
        total_q,
        max_seqlen_k,
        max_k_blocks,
        kBlockM,
        kBlockN,
        reordered
    )
    
    return fine_mask
