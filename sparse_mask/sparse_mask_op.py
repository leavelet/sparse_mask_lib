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
) -> torch.Tensor:
    """
    Python wrapper that allocates memory and calls the CUDA kernel.
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
        kBlockN
    )
    
    return fine_mask
