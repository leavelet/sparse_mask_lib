import torch
try:
    import sparse_mask_lib
except ImportError:
    print("Error: sparse_mask_lib not found. Please run 'python setup.py install'")

def prepare_sparse_mask(
    topk_indices: torch.Tensor,  # [total_q, topk] int32
    total_q: int,
    max_seqlen_k: int,
    kBlockN: int,
    kBlockM: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Python wrapper that allocates memory and calls the CUDA kernel.
    """
    device = topk_indices.device
    
    num_int64_per_block = (kBlockN + 63) // 64
    max_k_blocks = (max_seqlen_k + kBlockN - 1) // kBlockN
    num_q_tiles = (total_q + kBlockM - 1) // kBlockM
    
    fine_mask = torch.zeros(
        (total_q, max_k_blocks, num_int64_per_block),
        dtype=torch.int64,
        device=device
    )
    
    coarse_mask = torch.zeros(
        (num_q_tiles, max_k_blocks),
        dtype=torch.bool,
        device=device
    )
    
    sparse_mask_lib.prepare_sparse_mask(
        topk_indices,
        fine_mask,
        coarse_mask,
        total_q,
        max_seqlen_k,
        kBlockM,
        kBlockN
    )
    
    return fine_mask, coarse_mask
