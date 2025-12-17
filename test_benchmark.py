import torch
import time
from sparse_mask import prepare_sparse_mask

def test_correctness_and_perf():
    torch.manual_seed(42)
    device = torch.device("cuda")
    
    total_q = 8192
    max_seqlen_k = 8192
    topk = 2048
    
    kBlockM = 128
    kBlockN = 128
    
    # Calculate max_k_blocks and ensure (max_k_blocks * num_int32_per_block * sizeof(int)) is a multiple of 128
    max_k_blocks = (max_seqlen_k + kBlockN - 1) // kBlockN
    num_int32_per_block = (kBlockN + 31) // 32
    # Ensure (max_k_blocks * num_int32_per_block) is a multiple of 32
    product = max_k_blocks * num_int32_per_block
    if product % 32 != 0:
        # Round up product to multiple of 32
        aligned_product = ((product + 31) // 32) * 32
        max_k_blocks = aligned_product // num_int32_per_block
    
    print(f"Config: Q={total_q}, K_len={max_seqlen_k}, TopK={topk}, Tile={kBlockM}x{kBlockN}, max_k_blocks={max_k_blocks}")

    topk_indices = torch.randint(0, max_seqlen_k, (total_q, topk), dtype=torch.int32, device=device)
    
    for _ in range(5):
        prepare_sparse_mask(topk_indices, total_q, max_seqlen_k, max_k_blocks, kBlockN, kBlockM)
    
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(100):
        fine_mask = prepare_sparse_mask(topk_indices, total_q, max_seqlen_k, max_k_blocks, kBlockN, kBlockM)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event) / 100
    print(f"Average time: {elapsed_ms:.4f} ms")
    
    print(f"Fine mask shape: {fine_mask.shape}")
    
    q_idx = 0
    indices = topk_indices[q_idx].cpu().numpy()
    f_mask = fine_mask[q_idx].cpu()
    
    for k_idx in indices:
        if k_idx == -1: continue
        block_idx = k_idx // kBlockN
        token_in_block = k_idx % kBlockN
        word_idx = token_in_block // 32
        bit_idx = int(token_in_block % 32)
        
        val = f_mask[block_idx, word_idx].item()
        is_set = (val >> bit_idx) & 1
        assert is_set == 1, f"Bit not set for k_idx {k_idx}"

    print("Basic correctness check passed!")

if __name__ == "__main__":
    test_correctness_and_perf()
