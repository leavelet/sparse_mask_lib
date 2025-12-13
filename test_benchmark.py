import torch
import time
from sparse_mask_op import prepare_sparse_mask

def test_correctness_and_perf():
    torch.manual_seed(42)
    device = torch.device("cuda")
    
    total_q = 8192
    max_seqlen_k = 8192
    topk = 2048
    
    kBlockM = 128
    kBlockN = 128 
    
    print(f"Config: Q={total_q}, K_len={max_seqlen_k}, TopK={topk}, Tile={kBlockM}x{kBlockN}")

    topk_indices = torch.randint(0, max_seqlen_k, (total_q, topk), dtype=torch.int32, device=device)
    
    for _ in range(5):
        prepare_sparse_mask(topk_indices, total_q, max_seqlen_k, kBlockN, kBlockM)
    
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(100):
        fine_mask, coarse_mask = prepare_sparse_mask(topk_indices, total_q, max_seqlen_k, kBlockN, kBlockM)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event) / 100
    print(f"Average time: {elapsed_ms:.4f} ms")
    
    print(f"Fine mask shape: {fine_mask.shape}")
    print(f"Coarse mask shape: {coarse_mask.shape}")
    
    q_idx = 0
    indices = topk_indices[q_idx].cpu().numpy()
    f_mask = fine_mask[q_idx].cpu()
    
    for k_idx in indices:
        if k_idx == -1: continue
        block_idx = k_idx // kBlockN
        token_in_block = k_idx % kBlockN
        word_idx = token_in_block // 64
        bit_idx = token_in_block % 64
        
        val = f_mask[block_idx, word_idx].item()
        is_set = (val >> bit_idx) & 1
        assert is_set == 1, f"Bit not set for k_idx {k_idx}"

    print("Basic correctness check passed!")

if __name__ == "__main__":
    test_correctness_and_perf()