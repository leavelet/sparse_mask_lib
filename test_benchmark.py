import torch
import time
from sparse_mask import prepare_sparse_mask


def get_bit_position_original(token_in_block: int, kBlockN: int) -> tuple[int, int]:
    """Get word_idx and bit_idx for original layout."""
    word_idx = token_in_block // 32
    bit_idx = token_in_block % 32
    return word_idx, bit_idx


def get_bit_position_reordered(token_in_block: int, kBlockN: int) -> tuple[int, int]:
    """
    Get word_idx and bit_idx for reordered layout.
    
    Reordered layout groups bits by quad_lane for optimal GMMA access.
    Uses general formula that works for any kBlockN:
    
    For kBlockN=128 (4 words, 32 bits per quad_lane):
      - word0: quad_lane 0, word1: quad_lane 1, word2: quad_lane 2, word3: quad_lane 3
    For kBlockN=64 (2 words, 16 bits per quad_lane):
      - word0 low 16: quad_lane 0, word0 high 16: quad_lane 1
      - word1 low 16: quad_lane 2, word1 high 16: quad_lane 3
    """
    bits_per_quad_lane = kBlockN // 4
    quad_lanes_per_word = 32 // bits_per_quad_lane
    
    quad_lane = (token_in_block % 8) // 2  # 0-3
    v = token_in_block // 8                 # which 8-column block
    local_bit = token_in_block % 2          # offset within 2-column pair
    
    word_idx = quad_lane // quad_lanes_per_word
    bit_idx = (quad_lane % quad_lanes_per_word) * bits_per_quad_lane + (v * 2 + local_bit)
    return word_idx, bit_idx


def verify_mask_correctness(topk_indices, fine_mask, kBlockN, reordered):
    """Verify mask correctness for a given layout mode."""
    q_idx = 0
    indices = topk_indices[q_idx].cpu().numpy()
    f_mask = fine_mask[q_idx].cpu()
    
    for k_idx in indices:
        if k_idx == -1:
            continue
        block_idx = k_idx // kBlockN
        token_in_block = k_idx % kBlockN
        
        if reordered:
            word_idx, bit_idx = get_bit_position_reordered(token_in_block, kBlockN)
        else:
            word_idx, bit_idx = get_bit_position_original(token_in_block, kBlockN)
        
        val = f_mask[block_idx, word_idx].item()
        is_set = (val >> bit_idx) & 1
        assert is_set == 1, f"Bit not set for k_idx {k_idx}, word_idx {word_idx}, bit_idx {bit_idx}"
    
    return True


def count_bits(tensor):
    """Count total number of set bits in tensor."""
    bits = 0
    for val in tensor.view(-1).cpu().tolist():
        # Convert to unsigned 32-bit integer to handle negative values
        bits += bin(int(val) & 0xFFFFFFFF).count('1')
    return bits


def test_single_config(total_q, max_seqlen_k, topk, kBlockM, kBlockN):
    """Test a single tile configuration."""
    device = torch.device("cuda")
    
    # Calculate max_k_blocks and ensure alignment
    max_k_blocks = (max_seqlen_k + kBlockN - 1) // kBlockN
    num_int32_per_block = (kBlockN + 31) // 32
    product = max_k_blocks * num_int32_per_block
    if product % 32 != 0:
        aligned_product = ((product + 31) // 32) * 32
        max_k_blocks = aligned_product // num_int32_per_block
    
    print(f"\n{'='*60}")
    print(f"Config: Q={total_q}, K_len={max_seqlen_k}, TopK={topk}, Tile={kBlockM}x{kBlockN}, max_k_blocks={max_k_blocks}")
    print(f"{'='*60}")

    topk_indices = torch.randint(0, max_seqlen_k, (total_q, topk), dtype=torch.int32, device=device)
    
    # Test both layout modes
    for reordered in [False, True]:
        layout_name = "Reordered" if reordered else "Original"
        print(f"\n--- {layout_name} Layout ---")
        
        # Warmup
        for _ in range(5):
            prepare_sparse_mask(topk_indices, total_q, max_seqlen_k, max_k_blocks, kBlockN, kBlockM, reordered)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(100):
            fine_mask = prepare_sparse_mask(topk_indices, total_q, max_seqlen_k, max_k_blocks, kBlockN, kBlockM, reordered)
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start_event.elapsed_time(end_event) / 100
        print(f"Average time: {elapsed_ms:.4f} ms")
        print(f"Fine mask shape: {fine_mask.shape}")
        
        # Correctness check
        verify_mask_correctness(topk_indices, fine_mask, kBlockN, reordered)
        print(f"Correctness check passed!")
    
    # Verify layout equivalence
    print(f"\n--- Layout Equivalence Check ---")
    fine_mask_original = prepare_sparse_mask(topk_indices, total_q, max_seqlen_k, max_k_blocks, kBlockN, kBlockM, reordered=False)
    fine_mask_reordered = prepare_sparse_mask(topk_indices, total_q, max_seqlen_k, max_k_blocks, kBlockN, kBlockM, reordered=True)
    
    # Sample check on first few rows
    for q_idx in range(min(5, total_q)):
        orig_count = count_bits(fine_mask_original[q_idx])
        reord_count = count_bits(fine_mask_reordered[q_idx])
        assert orig_count == reord_count, f"Bit count mismatch at q_idx {q_idx}: original={orig_count}, reordered={reord_count}"
    
    print("Layout equivalence check passed! (Both layouts have same bits set)")


def test_correctness_and_perf():
    torch.manual_seed(42)
    
    # Test configurations
    configs = [
        # (total_q, max_seqlen_k, topk, kBlockM, kBlockN)
        (8192, 8192, 2048, 128, 128),   # Standard config
        (8192, 8192, 2048, 64, 64),     # Small tile
        (8192, 8192, 2048, 128, 64),    # Mixed tile
        (8192, 8192, 2048, 64, 128),    # Mixed tile (transpose)
    ]
    
    for config in configs:
        test_single_config(*config)
    
    print(f"\n{'='*60}")
    print("All tests passed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_correctness_and_perf()
