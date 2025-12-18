#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>

// CUDA kernel for preparing sparse mask
// 
// Template parameters:
//   kBlockM: Query tile size (M dimension)
//   kBlockN: Key tile size (N dimension)
//   Reordered: If true, use the optimized layout for GMMA access pattern
//
// Layout modes:
//   Original (Reordered=false): Bits are stored in column-major order within each block
//     - word0: cols [0-31], word1: cols [32-63], etc.
//   
//   Reordered (Reordered=true): Bits are grouped by quad_lane for optimal GMMA access
//     For kBlockN=128 (4 words, 32 bits per quad_lane):
//       - word0: quad_lane 0, word1: quad_lane 1, word2: quad_lane 2, word3: quad_lane 3
//     For kBlockN=64 (2 words, 16 bits per quad_lane):
//       - word0 low 16: quad_lane 0, word0 high 16: quad_lane 1
//       - word1 low 16: quad_lane 2, word1 high 16: quad_lane 3
//     
//     General formula:
//       BITS_PER_QUAD_LANE = kBlockN / 4
//       QUAD_LANES_PER_WORD = 32 / BITS_PER_QUAD_LANE
//       word_idx = quad_lane / QUAD_LANES_PER_WORD
//       bit_idx = (quad_lane % QUAD_LANES_PER_WORD) * BITS_PER_QUAD_LANE + (v * 2 + local_bit)
//     
//     This allows each thread to read only the bits it needs, achieving optimal data utilization

template<int kBlockM, int kBlockN, bool Reordered>
__global__ void prepare_sparse_mask_kernel(
    const int* __restrict__ topk_indices,   // [total_q, topk]
    int* __restrict__ fine_mask,            // [total_q, max_k_blocks, num_int32_per_block]
    const int total_q,
    const int topk,
    const int max_seqlen_k, 
    const int max_k_blocks
) {
    constexpr int NUM_INT32_PER_BLOCK = (kBlockN + 31) / 32;

    int q_idx = blockIdx.x * kBlockM + threadIdx.x; // BlockDim.x must equal kBlockM

    if (q_idx < total_q) {
        const int* my_indices = topk_indices + q_idx * topk;

        int* my_fine_mask_row = fine_mask + (int)q_idx * max_k_blocks * NUM_INT32_PER_BLOCK;

        for (int i = 0; i < topk; ++i) {
            int k_idx = my_indices[i];

            if (k_idx == -1) break;

            if (k_idx >= 0 && k_idx < max_seqlen_k) {
                int block_idx = k_idx / kBlockN;
                int token_in_block = k_idx % kBlockN;

                int word_idx, bit_idx;
                
                if constexpr (Reordered) {
                    // Reordered layout: group by quad_lane for optimal GMMA access
                    // Uses general formula that works for any kBlockN (64, 128, etc.)
                    constexpr int BITS_PER_QUAD_LANE = kBlockN / 4;
                    constexpr int QUAD_LANES_PER_WORD = 32 / BITS_PER_QUAD_LANE;
                    
                    int quad_lane = (token_in_block % 8) / 2;  // 0-3
                    int v = token_in_block / 8;                 // which 8-column block
                    int local_bit = token_in_block % 2;         // offset within 2-column pair
                    
                    word_idx = quad_lane / QUAD_LANES_PER_WORD;
                    bit_idx = (quad_lane % QUAD_LANES_PER_WORD) * BITS_PER_QUAD_LANE + (v * 2 + local_bit);
                } else {
                    // Original layout: sequential bit storage
                    word_idx = token_in_block / 32;
                    bit_idx = token_in_block % 32;
                }

                int* block_ptr = my_fine_mask_row + block_idx * NUM_INT32_PER_BLOCK;
                block_ptr[word_idx] |= (1 << bit_idx);
            }
        }
    }
}

// Dispatch function with reordered parameter
void launch_prepare_mask(
    torch::Tensor topk_indices,
    torch::Tensor fine_mask,
    int total_q,
    int max_seqlen_k,
    int max_k_blocks,
    int kBlockM,
    int kBlockN,
    bool reordered
) {
    TORCH_CHECK(topk_indices.is_cuda(), "topk_indices must be a CUDA tensor");
    TORCH_CHECK(fine_mask.is_cuda(), "fine_mask must be a CUDA tensor");

    int topk = topk_indices.size(1);

    dim3 grid((total_q + kBlockM - 1) / kBlockM);
    dim3 block(kBlockM);
    size_t smem_size = 0;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Macro to reduce code duplication for kernel dispatch
    #define DISPATCH_KERNEL(M, N) \
        if (reordered) { \
            prepare_sparse_mask_kernel<M, N, true><<<grid, block, smem_size, stream>>>( \
                topk_indices.data_ptr<int>(), \
                fine_mask.data_ptr<int>(), \
                total_q, topk, max_seqlen_k, max_k_blocks \
            ); \
        } else { \
            prepare_sparse_mask_kernel<M, N, false><<<grid, block, smem_size, stream>>>( \
                topk_indices.data_ptr<int>(), \
                fine_mask.data_ptr<int>(), \
                total_q, topk, max_seqlen_k, max_k_blocks \
            ); \
        }

    if (kBlockM == 128 && kBlockN == 128) {
        DISPATCH_KERNEL(128, 128)
    } 
    else if (kBlockM == 64 && kBlockN == 64) {
        DISPATCH_KERNEL(64, 64)
    }
    else if (kBlockM == 128 && kBlockN == 64) {
        DISPATCH_KERNEL(128, 64)
    }
    else if (kBlockM == 64 && kBlockN == 128) {
        DISPATCH_KERNEL(64, 128)
    }
    else if (kBlockM == 192 && kBlockN == 128) {
        DISPATCH_KERNEL(192, 128)
    }
    else {
        TORCH_CHECK(false, "Unsupported tile size combination: M=", kBlockM, ", N=", kBlockN);
    }

    #undef DISPATCH_KERNEL

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed : %s\n", cudaGetErrorString(err));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prepare_sparse_mask", &launch_prepare_mask, "Prepare Sparse Mask Kernel",
          py::arg("topk_indices"),
          py::arg("fine_mask"),
          py::arg("total_q"),
          py::arg("max_seqlen_k"),
          py::arg("max_k_blocks"),
          py::arg("kBlockM"),
          py::arg("kBlockN"),
          py::arg("reordered") = false);
}
