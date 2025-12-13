#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>

// cuda kernels

template<int kBlockM, int kBlockN>
__global__ void prepare_sparse_mask_kernel(
    const int32_t* __restrict__ topk_indices,   // [total_q, topk]
    int64_t* __restrict__ fine_mask,            // [total_q, max_k_blocks, num_int64_per_block]
    bool* __restrict__ coarse_mask,             // [num_q_tiles, max_k_blocks]
    const int total_q,
    const int topk,
    const int max_seqlen_k, 
    const int max_k_blocks
) {
    constexpr int NUM_INT64_PER_BLOCK = (kBlockN + 63) / 64;

    int q_idx = blockIdx.x * kBlockM + threadIdx.x; // BlockDim.x must equal kBlockM
    extern __shared__ int s_coarse_mask[]; // size: max_k_blocks

    for (int i = threadIdx.x; i < max_k_blocks; i += blockDim.x) {
        s_coarse_mask[i] = 0;
    }
    __syncthreads();

    if (q_idx < total_q) {
        const int32_t* my_indices = topk_indices + q_idx * topk;

        int64_t* my_fine_mask_row = fine_mask + (int64_t)q_idx * max_k_blocks * NUM_INT64_PER_BLOCK;

        for (int i = 0; i < topk; ++i) {
            int32_t k_idx = my_indices[i];

            if (k_idx == -1) break;

            if (k_idx >= 0 && k_idx < max_seqlen_k) {
                int block_idx = k_idx / kBlockN;
                int token_in_block = k_idx % kBlockN;
                int word_idx = token_in_block / 64;
                int bit_idx = token_in_block % 64;

                int64_t* block_ptr = my_fine_mask_row + block_idx * NUM_INT64_PER_BLOCK;
                block_ptr[word_idx] |= (1LL << bit_idx);

                if (block_idx < max_k_blocks) {
                    if(s_coarse_mask[block_idx] == 0) {
                        atomicOr(&s_coarse_mask[block_idx], 1);
                    }
                }
            }
        }
    }

    __syncthreads();

    int num_q_tiles = (total_q + kBlockM - 1) / kBlockM;
    if (blockIdx.x < num_q_tiles) {
        bool* my_coarse_row = coarse_mask + blockIdx.x * max_k_blocks;
        for (int i = threadIdx.x; i < max_k_blocks; i += blockDim.x) {
            my_coarse_row[i] = (s_coarse_mask[i] > 0);
        }
    }
}

// dispatch
void launch_prepare_mask(
    torch::Tensor topk_indices,
    torch::Tensor fine_mask,
    torch::Tensor coarse_mask,
    int total_q,
    int max_seqlen_k,
    int kBlockM,
    int kBlockN
) {
    TORCH_CHECK(topk_indices.is_cuda(), "topk_indices must be a CUDA tensor");
    TORCH_CHECK(fine_mask.is_cuda(), "fine_mask must be a CUDA tensor");
    TORCH_CHECK(coarse_mask.is_cuda(), "coarse_mask must be a CUDA tensor");

    int topk = topk_indices.size(1);
    int max_k_blocks = coarse_mask.size(1);

    dim3 grid((total_q + kBlockM - 1) / kBlockM);
    dim3 block(kBlockM);
    size_t smem_size = max_k_blocks * sizeof(int);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (kBlockM == 128 && kBlockN == 128) {
        prepare_sparse_mask_kernel<128, 128><<<grid, block, smem_size, stream>>>(
            topk_indices.data_ptr<int32_t>(),
            fine_mask.data_ptr<int64_t>(),
            coarse_mask.data_ptr<bool>(),
            total_q, topk, max_seqlen_k, max_k_blocks
        );
    } 
    else if (kBlockM == 64 && kBlockN == 64) {
        prepare_sparse_mask_kernel<64, 64><<<grid, block, smem_size, stream>>>(
            topk_indices.data_ptr<int32_t>(),
            fine_mask.data_ptr<int64_t>(),
            coarse_mask.data_ptr<bool>(),
            total_q, topk, max_seqlen_k, max_k_blocks
        );
    }
    else if (kBlockM == 128 && kBlockN == 64) {
        prepare_sparse_mask_kernel<128, 64><<<grid, block, smem_size, stream>>>(
            topk_indices.data_ptr<int32_t>(),
            fine_mask.data_ptr<int64_t>(),
            coarse_mask.data_ptr<bool>(),
            total_q, topk, max_seqlen_k, max_k_blocks
        );
    }
    else if (kBlockM == 64 && kBlockN == 128) {
         prepare_sparse_mask_kernel<64, 128><<<grid, block, smem_size, stream>>>(
            topk_indices.data_ptr<int32_t>(),
            fine_mask.data_ptr<int64_t>(),
            coarse_mask.data_ptr<bool>(),
            total_q, topk, max_seqlen_k, max_k_blocks
        );
    }
    else if (kBlockM == 192 && kBlockN == 128) {
         prepare_sparse_mask_kernel<192, 128><<<grid, block, smem_size, stream>>>(
            topk_indices.data_ptr<int32_t>(),
            fine_mask.data_ptr<int64_t>(),
            coarse_mask.data_ptr<bool>(),
            total_q, topk, max_seqlen_k, max_k_blocks
        );
    }
    else {
        TORCH_CHECK(false, "Unsupported tile size combination: M=", kBlockM, ", N=", kBlockN);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed : %s\n", cudaGetErrorString(err));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prepare_sparse_mask", &launch_prepare_mask, "Prepare Sparse Mask Kernel");
}