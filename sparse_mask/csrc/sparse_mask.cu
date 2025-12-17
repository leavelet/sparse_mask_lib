#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>

// cuda kernels

template<int kBlockM, int kBlockN>
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
                int word_idx = token_in_block / 32;
                int bit_idx = token_in_block % 32;

                int* block_ptr = my_fine_mask_row + block_idx * NUM_INT32_PER_BLOCK;
                block_ptr[word_idx] |= (1 << bit_idx);
            }
        }
    }
}

// dispatch
void launch_prepare_mask(
    torch::Tensor topk_indices,
    torch::Tensor fine_mask,
    int total_q,
    int max_seqlen_k,
    int max_k_blocks,
    int kBlockM,
    int kBlockN
) {
    TORCH_CHECK(topk_indices.is_cuda(), "topk_indices must be a CUDA tensor");
    TORCH_CHECK(fine_mask.is_cuda(), "fine_mask must be a CUDA tensor");

    int topk = topk_indices.size(1);

    dim3 grid((total_q + kBlockM - 1) / kBlockM);
    dim3 block(kBlockM);
    size_t smem_size = 0;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (kBlockM == 128 && kBlockN == 128) {
        prepare_sparse_mask_kernel<128, 128><<<grid, block, smem_size, stream>>>(
            topk_indices.data_ptr<int>(),
            fine_mask.data_ptr<int>(),
            total_q, topk, max_seqlen_k, max_k_blocks
        );
    } 
    else if (kBlockM == 64 && kBlockN == 64) {
        prepare_sparse_mask_kernel<64, 64><<<grid, block, smem_size, stream>>>(
            topk_indices.data_ptr<int>(),
            fine_mask.data_ptr<int>(),
            total_q, topk, max_seqlen_k, max_k_blocks
        );
    }
    else if (kBlockM == 128 && kBlockN == 64) {
        prepare_sparse_mask_kernel<128, 64><<<grid, block, smem_size, stream>>>(
            topk_indices.data_ptr<int>(),
            fine_mask.data_ptr<int>(),
            total_q, topk, max_seqlen_k, max_k_blocks
        );
    }
    else if (kBlockM == 64 && kBlockN == 128) {
         prepare_sparse_mask_kernel<64, 128><<<grid, block, smem_size, stream>>>(
            topk_indices.data_ptr<int>(),
            fine_mask.data_ptr<int>(),
            total_q, topk, max_seqlen_k, max_k_blocks
        );
    }
    else if (kBlockM == 192 && kBlockN == 128) {
         prepare_sparse_mask_kernel<192, 128><<<grid, block, smem_size, stream>>>(
            topk_indices.data_ptr<int>(),
            fine_mask.data_ptr<int>(),
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
