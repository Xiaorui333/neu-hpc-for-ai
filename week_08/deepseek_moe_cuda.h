#ifndef DEEPSEEK_MOE_CUDA_H
#define DEEPSEEK_MOE_CUDA_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Router kernel: Compute router logits
void deepseek_router_forward_cuda(
    const float* hidden_states,
    const float* router_weight,
    float* router_logits,
    int batch_size,
    int seq_len,
    int hidden_size,
    int num_experts,
    cudaStream_t stream = nullptr
);

// Top-k selection kernel
void deepseek_topk_forward_cuda(
    const float* router_logits,
    int* topk_indices,
    float* topk_weights,
    int batch_size,
    int seq_len,
    int num_experts,
    int top_k,
    cudaStream_t stream = nullptr
);

// Shared expert forward
void deepseek_shared_expert_forward_cuda(
    const float* hidden_states,
    const float* gate_weight,
    const float* up_weight,
    const float* down_weight,
    const float* activation,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_size,
    int intermediate_size,
    cudaStream_t stream = nullptr
);

// Routed expert forward (single expert)
void deepseek_routed_expert_forward_cuda(
    const float* hidden_states,
    const int* expert_indices,
    const float* expert_weights,
    const float* gate_weight,
    const float* up_weight,
    const float* down_weight,
    const float* activation,
    float* output,
    int num_tokens,
    int hidden_size,
    int intermediate_size,
    int expert_id,
    int num_experts,
    cudaStream_t stream = nullptr
);

// Scatter tokens to experts (for expert parallel)
void deepseek_scatter_tokens_cuda(
    const float* hidden_states,
    const int* expert_indices,
    const float* expert_weights,
    float* expert_inputs,
    float* expert_input_weights,
    int* expert_token_counts,
    int* expert_token_offsets,
    int* token_to_expert_pos,
    int batch_size,
    int seq_len,
    int hidden_size,
    int num_experts,
    int top_k,
    cudaStream_t stream = nullptr
);

// Gather expert outputs back
void deepseek_gather_expert_outputs_cuda(
    const float* expert_outputs,
    const int* expert_indices,
    const float* expert_weights,
    const int* expert_token_offsets,
    const int* token_to_expert_pos,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_size,
    int num_experts,
    int top_k,
    cudaStream_t stream = nullptr
);

// Activation function (Swish/SiLU)
void deepseek_swish_activation_cuda(
    const float* input,
    float* output,
    int num_elements,
    cudaStream_t stream = nullptr
);

#endif // DEEPSEEK_MOE_CUDA_H

