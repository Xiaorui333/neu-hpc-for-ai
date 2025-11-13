#include "deepseek_moe_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cmath>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Activation: Swish/SiLU = x * sigmoid(x)
__device__ inline float swish_activation(float x) {
    return x / (1.0f + expf(-x));
}

// Router forward kernel: Compute router logits
__global__ void router_forward_kernel(
    const float* __restrict__ hidden_states,
    const float* __restrict__ router_weight,
    float* __restrict__ router_logits,
    int batch_seq_size,
    int hidden_size,
    int num_experts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_seq_size) return;

    const float* h = hidden_states + idx * hidden_size;
    float* logits = router_logits + idx * num_experts;

    // Compute dot product for each expert
    for (int e = 0; e < num_experts; e++) {
        float sum = 0.0f;
        const float* w = router_weight + e * hidden_size;
        
        #pragma unroll 4
        for (int i = 0; i < hidden_size; i++) {
            sum += h[i] * w[i];
        }
        logits[e] = sum;
    }
}

// Top-k selection using shared memory
__global__ void topk_selection_kernel(
    const float* __restrict__ router_logits,
    int* __restrict__ topk_indices,
    float* __restrict__ topk_weights,
    int batch_seq_size,
    int num_experts,
    int top_k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_seq_size) return;

    const float* logits = router_logits + idx * num_experts;
    int* indices = topk_indices + idx * top_k;
    float* weights = topk_weights + idx * top_k;

    // Simple selection sort for top-k (for small k, this is efficient)
    // For larger k, use bitonic sort or heap
    for (int i = 0; i < top_k; i++) {
        float max_val = -1e30f;
        int max_idx = -1;
        
        for (int j = 0; j < num_experts; j++) {
            bool already_selected = false;
            for (int prev = 0; prev < i; prev++) {
                if (indices[prev] == j) {
                    already_selected = true;
                    break;
                }
            }
            
            if (!already_selected && logits[j] > max_val) {
                max_val = logits[j];
                max_idx = j;
            }
        }
        
        indices[i] = max_idx;
        weights[i] = max_val;
    }

    // Softmax normalization
    float max_val = weights[0];
    for (int i = 1; i < top_k; i++) {
        if (weights[i] > max_val) max_val = weights[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < top_k; i++) {
        weights[i] = expf(weights[i] - max_val);
        sum += weights[i];
    }

    for (int i = 0; i < top_k; i++) {
        weights[i] /= sum;
    }
}

// Shared expert MLP forward
__global__ void shared_expert_forward_kernel(
    const float* __restrict__ hidden_states,
    const float* __restrict__ gate_weight,
    const float* __restrict__ up_weight,
    const float* __restrict__ down_weight,
    float* __restrict__ output,
    int batch_seq_size,
    int hidden_size,
    int intermediate_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_seq_size) return;

    const float* h = hidden_states + idx * hidden_size;
    float* out = output + idx * hidden_size;

    extern __shared__ float shared_mem[];
    float* gate_output = shared_mem;
    float* up_output = gate_output + intermediate_size;

    // Gate projection
    for (int i = threadIdx.x; i < intermediate_size; i += blockDim.x) {
        float sum = 0.0f;
        const float* w = gate_weight + i * hidden_size;
        for (int j = 0; j < hidden_size; j++) {
            sum += h[j] * w[j];
        }
        gate_output[i] = sum;
    }
    __syncthreads();

    // Up projection
    for (int i = threadIdx.x; i < intermediate_size; i += blockDim.x) {
        float sum = 0.0f;
        const float* w = up_weight + i * hidden_size;
        for (int j = 0; j < hidden_size; j++) {
            sum += h[j] * w[j];
        }
        up_output[i] = sum;
    }
    __syncthreads();

    // Activation: gate * up (Swish applied to gate)
    for (int i = threadIdx.x; i < intermediate_size; i += blockDim.x) {
        gate_output[i] = swish_activation(gate_output[i]) * up_output[i];
    }
    __syncthreads();

    // Down projection
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float sum = 0.0f;
        const float* w = down_weight + i * intermediate_size;
        for (int j = 0; j < intermediate_size; j++) {
            sum += gate_output[j] * w[j];
        }
        out[i] = sum;
    }
}

// Routed expert forward (processes tokens for a specific expert)
__global__ void routed_expert_forward_kernel(
    const float* __restrict__ expert_inputs,
    const float* __restrict__ gate_weight,
    const float* __restrict__ up_weight,
    const float* __restrict__ down_weight,
    const float* __restrict__ expert_weights,
    float* __restrict__ expert_outputs,
    int num_tokens,
    int hidden_size,
    int intermediate_size
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    const float* h = expert_inputs + token_idx * hidden_size;
    float* out = expert_outputs + token_idx * hidden_size;
    float weight = expert_weights[token_idx];

    extern __shared__ float shared_mem[];
    float* gate_output = shared_mem;
    float* up_output = gate_output + intermediate_size;

    // Gate projection
    for (int i = threadIdx.x; i < intermediate_size; i += blockDim.x) {
        float sum = 0.0f;
        const float* w = gate_weight + i * hidden_size;
        for (int j = 0; j < hidden_size; j++) {
            sum += h[j] * w[j];
        }
        gate_output[i] = sum;
    }
    __syncthreads();

    // Up projection
    for (int i = threadIdx.x; i < intermediate_size; i += blockDim.x) {
        float sum = 0.0f;
        const float* w = up_weight + i * hidden_size;
        for (int j = 0; j < hidden_size; j++) {
            sum += h[j] * w[j];
        }
        up_output[i] = sum;
    }
    __syncthreads();

    // Activation: gate * up (Swish applied to gate)
    for (int i = threadIdx.x; i < intermediate_size; i += blockDim.x) {
        gate_output[i] = swish_activation(gate_output[i]) * up_output[i];
    }
    __syncthreads();

    // Down projection with expert weight scaling
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float sum = 0.0f;
        const float* w = down_weight + i * intermediate_size;
        for (int j = 0; j < intermediate_size; j++) {
            sum += gate_output[j] * w[j];
        }
        out[i] = sum * weight;
    }
}

// Scatter tokens to experts (for expert parallel distribution)
// Also stores token-to-expert mapping for gather
__global__ void scatter_tokens_kernel(
    const float* __restrict__ hidden_states,
    const int* __restrict__ expert_indices,
    const float* __restrict__ expert_weights,
    float* __restrict__ expert_inputs,
    float* __restrict__ expert_input_weights,
    int* __restrict__ expert_token_counts,
    int* __restrict__ expert_token_offsets,
    int* __restrict__ token_to_expert_pos,  // Maps (token_idx, expert_idx) -> position in expert buffer
    int batch_seq_size,
    int hidden_size,
    int num_experts,
    int top_k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_seq_size) return;

    const int* indices = expert_indices + idx * top_k;
    const float* weights = expert_weights + idx * top_k;
    const float* h = hidden_states + idx * hidden_size;

    // For each selected expert, scatter the token
    for (int k = 0; k < top_k; k++) {
        int expert_id = indices[k];
        float weight = weights[k];
        
        // Atomic increment to get token position for this expert
        int token_pos = atomicAdd(&expert_token_counts[expert_id], 1);
        int token_idx = expert_token_offsets[expert_id] + token_pos;
        
        // Store mapping: (token_idx, expert_idx) -> position in expert buffer
        token_to_expert_pos[idx * num_experts + expert_id] = token_pos;
        
        // Copy hidden state to expert input buffer
        float* expert_input = expert_inputs + token_idx * hidden_size;
        for (int i = 0; i < hidden_size; i++) {
            expert_input[i] = h[i];
        }
        
        // Store weight
        expert_input_weights[token_idx] = weight;
    }
}

// Gather expert outputs back
__global__ void gather_expert_outputs_kernel(
    const float* __restrict__ expert_outputs,
    const int* __restrict__ expert_indices,
    const float* __restrict__ expert_weights,
    const int* __restrict__ expert_token_offsets,
    const int* __restrict__ token_to_expert_pos,
    float* __restrict__ output,
    int batch_seq_size,
    int hidden_size,
    int num_experts,
    int top_k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_seq_size) return;

    float* out = output + idx * hidden_size;
    const int* indices = expert_indices + idx * top_k;
    const float* weights = expert_weights + idx * top_k;

    // Initialize output to zero
    for (int i = 0; i < hidden_size; i++) {
        out[i] = 0.0f;
    }

    // Accumulate outputs from all selected experts
    for (int k = 0; k < top_k; k++) {
        int expert_id = indices[k];
        float weight = weights[k];
        
        // Find token position in expert output using the mapping
        int token_pos = token_to_expert_pos[idx * num_experts + expert_id];
        int expert_token_idx = expert_token_offsets[expert_id] + token_pos;
        const float* expert_out = expert_outputs + expert_token_idx * hidden_size;
        
        // Accumulate with routing weight
        for (int i = 0; i < hidden_size; i++) {
            out[i] += expert_out[i] * weight;
        }
    }
}

// Swish activation kernel
__global__ void swish_activation_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    output[idx] = swish_activation(input[idx]);
}

// Host wrapper functions
void deepseek_router_forward_cuda(
    const float* hidden_states,
    const float* router_weight,
    float* router_logits,
    int batch_size,
    int seq_len,
    int hidden_size,
    int num_experts,
    cudaStream_t stream
) {
    int batch_seq_size = batch_size * seq_len;
    int num_blocks = (batch_seq_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    router_forward_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        hidden_states, router_weight, router_logits,
        batch_seq_size, hidden_size, num_experts
    );
}

void deepseek_topk_forward_cuda(
    const float* router_logits,
    int* topk_indices,
    float* topk_weights,
    int batch_size,
    int seq_len,
    int num_experts,
    int top_k,
    cudaStream_t stream
) {
    int batch_seq_size = batch_size * seq_len;
    int num_blocks = (batch_seq_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    topk_selection_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        router_logits, topk_indices, topk_weights,
        batch_seq_size, num_experts, top_k
    );
}

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
    cudaStream_t stream
) {
    int batch_seq_size = batch_size * seq_len;
    int num_blocks = (batch_seq_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t shared_mem_size = 2 * intermediate_size * sizeof(float);
    
    shared_expert_forward_kernel<<<num_blocks, BLOCK_SIZE, shared_mem_size, stream>>>(
        hidden_states, gate_weight, up_weight, down_weight, output,
        batch_seq_size, hidden_size, intermediate_size
    );
}

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
    cudaStream_t stream
) {
    int num_blocks = (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t shared_mem_size = 2 * intermediate_size * sizeof(float);
    
    routed_expert_forward_kernel<<<num_blocks, BLOCK_SIZE, shared_mem_size, stream>>>(
        hidden_states, gate_weight, up_weight, down_weight, expert_weights, output,
        num_tokens, hidden_size, intermediate_size
    );
}

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
    cudaStream_t stream
) {
    int batch_seq_size = batch_size * seq_len;
    int num_blocks = (batch_seq_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    scatter_tokens_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        hidden_states, expert_indices, expert_weights, expert_inputs,
        expert_input_weights, expert_token_counts, expert_token_offsets,
        token_to_expert_pos, batch_seq_size, hidden_size, num_experts, top_k
    );
}

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
    cudaStream_t stream
) {
    int batch_seq_size = batch_size * seq_len;
    int num_blocks = (batch_seq_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    gather_expert_outputs_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        expert_outputs, expert_indices, expert_weights, expert_token_offsets,
        token_to_expert_pos, output, batch_seq_size, hidden_size, num_experts, top_k
    );
}

void deepseek_swish_activation_cuda(
    const float* input,
    float* output,
    int num_elements,
    cudaStream_t stream
) {
    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    swish_activation_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        input, output, num_elements
    );
}

