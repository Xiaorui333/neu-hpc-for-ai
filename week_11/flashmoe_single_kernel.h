#ifndef FLASHMOE_SINGLE_KERNEL_H
#define FLASHMOE_SINGLE_KERNEL_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * FlashMoE Single Kernel Forward Pass
 * 
 * This function implements the MoE operator in a single persistent kernel
 * with non-blocking communication to overlap computation and communication.
 * 
 * Features:
 * - Symmetric tensor layout for efficient multi-GPU distribution
 * - Task abstraction for flexible scheduling
 * - Task queue for dynamic work assignment
 * - Non-blocking communication overlap
 * 
 * @param hidden_states Input tensor (batch_size * seq_len, hidden_size)
 * @param router_weight Router weight matrix (num_experts, hidden_size)
 * @param expert_gate_weights Gate projection weights for all experts
 * @param expert_up_weights Up projection weights for all experts
 * @param expert_down_weights Down projection weights for all experts
 * @param output Output tensor (batch_size * seq_len, hidden_size)
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param hidden_size Hidden dimension size
 * @param num_experts Total number of experts across all devices
 * @param num_local_experts Number of experts on this device
 * @param intermediate_size Intermediate dimension size
 * @param top_k Number of experts to route to per token
 * @param num_devices Total number of GPU devices
 * @param device_id Current device ID
 * @param stream CUDA stream for execution
 */
void flashmoe_single_kernel_forward(
    const float* hidden_states,
    const float* router_weight,
    const float* expert_gate_weights,
    const float* expert_up_weights,
    const float* expert_down_weights,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_size,
    int num_experts,
    int num_local_experts,
    int intermediate_size,
    int top_k,
    int num_devices,
    int device_id,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // FLASHMOE_SINGLE_KERNEL_H

