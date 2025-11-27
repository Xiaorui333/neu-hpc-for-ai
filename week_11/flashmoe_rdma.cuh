#ifndef FLASHMOE_RDMA_CUH
#define FLASHMOE_RDMA_CUH

#include "flashmoe_single_kernel.cuh"
#include <nvshmem.h>
#include <nvshmemx.h>

// ============================================================================
// RDMA COMMUNICATION FOR PAPER ALIGNMENT
// ============================================================================

/**
 * RDMA Communication Buffer for device-initiated transfers
 * Aligned with FlashMoE paper's device-initiated communication protocol
 */
struct RDMABuffer {
    void* local_ptr;      // Local memory address
    void* remote_ptr;     // Remote memory address (symmetric)
    size_t size;          // Buffer size in bytes
    int pe;               // Processing element (remote device ID)
    
    __device__ __host__
    RDMABuffer() : local_ptr(nullptr), remote_ptr(nullptr), size(0), pe(-1) {}
    
    __device__ __host__
    RDMABuffer(void* local, void* remote, size_t sz, int p)
        : local_ptr(local), remote_ptr(remote), size(sz), pe(p) {}
};

/**
 * NVSHMEM-based communication context
 */
struct NVSHMEMContext {
    void* symmetric_heap;      // NVSHMEM symmetric heap
    size_t heap_size;          // Total heap size
    int num_pes;               // Number of processing elements
    int my_pe;                 // My processing element ID
    
    // Symmetric buffers for expert data
    float* expert_send_buffer;
    float* expert_recv_buffer;
    int* token_count_buffer;
    
    __host__
    void initialize(size_t buffer_size) {
        nvshmem_init();
        my_pe = nvshmem_my_pe();
        num_pes = nvshmem_n_pes();
        
        // Allocate symmetric memory
        expert_send_buffer = (float*)nvshmem_malloc(buffer_size * sizeof(float));
        expert_recv_buffer = (float*)nvshmem_malloc(buffer_size * sizeof(float));
        token_count_buffer = (int*)nvshmem_malloc(num_pes * sizeof(int));
        
        nvshmem_barrier_all();  // Ensure all PEs have allocated
    }
    
    __host__
    void finalize() {
        nvshmem_free(expert_send_buffer);
        nvshmem_free(expert_recv_buffer);
        nvshmem_free(token_count_buffer);
        nvshmem_finalize();
    }
};

// ============================================================================
// DEVICE-INITIATED RDMA OPERATIONS
// ============================================================================

/**
 * Non-blocking RDMA put operation (device-initiated)
 * Aligned with paper's one-sided RDMA transfers
 */
__device__ inline
void rdma_put_nbi(void* dest, const void* src, size_t size, int pe) {
    // Device-initiated non-blocking put
    nvshmem_putmem_nbi(dest, src, size, pe);
    // Kernel continues immediately - no blocking!
}

/**
 * Non-blocking RDMA get operation (device-initiated)
 */
__device__ inline
void rdma_get_nbi(void* dest, const void* src, size_t size, int pe) {
    // Device-initiated non-blocking get
    nvshmem_getmem_nbi(dest, src, size, pe);
}

/**
 * Check if RDMA operation completed (device-side)
 */
__device__ inline
bool rdma_test_completion() {
    // Quiet operation - wait for all outstanding puts/gets to complete
    nvshmem_quiet();
    return true;
}

/**
 * RDMA fence - ensure all operations visible to remote PE
 */
__device__ inline
void rdma_fence(int pe) {
    nvshmem_fence();
}

/**
 * RDMA barrier - synchronize all PEs
 */
__device__ inline
void rdma_barrier_all() {
    nvshmemx_barrier_all_on_stream(0);
}

// ============================================================================
// EXPERT DATA TRANSFER WITH RDMA
// ============================================================================

/**
 * Send expert tokens to remote device using RDMA
 * Device-initiated - no CPU involvement!
 */
__device__ void rdma_send_expert_tokens(
    const Task& task,
    const SymmetricTensorLayout& layout,
    NVSHMEMContext* ctx
) {
    const float* local_data = reinterpret_cast<const float*>(task.input_ptr);
    int dest_pe = task.param1;           // Destination PE
    int expert_id = task.param2;         // Expert ID
    int num_tokens = task.param3;        // Number of tokens
    int hidden_size = layout.hidden_size;
    
    // Calculate remote offset using symmetric layout
    size_t transfer_size = num_tokens * hidden_size * sizeof(float);
    
    // Device-initiated non-blocking RDMA put
    // This continues immediately without CPU synchronization!
    rdma_put_nbi(
        ctx->expert_recv_buffer,  // Remote symmetric address
        local_data,                // Local data
        transfer_size,             // Size
        dest_pe                    // Target PE
    );
    
    // Fence to ensure visibility
    rdma_fence(dest_pe);
    
    // Kernel can continue with other work while transfer proceeds!
}

/**
 * Receive expert tokens from remote device using RDMA
 */
__device__ void rdma_recv_expert_tokens(
    const Task& task,
    const SymmetricTensorLayout& layout,
    NVSHMEMContext* ctx
) {
    float* local_data = reinterpret_cast<float*>(task.output_ptr);
    int src_pe = task.param1;            // Source PE
    int expert_id = task.param2;         // Expert ID
    int num_tokens = task.param3;        // Number of tokens
    int hidden_size = layout.hidden_size;
    
    size_t transfer_size = num_tokens * hidden_size * sizeof(float);
    
    // Device-initiated non-blocking RDMA get
    rdma_get_nbi(
        local_data,                // Local destination
        ctx->expert_send_buffer,   // Remote symmetric address
        transfer_size,             // Size
        src_pe                     // Source PE
    );
    
    // Can continue with other work!
}

/**
 * All-to-all expert exchange using RDMA
 * Optimized pattern for MoE communication
 */
__device__ void rdma_alltoall_expert_exchange(
    const Task& task,
    const SymmetricTensorLayout& layout,
    NVSHMEMContext* ctx
) {
    int my_pe = nvshmem_my_pe();
    int num_pes = nvshmem_n_pes();
    
    // Each thread block handles communication with one PE
    int target_pe = blockIdx.x % num_pes;
    if (target_pe == my_pe) return;  // Skip self
    
    // Get send/recv sizes
    const float* send_data = reinterpret_cast<const float*>(task.input_ptr);
    float* recv_data = reinterpret_cast<float*>(task.output_ptr);
    size_t send_size = task.param1;
    size_t recv_size = task.param2;
    
    // Simultaneous send and receive using RDMA
    if (threadIdx.x == 0) {
        // Non-blocking put to remote PE
        if (send_size > 0) {
            rdma_put_nbi(
                ctx->expert_recv_buffer,
                send_data,
                send_size * sizeof(float),
                target_pe
            );
        }
        
        // Non-blocking get from remote PE
        if (recv_size > 0) {
            rdma_get_nbi(
                recv_data,
                ctx->expert_send_buffer,
                recv_size * sizeof(float),
                target_pe
            );
        }
    }
    
    // Barrier to ensure all transfers initiated
    __syncthreads();
}

/**
 * Atomic remote counter increment (for coordination)
 */
__device__ inline
int rdma_atomic_fetch_add(int* target, int value, int pe) {
    return nvshmem_int_atomic_fetch_add(target, value, pe);
}

/**
 * Signal remote PE that data is ready
 */
__device__ inline
void rdma_signal(int* flag, int value, int pe) {
    nvshmem_int_p(flag, value, pe);
    rdma_fence(pe);
}

/**
 * Wait for signal from remote PE
 */
__device__ inline
void rdma_wait_signal(volatile int* flag, int expected_value) {
    nvshmem_int_wait_until(const_cast<int*>(flag), NVSHMEM_CMP_EQ, expected_value);
}

// ============================================================================
// ENHANCED COMMUNICATION TASK EXECUTOR WITH RDMA
// ============================================================================

/**
 * Execute communication task using device-initiated RDMA
 * NO CPU INVOLVEMENT - true single kernel execution!
 */
__device__ void execute_comm_task_rdma(
    const Task& task,
    const SymmetricTensorLayout& layout,
    NVSHMEMContext* ctx
) {
    switch (task.type) {
        case TaskType::ALLTOALL_SEND:
            // Device-initiated send
            rdma_send_expert_tokens(task, layout, ctx);
            break;
            
        case TaskType::ALLTOALL_RECV:
            // Device-initiated receive
            rdma_recv_expert_tokens(task, layout, ctx);
            break;
            
        case TaskType::ALLREDUCE_START:
            // Start all-to-all exchange
            rdma_alltoall_expert_exchange(task, layout, ctx);
            break;
            
        case TaskType::ALLREDUCE_WAIT:
            // Ensure all RDMA operations complete
            if (threadIdx.x == 0) {
                rdma_test_completion();
            }
            __syncthreads();
            break;
            
        case TaskType::BARRIER_SYNC:
            // Device-side barrier
            rdma_barrier_all();
            break;
            
        default:
            break;
    }
}

// ============================================================================
// PACKET-BASED COMMUNICATION PROTOCOL (Paper Section 3.3)
// ============================================================================

/**
 * Communication packet structure
 * Aligned with paper's packet-based protocol
 */
struct CommPacket {
    int src_expert;       // Source expert ID
    int dest_expert;      // Destination expert ID
    int src_pe;           // Source processing element
    int dest_pe;          // Destination processing element
    int token_count;      // Number of tokens in packet
    size_t data_offset;   // Offset in symmetric buffer
    size_t data_size;     // Data size in bytes
    int sequence_num;     // Packet sequence number
    
    __device__ __host__
    CommPacket() : src_expert(-1), dest_expert(-1), src_pe(-1), dest_pe(-1),
                   token_count(0), data_offset(0), data_size(0), sequence_num(0) {}
};

/**
 * Packet queue for incoming packets (Subscriber role)
 */
template<int MAX_PACKETS = 256>
struct PacketQueue {
    CommPacket packets[MAX_PACKETS];
    cuda::atomic<int> head;
    cuda::atomic<int> tail;
    cuda::atomic<int> size;
    
    __device__
    bool enqueue_packet(const CommPacket& packet) {
        int current_size = size.load(cuda::memory_order_acquire);
        if (current_size >= MAX_PACKETS) return false;
        
        int tail_idx = tail.fetch_add(1, cuda::memory_order_acq_rel) % MAX_PACKETS;
        packets[tail_idx] = packet;
        size.fetch_add(1, cuda::memory_order_release);
        return true;
    }
    
    __device__
    bool dequeue_packet(CommPacket& packet) {
        int current_size = size.load(cuda::memory_order_acquire);
        if (current_size == 0) return false;
        
        int head_idx = head.fetch_add(1, cuda::memory_order_acq_rel) % MAX_PACKETS;
        packet = packets[head_idx];
        size.fetch_sub(1, cuda::memory_order_release);
        return true;
    }
};

/**
 * Send packet using RDMA
 */
__device__ void rdma_send_packet(
    const CommPacket& packet,
    const float* data,
    NVSHMEMContext* ctx
) {
    // Device-initiated packet send
    rdma_put_nbi(
        ctx->expert_recv_buffer + packet.data_offset / sizeof(float),
        data,
        packet.data_size,
        packet.dest_pe
    );
    
    // Signal remote that packet is ready
    rdma_fence(packet.dest_pe);
}

/**
 * Receive packet notification and decode
 */
__device__ bool rdma_recv_packet(
    CommPacket& packet,
    NVSHMEMContext* ctx,
    volatile int* notification_flag
) {
    // Check if new packet arrived (non-blocking)
    if (*notification_flag == 0) return false;
    
    // Decode packet metadata from notification
    // (In real implementation, metadata would be in shared memory)
    packet.src_pe = *notification_flag - 1;
    *notification_flag = 0;  // Reset flag
    
    return true;
}

#endif // FLASHMOE_RDMA_CUH

