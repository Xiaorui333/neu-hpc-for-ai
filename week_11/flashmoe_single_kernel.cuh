#ifndef FLASHMOE_SINGLE_KERNEL_CUH
#define FLASHMOE_SINGLE_KERNEL_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cuda/atomic>

namespace cg = cooperative_groups;

// ============================================================================
// DELIVERABLE 1: SYMMETRIC TENSOR LAYOUT
// ============================================================================

/**
 * Symmetric Tensor Layout for uniformly distributed tensors across GPUs.
 * Ensures efficient data access and communication patterns.
 */
struct SymmetricTensorLayout {
    // Tensor dimensions
    int batch_size;
    int seq_len;
    int hidden_size;
    int num_experts;
    
    // Layout configuration
    int num_devices;          // Number of GPUs
    int device_id;            // Current device ID
    int experts_per_device;   // Experts assigned to each device
    
    // Memory layout parameters
    int tokens_per_device;    // Tokens per device (for tensor parallelism)
    int local_expert_start;   // Starting expert index for this device
    int local_expert_end;     // Ending expert index for this device
    
    // Strided memory layout for coalesced access
    int token_stride;         // Stride between tokens
    int expert_stride;        // Stride between expert data
    int hidden_stride;        // Stride for hidden dimensions
    
    __device__ __host__
    SymmetricTensorLayout(
        int bs, int sl, int hs, int ne, 
        int ndev, int devid
    ) : batch_size(bs), seq_len(sl), hidden_size(hs), num_experts(ne),
        num_devices(ndev), device_id(devid) {
        
        // Distribute experts evenly across devices
        experts_per_device = (num_experts + num_devices - 1) / num_devices;
        local_expert_start = device_id * experts_per_device;
        local_expert_end = min(local_expert_start + experts_per_device, num_experts);
        
        // Calculate strides for symmetric layout
        int total_tokens = batch_size * seq_len;
        tokens_per_device = (total_tokens + num_devices - 1) / num_devices;
        
        token_stride = hidden_size;
        expert_stride = tokens_per_device * hidden_size;
        hidden_stride = 1;
    }
    
    // Get global token index from local index
    __device__ __host__ inline
    int get_global_token_idx(int local_token_idx) const {
        return device_id * tokens_per_device + local_token_idx;
    }
    
    // Get local expert index from global expert index
    __device__ __host__ inline
    int get_local_expert_idx(int global_expert_idx) const {
        return global_expert_idx - local_expert_start;
    }
    
    // Check if expert is local to this device
    __device__ __host__ inline
    bool is_local_expert(int expert_idx) const {
        return expert_idx >= local_expert_start && expert_idx < local_expert_end;
    }
    
    // Get memory offset for token in symmetric layout
    __device__ __host__ inline
    int get_token_offset(int token_idx, int hidden_idx) const {
        return token_idx * token_stride + hidden_idx * hidden_stride;
    }
    
    // Get memory offset for expert data
    __device__ __host__ inline
    int get_expert_offset(int expert_idx, int token_idx, int hidden_idx) const {
        int local_expert = get_local_expert_idx(expert_idx);
        return local_expert * expert_stride + 
               token_idx * token_stride + 
               hidden_idx * hidden_stride;
    }
};


// ============================================================================
// DELIVERABLE 2: TASK ABSTRACTION
// ============================================================================

/**
 * Task types for different stages of MoE computation
 */
enum class TaskType : uint8_t {
    ROUTER_COMPUTE,      // Compute router logits
    TOPK_SELECT,         // Select top-k experts
    SCATTER_TOKENS,      // Scatter tokens to experts
    EXPERT_COMPUTE,      // Compute expert forward pass
    GATHER_OUTPUTS,      // Gather expert outputs
    ALLTOALL_SEND,       // Non-blocking AllToAll send
    ALLTOALL_RECV,       // Non-blocking AllToAll receive
    ALLREDUCE_START,     // Start non-blocking AllReduce
    ALLREDUCE_WAIT,      // Wait for AllReduce completion
    BARRIER_SYNC         // Synchronization barrier
};

/**
 * Task status for tracking task lifecycle
 */
enum class TaskStatus : uint8_t {
    PENDING,       // Task is queued, not started
    READY,         // Task is ready to execute (dependencies met)
    IN_PROGRESS,   // Task is currently executing
    WAITING_COMM,  // Task is waiting for communication
    COMPLETED,     // Task has completed successfully
    FAILED         // Task failed (for error handling)
};

/**
 * Task structure representing a unit of work
 * This abstraction allows flexible scheduling and dependency tracking
 */
struct Task {
    TaskType type;
    TaskStatus status;
    
    // Task identification
    uint32_t task_id;
    uint32_t priority;  // Higher priority = execute first
    
    // Dependency tracking
    uint32_t num_dependencies;
    uint32_t dependencies[4];  // Max 4 dependencies per task
    
    // Task-specific data pointers
    void* input_ptr;
    void* output_ptr;
    void* weight_ptr;
    void* aux_ptr;
    
    // Task parameters
    int param1;  // Generic parameters for different task types
    int param2;
    int param3;
    int param4;
    
    // Communication handles (for non-blocking operations)
    void* comm_request;
    
    // Timing information
    uint64_t start_time;
    uint64_t end_time;
    
    __device__ __host__
    Task() : type(TaskType::ROUTER_COMPUTE), 
             status(TaskStatus::PENDING),
             task_id(0), priority(0), 
             num_dependencies(0),
             input_ptr(nullptr), output_ptr(nullptr), 
             weight_ptr(nullptr), aux_ptr(nullptr),
             param1(0), param2(0), param3(0), param4(0),
             comm_request(nullptr),
             start_time(0), end_time(0) {
        for (int i = 0; i < 4; i++) dependencies[i] = 0;
    }
    
    // Check if all dependencies are met
    __device__ inline
    bool are_dependencies_met(const TaskStatus* task_statuses) const {
        for (uint32_t i = 0; i < num_dependencies; i++) {
            if (task_statuses[dependencies[i]] != TaskStatus::COMPLETED) {
                return false;
            }
        }
        return true;
    }
    
    // Add a dependency
    __device__ __host__ inline
    void add_dependency(uint32_t dep_task_id) {
        if (num_dependencies < 4) {
            dependencies[num_dependencies++] = dep_task_id;
        }
    }
};


// ============================================================================
// DELIVERABLE 3: TASK QUEUE
// ============================================================================

/**
 * Lock-free task queue for managing concurrent task execution
 * Uses atomic operations for thread-safe enqueue/dequeue
 */
template<int MAX_TASKS = 1024>
class TaskQueue {
private:
    Task tasks[MAX_TASKS];
    TaskStatus task_statuses[MAX_TASKS];
    
    // Queue management
    cuda::atomic<uint32_t, cuda::thread_scope_device> head;
    cuda::atomic<uint32_t, cuda::thread_scope_device> tail;
    cuda::atomic<uint32_t, cuda::thread_scope_device> size;
    cuda::atomic<uint32_t, cuda::thread_scope_device> completed_count;
    
    // Priority queue indices (for fast priority-based dequeue)
    cuda::atomic<uint32_t, cuda::thread_scope_device> ready_tasks[MAX_TASKS];
    cuda::atomic<uint32_t, cuda::thread_scope_device> num_ready;
    
public:
    __device__ __host__
    TaskQueue() : head(0), tail(0), size(0), completed_count(0), num_ready(0) {
        for (int i = 0; i < MAX_TASKS; i++) {
            task_statuses[i] = TaskStatus::PENDING;
            ready_tasks[i].store(0, cuda::memory_order_relaxed);
        }
    }
    
    // Enqueue a task (thread-safe)
    __device__ inline
    bool enqueue(const Task& task) {
        uint32_t current_size = size.load(cuda::memory_order_acquire);
        if (current_size >= MAX_TASKS) {
            return false;  // Queue full
        }
        
        uint32_t tail_idx = tail.fetch_add(1, cuda::memory_order_acq_rel) % MAX_TASKS;
        tasks[tail_idx] = task;
        tasks[tail_idx].task_id = tail_idx;
        task_statuses[tail_idx] = TaskStatus::PENDING;
        size.fetch_add(1, cuda::memory_order_release);
        
        return true;
    }
    
    // Try to dequeue a ready task (priority-based)
    __device__ inline
    bool try_dequeue_ready(Task& task, uint32_t& task_id) {
        // Check if there are ready tasks
        uint32_t ready_count = num_ready.load(cuda::memory_order_acquire);
        if (ready_count == 0) {
            // Scan for newly ready tasks
            update_ready_tasks();
            ready_count = num_ready.load(cuda::memory_order_acquire);
            if (ready_count == 0) {
                return false;
            }
        }
        
        // Try to claim a ready task
        for (int i = 0; i < MAX_TASKS; i++) {
            uint32_t idx = ready_tasks[i].load(cuda::memory_order_acquire);
            if (idx > 0 && task_statuses[idx - 1] == TaskStatus::READY) {
                // Try to claim this task
                TaskStatus expected = TaskStatus::READY;
                if (__atomic_compare_exchange_n(
                    reinterpret_cast<uint8_t*>(&task_statuses[idx - 1]),
                    reinterpret_cast<uint8_t*>(&expected),
                    static_cast<uint8_t>(TaskStatus::IN_PROGRESS),
                    false,
                    __ATOMIC_ACQ_REL,
                    __ATOMIC_ACQUIRE)) {
                    
                    task = tasks[idx - 1];
                    task_id = idx - 1;
                    ready_tasks[i].store(0, cuda::memory_order_release);
                    num_ready.fetch_sub(1, cuda::memory_order_release);
                    return true;
                }
            }
        }
        
        return false;
    }
    
    // Update list of ready tasks (scan pending tasks)
    __device__ inline
    void update_ready_tasks() {
        uint32_t current_size = size.load(cuda::memory_order_acquire);
        
        for (uint32_t i = 0; i < current_size && i < MAX_TASKS; i++) {
            if (task_statuses[i] == TaskStatus::PENDING) {
                // Check if dependencies are met
                if (tasks[i].are_dependencies_met(task_statuses)) {
                    TaskStatus expected = TaskStatus::PENDING;
                    if (__atomic_compare_exchange_n(
                        reinterpret_cast<uint8_t*>(&task_statuses[i]),
                        reinterpret_cast<uint8_t*>(&expected),
                        static_cast<uint8_t>(TaskStatus::READY),
                        false,
                        __ATOMIC_ACQ_REL,
                        __ATOMIC_ACQUIRE)) {
                        
                        // Add to ready queue
                        uint32_t ready_idx = num_ready.fetch_add(1, cuda::memory_order_acq_rel);
                        if (ready_idx < MAX_TASKS) {
                            ready_tasks[ready_idx].store(i + 1, cuda::memory_order_release);
                        }
                    }
                }
            }
        }
    }
    
    // Mark task as completed
    __device__ inline
    void mark_completed(uint32_t task_id) {
        if (task_id < MAX_TASKS) {
            task_statuses[task_id] = TaskStatus::COMPLETED;
            completed_count.fetch_add(1, cuda::memory_order_release);
        }
    }
    
    // Mark task as waiting for communication
    __device__ inline
    void mark_waiting_comm(uint32_t task_id) {
        if (task_id < MAX_TASKS) {
            task_statuses[task_id] = TaskStatus::WAITING_COMM;
        }
    }
    
    // Get task status
    __device__ inline
    TaskStatus get_status(uint32_t task_id) const {
        if (task_id < MAX_TASKS) {
            return task_statuses[task_id];
        }
        return TaskStatus::FAILED;
    }
    
    // Check if all tasks are completed
    __device__ inline
    bool all_completed() const {
        return completed_count.load(cuda::memory_order_acquire) == 
               size.load(cuda::memory_order_acquire);
    }
    
    // Get number of completed tasks
    __device__ inline
    uint32_t get_completed_count() const {
        return completed_count.load(cuda::memory_order_acquire);
    }
    
    // Get total number of tasks
    __device__ inline
    uint32_t get_size() const {
        return size.load(cuda::memory_order_acquire);
    }
    
    // Reset queue (for reuse)
    __device__ inline
    void reset() {
        head.store(0, cuda::memory_order_release);
        tail.store(0, cuda::memory_order_release);
        size.store(0, cuda::memory_order_release);
        completed_count.store(0, cuda::memory_order_release);
        num_ready.store(0, cuda::memory_order_release);
        
        for (int i = 0; i < MAX_TASKS; i++) {
            task_statuses[i] = TaskStatus::PENDING;
            ready_tasks[i].store(0, cuda::memory_order_release);
        }
    }
};


// ============================================================================
// COMMUNICATION PRIMITIVES FOR NON-BLOCKING OPERATIONS
// ============================================================================

/**
 * Non-blocking communication request handle
 * Used to track asynchronous communication operations
 */
struct CommRequest {
    bool is_send;
    bool is_complete;
    int peer_device;
    void* buffer;
    size_t size;
    cudaStream_t stream;
    cudaEvent_t completion_event;
    
    __host__
    CommRequest() : is_send(false), is_complete(false), 
                   peer_device(-1), buffer(nullptr), 
                   size(0), stream(nullptr), completion_event(nullptr) {}
    
    __host__
    void initiate_send(void* buf, size_t sz, int peer, cudaStream_t strm) {
        is_send = true;
        is_complete = false;
        peer_device = peer;
        buffer = buf;
        size = sz;
        stream = strm;
        cudaEventCreate(&completion_event);
    }
    
    __host__
    void initiate_recv(void* buf, size_t sz, int peer, cudaStream_t strm) {
        is_send = false;
        is_complete = false;
        peer_device = peer;
        buffer = buf;
        size = sz;
        stream = strm;
        cudaEventCreate(&completion_event);
    }
    
    __host__
    bool test_completion() {
        if (is_complete) return true;
        cudaError_t err = cudaEventQuery(completion_event);
        if (err == cudaSuccess) {
            is_complete = true;
            return true;
        }
        return false;
    }
    
    __host__
    void wait() {
        if (!is_complete) {
            cudaEventSynchronize(completion_event);
            is_complete = true;
        }
    }
    
    __host__
    ~CommRequest() {
        if (completion_event) {
            cudaEventDestroy(completion_event);
        }
    }
};

#endif // FLASHMOE_SINGLE_KERNEL_CUH

