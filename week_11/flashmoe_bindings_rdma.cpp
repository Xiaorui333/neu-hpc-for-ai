#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cuda_runtime.h>

/**
 * Python bindings for FlashMoE RDMA version
 */

// Forward declaration of the CUDA kernel wrapper
extern "C" void flashmoe_rdma_forward(
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

// Python wrapper function
torch::Tensor flashmoe_rdma_forward_py(
    torch::Tensor hidden_states,
    torch::Tensor router_weight,
    torch::Tensor expert_gate_weights,
    torch::Tensor expert_up_weights,
    torch::Tensor expert_down_weights,
    int top_k,
    int num_experts,
    int num_devices
) {
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be a CUDA tensor");
    TORCH_CHECK(router_weight.is_cuda(), "router_weight must be a CUDA tensor");
    
    auto batch_size = hidden_states.size(0);
    auto seq_len = hidden_states.size(1);
    auto hidden_size = hidden_states.size(2);
    auto intermediate_size = expert_gate_weights.size(1);
    auto num_local_experts = num_experts / num_devices;
    int device_id = hidden_states.device().index();
    
    // Allocate output tensor
    auto output = torch::empty_like(hidden_states);
    
    // Get current CUDA stream from PyTorch (avoid conflicts with default stream)
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_id);
    
    // Call CUDA kernel
    flashmoe_rdma_forward(
        hidden_states.data_ptr<float>(),
        router_weight.data_ptr<float>(),
        expert_gate_weights.data_ptr<float>(),
        expert_up_weights.data_ptr<float>(),
        expert_down_weights.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, seq_len, hidden_size,
        num_experts, num_local_experts, intermediate_size, top_k,
        num_devices, device_id, stream
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, 
                "FlashMoE RDMA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

// Module binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flashmoe_rdma_forward", &flashmoe_rdma_forward_py, 
          "FlashMoE RDMA forward pass with device-initiated communication",
          py::arg("hidden_states"),
          py::arg("router_weight"),
          py::arg("expert_gate_weights"),
          py::arg("expert_up_weights"),
          py::arg("expert_down_weights"),
          py::arg("top_k") = 2,
          py::arg("num_experts") = 8,
          py::arg("num_devices") = 1);
    
    m.def("get_version", []() { return "1.0.0-rdma"; }, "Get FlashMoE RDMA version");
    m.def("has_nvshmem", []() { 
        #ifdef USE_NVSHMEM
        return true;
        #else
        return false;
        #endif
    }, "Check if NVSHMEM support is compiled");
}

