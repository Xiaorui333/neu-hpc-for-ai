#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include "deepseek_moe_cuda.h"

// Router forward
std::vector<torch::Tensor> router_forward(
    torch::Tensor hidden_states,
    torch::Tensor router_weight
) {
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be a CUDA tensor");
    TORCH_CHECK(router_weight.is_cuda(), "router_weight must be a CUDA tensor");
    
    // Ensure float32 and contiguous
    hidden_states = hidden_states.to(torch::kFloat32).contiguous();
    router_weight = router_weight.to(torch::kFloat32).contiguous();
    
    auto num_experts = router_weight.size(0);
    auto hidden_size = router_weight.size(1);
    
    // Ensure last dimension matches
    TORCH_CHECK(hidden_states.size(-1) == hidden_size, 
                "hidden_states last dimension must match router_weight");
    
    // Flatten to 2D for kernel (always work with flattened)
    auto hidden_states_flat = hidden_states.view({-1, hidden_size});
    int64_t batch_seq_size = hidden_states_flat.size(0);
    
    auto router_logits_flat = torch::zeros(
        {batch_seq_size, num_experts},
        torch::TensorOptions().dtype(hidden_states.dtype()).device(hidden_states.device())
    );
    
    auto stream = at::cuda::getCurrentCUDAStream();
    
    deepseek_router_forward_cuda(
        hidden_states_flat.data_ptr<float>(),
        router_weight.data_ptr<float>(),
        router_logits_flat.data_ptr<float>(),
        batch_seq_size, 1, hidden_size, num_experts, stream  // seq_len=1 for flattened
    );
    
    // Always return 2D, let Python handle reshape
    return {router_logits_flat};
}

// Top-k selection
std::vector<torch::Tensor> topk_forward(
    torch::Tensor router_logits,
    int top_k
) {
    TORCH_CHECK(router_logits.is_cuda(), "router_logits must be a CUDA tensor");
    
    // Ensure float32 and contiguous
    router_logits = router_logits.to(torch::kFloat32).contiguous();
    
    // Handle both 2D and 3D inputs
    int64_t batch_size, seq_len, batch_seq_size, num_experts;
    if (router_logits.dim() == 2) {
        // 2D input: (batch_seq_size, num_experts)
        batch_seq_size = router_logits.size(0);
        num_experts = router_logits.size(1);
        batch_size = 1;
        seq_len = batch_seq_size;
    } else if (router_logits.dim() == 3) {
        // 3D input: (batch_size, seq_len, num_experts)
        batch_size = router_logits.size(0);
        seq_len = router_logits.size(1);
        num_experts = router_logits.size(2);
        batch_seq_size = batch_size * seq_len;
    } else {
        TORCH_CHECK(false, "router_logits must be 2D or 3D");
    }
    
    // Flatten to 2D for kernel
    auto router_logits_flat = router_logits.view({batch_seq_size, num_experts});
    
    auto topk_indices_flat = torch::zeros(
        {batch_seq_size, top_k},
        torch::TensorOptions().dtype(torch::kInt32).device(router_logits.device())
    );
    auto topk_weights_flat = torch::zeros(
        {batch_seq_size, top_k},
        torch::TensorOptions().dtype(router_logits.dtype()).device(router_logits.device())
    );
    
    auto stream = at::cuda::getCurrentCUDAStream();
    
    deepseek_topk_forward_cuda(
        router_logits_flat.data_ptr<float>(),
        topk_indices_flat.data_ptr<int>(),
        topk_weights_flat.data_ptr<float>(),
        batch_seq_size, 1, num_experts, top_k, stream  // seq_len=1 for flattened
    );
    
    // Reshape back if original was 3D
    if (router_logits.dim() == 3) {
        auto topk_indices = topk_indices_flat.view({batch_size, seq_len, top_k});
        auto topk_weights = topk_weights_flat.view({batch_size, seq_len, top_k});
        return {topk_indices, topk_weights};
    } else {
        return {topk_indices_flat, topk_weights_flat};
    }
}

// Shared expert forward
torch::Tensor shared_expert_forward(
    torch::Tensor hidden_states,
    torch::Tensor gate_weight,
    torch::Tensor up_weight,
    torch::Tensor down_weight
) {
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be a CUDA tensor");
    
    // Ensure float32 and contiguous
    hidden_states = hidden_states.to(torch::kFloat32).contiguous();
    gate_weight = gate_weight.to(torch::kFloat32).contiguous();
    up_weight = up_weight.to(torch::kFloat32).contiguous();
    down_weight = down_weight.to(torch::kFloat32).contiguous();
    
    auto intermediate_size = gate_weight.size(0);
    auto hidden_size = gate_weight.size(1);
    
    // Ensure last dimension matches
    TORCH_CHECK(hidden_states.size(-1) == hidden_size, 
                "hidden_states last dimension must match gate_weight");
    
    // Flatten to 2D for kernel (always work with flattened)
    auto hidden_states_flat = hidden_states.view({-1, hidden_size});
    int64_t batch_seq_size = hidden_states_flat.size(0);
    
    auto output_flat = torch::zeros_like(hidden_states_flat);
    
    auto stream = at::cuda::getCurrentCUDAStream();
    
    deepseek_shared_expert_forward_cuda(
        hidden_states_flat.data_ptr<float>(),
        gate_weight.data_ptr<float>(),
        up_weight.data_ptr<float>(),
        down_weight.data_ptr<float>(),
        nullptr,  // activation (not used in this version)
        output_flat.data_ptr<float>(),
        batch_seq_size, 1, hidden_size, intermediate_size, stream  // seq_len=1 for flattened
    );
    
    // Reshape to match input shape
    if (hidden_states.dim() == 3) {
        auto batch_size = hidden_states.size(0);
        auto seq_len = hidden_states.size(1);
        return output_flat.view({batch_size, seq_len, hidden_size});
    } else {
        return output_flat;
    }
}

// Routed expert forward (for a single expert)
torch::Tensor routed_expert_forward(
    torch::Tensor expert_inputs,
    torch::Tensor expert_weights,
    torch::Tensor gate_weight,
    torch::Tensor up_weight,
    torch::Tensor down_weight
) {
    TORCH_CHECK(expert_inputs.is_cuda(), "expert_inputs must be a CUDA tensor");
    
    // Ensure float32 and contiguous
    expert_inputs = expert_inputs.to(torch::kFloat32).contiguous();
    expert_weights = expert_weights.to(torch::kFloat32).contiguous();
    gate_weight = gate_weight.to(torch::kFloat32).contiguous();
    up_weight = up_weight.to(torch::kFloat32).contiguous();
    down_weight = down_weight.to(torch::kFloat32).contiguous();
    
    auto num_tokens = expert_inputs.size(0);
    auto hidden_size = expert_inputs.size(1);
    auto intermediate_size = gate_weight.size(0);
    
    auto output = torch::zeros_like(expert_inputs);
    
    auto stream = at::cuda::getCurrentCUDAStream();
    
    deepseek_routed_expert_forward_cuda(
        expert_inputs.data_ptr<float>(),
        nullptr,  // expert_indices (not needed here)
        expert_weights.data_ptr<float>(),
        gate_weight.data_ptr<float>(),
        up_weight.data_ptr<float>(),
        down_weight.data_ptr<float>(),
        nullptr,  // activation
        output.data_ptr<float>(),
        num_tokens, hidden_size, intermediate_size, 0, 0, stream
    );
    
    return output;
}

// Scatter tokens to experts
std::vector<torch::Tensor> scatter_tokens(
    torch::Tensor hidden_states,
    torch::Tensor expert_indices,
    torch::Tensor expert_weights,
    int num_experts,
    int top_k
) {
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be a CUDA tensor");
    
    // Ensure float32 and contiguous for float tensors
    hidden_states = hidden_states.to(torch::kFloat32).contiguous();
    expert_weights = expert_weights.to(torch::kFloat32).contiguous();
    
    // Ensure expert_indices is int32, CUDA, and contiguous
    expert_indices = expert_indices.to(torch::kInt32).to(hidden_states.device()).contiguous();
    
    // Handle both 2D and 3D inputs
    int64_t batch_size, seq_len, batch_seq_size, hidden_size;
    if (hidden_states.dim() == 2) {
        // 2D input: (batch_seq_size, hidden_size)
        batch_seq_size = hidden_states.size(0);
        hidden_size = hidden_states.size(1);
        batch_size = 1;
        seq_len = batch_seq_size;
    } else if (hidden_states.dim() == 3) {
        // 3D input: (batch_size, seq_len, hidden_size)
        batch_size = hidden_states.size(0);
        seq_len = hidden_states.size(1);
        hidden_size = hidden_states.size(2);
        batch_seq_size = batch_size * seq_len;
    } else {
        TORCH_CHECK(false, "hidden_states must be 2D or 3D");
    }
    
    // Allocate buffers for expert inputs
    // Estimate max tokens per expert (could be optimized)
    int max_tokens_per_expert = batch_seq_size * top_k / num_experts + 100;
    int total_expert_tokens = max_tokens_per_expert * num_experts;
    
    auto expert_inputs = torch::zeros(
        {total_expert_tokens, hidden_size},
        torch::TensorOptions().dtype(hidden_states.dtype()).device(hidden_states.device())
    );
    auto expert_input_weights = torch::zeros(
        {total_expert_tokens},
        torch::TensorOptions().dtype(hidden_states.dtype()).device(hidden_states.device())
    );
    auto expert_token_counts = torch::zeros(
        {num_experts},
        torch::TensorOptions().dtype(torch::kInt32).device(hidden_states.device())
    );
    // Create offsets on CPU first, then copy to GPU
    auto expert_token_offsets_cpu = torch::zeros(
        {num_experts},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
    );
    auto expert_token_offsets_accessor = expert_token_offsets_cpu.accessor<int32_t, 1>();
    for (int i = 0; i < num_experts; i++) {
        expert_token_offsets_accessor[i] = i * max_tokens_per_expert;
    }
    auto expert_token_offsets = expert_token_offsets_cpu.to(hidden_states.device());
    
    auto token_to_expert_pos = torch::full(
        {batch_seq_size, num_experts}, -1,
        torch::TensorOptions().dtype(torch::kInt32).device(hidden_states.device())
    );
    
    // Flatten expert_indices and expert_weights to 2D
    auto expert_indices_flat = expert_indices.view({-1, top_k});
    auto expert_weights_flat = expert_weights.view({-1, top_k});
    
    auto stream = at::cuda::getCurrentCUDAStream();
    
    deepseek_scatter_tokens_cuda(
        hidden_states.view({-1, hidden_size}).contiguous().data_ptr<float>(),
        expert_indices_flat.contiguous().data_ptr<int>(),
        expert_weights_flat.contiguous().data_ptr<float>(),
        expert_inputs.data_ptr<float>(),
        expert_input_weights.data_ptr<float>(),
        expert_token_counts.data_ptr<int>(),
        expert_token_offsets.data_ptr<int>(),
        token_to_expert_pos.data_ptr<int>(),
        batch_seq_size, 1, hidden_size, num_experts, top_k, stream  // seq_len=1 for flattened
    );
    
    return {expert_inputs, expert_input_weights, expert_token_counts, 
            expert_token_offsets, token_to_expert_pos};
}

// Gather expert outputs
torch::Tensor gather_expert_outputs(
    torch::Tensor expert_outputs,
    torch::Tensor expert_indices,
    torch::Tensor expert_weights,
    torch::Tensor expert_token_offsets,
    torch::Tensor token_to_expert_pos,
    int batch_size,
    int seq_len,
    int hidden_size,
    int num_experts,
    int top_k
) {
    TORCH_CHECK(expert_outputs.is_cuda(), "expert_outputs must be a CUDA tensor");
    
    // Ensure float32 and contiguous
    expert_outputs = expert_outputs.to(torch::kFloat32).contiguous();
    expert_weights = expert_weights.to(torch::kFloat32).contiguous();
    
    // Ensure expert_indices is int32, CUDA, and contiguous
    expert_indices = expert_indices.to(torch::kInt32).to(expert_outputs.device()).contiguous();
    
    auto output = torch::zeros(
        {batch_size, seq_len, hidden_size},
        torch::TensorOptions().dtype(torch::kFloat32).device(expert_outputs.device())
    );
    
    auto stream = at::cuda::getCurrentCUDAStream();
    
    deepseek_gather_expert_outputs_cuda(
        expert_outputs.data_ptr<float>(),
        expert_indices.view({-1, top_k}).contiguous().data_ptr<int>(),
        expert_weights.view({-1, top_k}).contiguous().data_ptr<float>(),
        expert_token_offsets.data_ptr<int>(),
        token_to_expert_pos.data_ptr<int>(),
        output.data_ptr<float>(),
        batch_size, seq_len, hidden_size, num_experts, top_k, stream
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("router_forward", &router_forward, "DeepseekV3 Router forward (CUDA)");
    m.def("topk_forward", &topk_forward, "DeepseekV3 Top-K selection (CUDA)");
    m.def("shared_expert_forward", &shared_expert_forward, "DeepseekV3 Shared Expert forward (CUDA)");
    m.def("routed_expert_forward", &routed_expert_forward, "DeepseekV3 Routed Expert forward (CUDA)");
    m.def("scatter_tokens", &scatter_tokens, "Scatter tokens to experts (CUDA)");
    m.def("gather_expert_outputs", &gather_expert_outputs, "Gather expert outputs (CUDA)");
}

