# modal_flashmoe.py
# -----------------------------------------------------------------------------
# FlashMoE RDMA Demo on Modal
# - CUDA 12.1 to match torch==2.4.1+cu121
# - NVSHMEM 2.9.0-2 install (binary preferred; fallback to source)
# - Build PyTorch CUDA extension (flashmoe_rdma_cuda)
# - Single-GPU smoke test for fused RDMA kernel entry
# - Extra: environment check & quick build
# -----------------------------------------------------------------------------

import modal
import subprocess
import os
import shlex
from textwrap import dedent

app = modal.App("flashmoe-rdma")

# =============================================================================
# Modal Image
# =============================================================================
# 说明：
# - 选用 nvidia/cuda:12.1.0-devel-ubuntu22.04 与 torch cu121 对齐
# - 预装构建依赖，NVSHMEM 在运行期安装（更灵活）
flashmoe_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "build-essential",
        "git",
        "wget",
        "curl",
        # for source fallback
        "autoconf",
        "automake",
        "libtool",
        "pkg-config",
        "rdma-core",
        "libibverbs-dev",
        "cmake",
        "ninja-build",
        "python3-dev",
        "patchelf",
    )
    .pip_install(
        "torch==2.4.1+cu121",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install("numpy", "setuptools", "wheel", "ninja", "pybind11")
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "NVSHMEM_HOME": "/usr/local/nvshmem",
            "NVSHMEM_SYMMETRIC_SIZE": "512M",
            # A100/H100 默认架构（可被外部覆盖）
            "TORCH_CUDA_ARCH_LIST": "8.0;9.0",
        }
    )
    .add_local_dir(".", remote_path="/root/flashmoe")  # 你的源码目录
)


# =============================================================================
# Helpers
# =============================================================================
def _run(cmd: str, timeout=None, check=False, text=True):
    """Small wrapper for subprocess.run with common defaults."""
    res = subprocess.run(
        cmd if isinstance(cmd, list) else shlex.split(cmd),
        capture_output=True,
        text=text,
        timeout=timeout,
    )
    if check and res.returncode != 0:
        raise RuntimeError(
            f"Command failed ({res.returncode}): {cmd}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
    return res


def _print_tail(title: str, s: str, n: int = 80):
    print(f"--- {title} (last {n} lines) ---")
    lines = s.splitlines()
    for line in lines[-n:]:
        print(line)
    print("-" * 80)


# =============================================================================
# NVSHMEM Install (binary first, fallback source)
# =============================================================================
def install_nvshmem():
    """Install NVSHMEM at runtime (binary preferred; fallback to source)."""
    # Fast path
    if os.path.exists("/usr/local/nvshmem/lib/libnvshmem.so"):
        print("✓ NVSHMEM already installed")
        return True

    print("Installing NVSHMEM 2.9.0-2...")

    # ---- Method 1: Binary for Ubuntu 22.04 + CUDA 12.1
    print("\n[Method 1] Trying pre-compiled binary (CUDA 12.1)...")
    binary_url = (
        "https://developer.download.nvidia.com/compute/redist/nvshmem/2.9.0/"
        "builds/cuda12.1/txz/ubuntu22_04/x64/libnvshmem_2.9.0-2+cuda12.1_amd64.txz"
    )

    r = _run(
        f"bash -lc 'curl -L -s --fail {binary_url} -o /tmp/nvshmem_binary.txz'",
        timeout=120,
    )
    if r.returncode == 0 and os.path.exists("/tmp/nvshmem_binary.txz"):
        print("  ✓ Downloaded binary txz")
        _run("bash -lc 'mkdir -p /tmp/nvshmem_extract && tar -xf /tmp/nvshmem_binary.txz -C /tmp/nvshmem_extract'")
        # Try install into /usr/local/nvshmem
        _run("bash -lc 'mkdir -p /usr/local/nvshmem'")
        # Attempt common layouts
        attempts = [
            "test -d /tmp/nvshmem_extract/lib && cp -r /tmp/nvshmem_extract/* /usr/local/nvshmem/",
            "DIR=$(find /tmp/nvshmem_extract -type f -name libnvshmem.so -print -quit | xargs dirname | xargs dirname); "
            "cp -r $DIR/* /usr/local/nvshmem/",
        ]
        for i, cmd in enumerate(attempts, 1):
            _run(f"bash -lc '{cmd}'")
            if os.path.exists("/usr/local/nvshmem/lib/libnvshmem.so"):
                print(f"  ✓ Installed from binary (method {i})")
                print("✓ NVSHMEM installed from binary")
                return True
        print("  ✗ Binary extraction layout not recognized")

    print("  ✗ Binary installation failed; fallback to source...")

    # ---- Method 2: Build from source (reduced build set)
    print("\n[Method 2] Building from source (this takes several minutes)...")
    source_url = (
        "https://developer.download.nvidia.com/compute/redist/nvshmem/2.9.0/"
        "source/nvshmem_src_2.9.0-2.txz"
    )
    r = _run(
        f"bash -lc 'curl -L -s --fail {source_url} -o /tmp/nvshmem_src.txz'",
        timeout=180,
    )
    if r.returncode != 0:
        print("  ✗ Source download failed")
        _print_tail("curl stderr", r.stderr, 40)
        return False

    _run("bash -lc 'tar -xf /tmp/nvshmem_src.txz -C /tmp'")
    print("  ✓ Source unpacked")

    # Speed up: skip examples/tests/MPI; increase timeout
    build_cmd = dedent(
        """
        cd /tmp/nvshmem_src_2.9.0-2 && \
        make -j"$(nproc)" install PREFIX=/usr/local/nvshmem \
            NVSHMEM_BUILD_IBGDA=1 \
            NVSHMEM_ENABLE_ALL_DEVICE_INLINING=1 \
            NVSHMEM_USE_GDRCOPY=0 \
            NVSHMEM_MPI_SUPPORT=0 \
            NVSHMEM_SHMEM_SUPPORT=0 \
            NVSHMEM_BUILD_EXAMPLES=0 \
            NVSHMEM_BUILD_TESTS=0
        """
    ).strip()

    r = _run(f"bash -lc '{build_cmd}'", timeout=1800)  # up to 30 minutes just in case
    if r.returncode == 0:
        print("✓ NVSHMEM installed from source")
        return True
    else:
        print("✗ NVSHMEM source build failed")
        _print_tail("build stdout", r.stdout, 120)
        _print_tail("build stderr", r.stderr, 120)
        return False


# =============================================================================
# Build Extension
# =============================================================================
def build_extension():
    """Build the PyTorch CUDA extension in-place."""
    # rpath 避免运行期再 export LD_LIBRARY_PATH
    os.environ["LD_LIBRARY_PATH"] = f"/usr/local/nvshmem/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ["CPATH"] = f"/usr/local/nvshmem/include:{os.environ.get('CPATH', '')}"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;9.0")
    
    # Force use g++ (PyTorch looks for clang++ by default on some systems)
    os.environ["CXX"] = "g++"
    os.environ["CC"] = "gcc"

    print("Building extension (setup_flashmoe_rdma.py) ...")
    r = _run("bash -lc 'python setup_flashmoe_rdma.py build_ext --inplace'", timeout=1800)
    if r.returncode == 0:
        print("✓ Build successful")
        return True
    print("✗ Build failed")
    _print_tail("build stdout", r.stdout, 120)
    _print_tail("build stderr", r.stderr, 120)
    return False


# =============================================================================
# App Functions
# =============================================================================
@app.function(image=flashmoe_image, gpu="A100-40GB", timeout=3600)
def run():
    """
    End-to-end: NVSHMEM install -> build extension -> single-GPU smoke test.
    """
    os.chdir("/root/flashmoe")

    # Symmetric heap size for NVSHMEM
    os.environ["NVSHMEM_SYMMETRIC_SIZE"] = "512M"

    print("=" * 80)
    print("FlashMoE RDMA Fused Kernel - End-to-End")
    print("=" * 80)

    print("\n[1/3] NVSHMEM Setup")
    print("-" * 80)
    if not install_nvshmem():
        print("NVSHMEM is not available; aborting.")
        return
    print()

    print("[2/3] Build FlashMoE Extension")
    print("-" * 80)
    print("NOTE: Building with core structures, NVSHMEM disabled for stability")
    print("      To enable NVSHMEM: set USE_NVSHMEM=1 environment variable")
    if not build_extension():
        print("Build failed; aborting.")
        return
    print()

    print("[3/3] Run RDMA Kernel (single-GPU smoke test)")
    print("-" * 80)

    test_code = dedent(
        r"""
        import torch
        import importlib
        import os
        import sys
        
        # Add build directory to Python path
        sys.path.insert(0, '/root/flashmoe')

        print("CUDA visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
        print("Torch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))

        mod = importlib.import_module("flashmoe_rdma_cuda")
        print("Module version:", getattr(mod, "get_version", lambda: "N/A")())
        print("NVSHMEM support:", getattr(mod, "has_nvshmem", lambda: "N/A")())

        # Demo tensors
        batch_size, seq_len, hidden_dim = 4, 128, 512
        num_experts, intermediate_dim, num_devices = 8, 2048, 1
        top_k = 2

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Input tensors
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        router_weight = torch.randn(hidden_dim, num_experts, device=device)
        
        # Expert weights (simplified, all experts same size)
        expert_gate = torch.randn(num_experts, intermediate_dim, hidden_dim, device=device)
        expert_up = torch.randn(num_experts, intermediate_dim, hidden_dim, device=device)
        expert_down = torch.randn(num_experts, hidden_dim, intermediate_dim, device=device)

        print(f"Input shape: {hidden_states.shape}")
        print(f"Router weight shape: {router_weight.shape}")
        print(f"Num experts: {num_experts}, Top-K: {top_k}, Devices: {num_devices}")
        print()

        # Call FlashMoE RDMA forward
        if hasattr(mod, "flashmoe_rdma_forward"):
            try:
                output = mod.flashmoe_rdma_forward(
                    hidden_states, router_weight,
                    expert_gate, expert_up, expert_down,
                    top_k, num_experts, num_devices
                )
                print("✓ Kernel executed successfully!")
                print(f"  Output shape: {output.shape}")
                print(f"  Output dtype: {output.dtype}")
                print(f"  Output device: {output.device}")
            except Exception as e:
                print(f"✗ Kernel execution failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("⚠ flashmoe_rdma_cuda has no 'flashmoe_rdma_forward'; check bindings.")
        """
    ).strip()

    with open("/tmp/test_flashmoe.py", "w") as f:
        f.write(test_code)

    r = _run("bash -lc 'python /tmp/test_flashmoe.py'", timeout=1200)
    print(r.stdout)
    if r.returncode != 0:
        print("Smoke test stderr:")
        _print_tail("stderr", r.stderr, 120)
    print("=" * 80)
    print("Done.")
    print("=" * 80)


@app.function(image=flashmoe_image, gpu="A100-40GB", timeout=1800)
def test_build():
    """
    Quick path: Install NVSHMEM + build extension (no run).
    """
    os.chdir("/root/flashmoe")
    os.environ["NVSHMEM_SYMMETRIC_SIZE"] = "512M"

    print("FlashMoE Quick Build Test")
    print("=" * 80)

    if not install_nvshmem():
        print("✗ NVSHMEM install failed")
        return

    if build_extension():
        print("✓ Module built OK, try importing ...")
        r = _run("bash -lc \"python -c 'import flashmoe_rdma_cuda as m; print(m.get_version() if hasattr(m, \"get_version\") else \"no_version\")'\"")
        print(r.stdout if r.stdout else "(no output)")
    else:
        print("✗ Build failed")


@app.function(image=flashmoe_image, gpu="A100-40GB", timeout=600)
def check_env():
    """
    Print nvcc / GPU / torch info to validate runtime.
    """
    print("Environment Check")
    print("=" * 80)

    print("\n[CUDA]")
    _run("nvcc --version", check=False)
    print(os.popen("nvcc --version").read())

    print("\n[GPU]")
    print(os.popen("nvidia-smi -L").read())

    print("\n[PyTorch]")
    code = dedent(
        """
        import torch, os
        print("PyTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
        print("TORCH_CUDA_ARCH_LIST:", os.environ.get("TORCH_CUDA_ARCH_LIST"))
        """
    )
    print(os.popen(f"python - <<'PY'\n{code}\nPY").read())

    print("=" * 80)


# =============================================================================
# Local Entrypoint
# =============================================================================
@app.local_entrypoint()
def main():
    # 默认跑 e2e
    run.remote()
