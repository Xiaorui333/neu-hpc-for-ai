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
# - 选用 nvidia/cuda:12.1.1-devel-ubuntu22.04 与 torch cu121 对齐（避免弃用警告）
# - 预装构建依赖，NVSHMEM 在运行期安装（更灵活）
# - 包含 OpenMPI + PMIx + UCX 以支持 nvshmrun 多 PE 启动
flashmoe_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
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
        # Multi-PE launcher support (nvshmrun)
        "openmpi-bin",
        "libopenmpi-dev",
        "libpmix-dev",
        "libevent-dev",
        "libucx0",  # UCX library (Ubuntu 22.04 package name)
        "libucx-dev",  # UCX development headers
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
            "NVSHMEM_SYMMETRIC_SIZE": "768M",  # Increased for multi-PE
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

    print("Installing NVSHMEM 2.9.0-2 from source...")
    print("(This takes several minutes)")
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

    # Build with MPI/SHMEM support to enable nvshmrun launcher
    # Auto-detect MPI prefix and pass to NVSHMEM Makefile
    build_script = dedent(
        """
        #!/bin/bash
        set -e
        cd /tmp/nvshmem_src_2.9.0-2
        
        # Auto-detect MPI prefix from mpicc location
        MPICC=$(which mpicc || echo "")
        if [ -z "$MPICC" ]; then
            echo "Error: mpicc not found. Install openmpi-bin and libopenmpi-dev."
            exit 1
        fi
        
        # Extract MPI prefix (root directory containing bin/mpicc)
        MPI_PREFIX=""
        
        # Method 1: From mpicc path (most reliable for system packages)
        # /usr/bin/mpicc -> /usr
        # /usr/local/bin/mpicc -> /usr/local
        MPICC_DIR=$(dirname "$MPICC")
        if [ "$MPICC_DIR" = "/usr/bin" ]; then
            MPI_PREFIX="/usr"
        elif [ "$MPICC_DIR" = "/usr/local/bin" ]; then
            MPI_PREFIX="/usr/local"
        else
            # Extract parent of bin directory
            MPI_PREFIX=$(dirname "$MPICC_DIR")
        fi
        
        # Method 2: Verify by checking if bin/mpicc exists at prefix
        if [ -n "$MPI_PREFIX" ] && [ ! -f "$MPI_PREFIX/bin/mpicc" ]; then
            # Try to find the actual root
            # For Ubuntu: mpicc might be in /usr/bin, but MPI libs in /usr/lib/x86_64-linux-gnu/openmpi
            # The prefix should still be /usr
            if [ -f "/usr/bin/mpicc" ]; then
                MPI_PREFIX="/usr"
            elif [ -f "/usr/local/bin/mpicc" ]; then
                MPI_PREFIX="/usr/local"
            fi
        fi
        
        # Method 3: Use mpicc --showme to verify (but extract root, not subdir)
        if [ -z "$MPI_PREFIX" ] || [ ! -d "$MPI_PREFIX" ]; then
            MPI_INCLUDE=$(mpicc --showme:compile 2>/dev/null | tr ' ' '\\n' | grep '^-I' | head -1 | sed 's/^-I//' || echo "")
            if [ -n "$MPI_INCLUDE" ]; then
                # Extract root: /usr/include/openmpi -> /usr
                # /usr/lib/x86_64-linux-gnu/openmpi/include -> /usr
                if [[ "$MPI_INCLUDE" == "/usr"* ]]; then
                    MPI_PREFIX="/usr"
                elif [[ "$MPI_INCLUDE" == "/usr/local"* ]]; then
                    MPI_PREFIX="/usr/local"
                fi
            fi
        fi
        
        # Final check: ensure prefix is valid
        if [ -z "$MPI_PREFIX" ] || [ ! -d "$MPI_PREFIX" ]; then
            echo "Error: Could not detect MPI prefix from mpicc: $MPICC"
            exit 1
        fi
        
        echo "Detected MPI prefix: $MPI_PREFIX"
        echo "mpicc location: $MPICC"
        
        # NVSHMEM Makefile expects /usr/local/ompi structure
        # Create complete symlink structure to match Makefile expectations
        mkdir -p /usr/local/ompi/{bin,include,lib}
        
        # Link MPI compilers
        if [ ! -e /usr/local/ompi/bin/mpicc ]; then
            ln -sf "$MPICC" /usr/local/ompi/bin/mpicc
            echo "Created symlink: /usr/local/ompi/bin/mpicc -> $MPICC"
        fi
        
        MPICXX=$(which mpicxx || which mpic++ || echo "")
        if [ -n "$MPICXX" ] && [ ! -e /usr/local/ompi/bin/mpicxx ]; then
            ln -sf "$MPICXX" /usr/local/ompi/bin/mpicxx
            echo "Created symlink: /usr/local/ompi/bin/mpicxx -> $MPICXX"
        fi
        
        # Check for oshcc (OpenSHMEM compiler) - required for NVSHMEM_SHMEM_SUPPORT
        OSHCC=$(which oshcc || echo "")
        ENABLE_SHMEM=1
        if [ -n "$OSHCC" ] && [ ! -e /usr/local/ompi/bin/oshcc ]; then
            ln -sf "$OSHCC" /usr/local/ompi/bin/oshcc
            echo "Created symlink: /usr/local/ompi/bin/oshcc -> $OSHCC"
        elif [ -z "$OSHCC" ]; then
            # oshcc not found - disable SHMEM support (we mainly need MPI for nvshmrun)
            echo "Warning: oshcc not found, disabling NVSHMEM_SHMEM_SUPPORT"
            echo "  (MPI support is sufficient for nvshmrun multi-GPU execution)"
            ENABLE_SHMEM=0
        fi
        
        # Link MPI headers (if they exist)
        for mpi_inc in /usr/include/openmpi /usr/lib/x86_64-linux-gnu/openmpi/include; do
            if [ -d "$mpi_inc" ]; then
                # Create symlinks for each header file
                find "$mpi_inc" -maxdepth 1 -type f -name "*.h" -exec ln -sf {} /usr/local/ompi/include/ \\; 2>/dev/null || true
                # Also link subdirectories if any
                find "$mpi_inc" -maxdepth 1 -type d ! -path "$mpi_inc" -exec ln -sf {} /usr/local/ompi/include/ \\; 2>/dev/null || true
                break
            fi
        done
        
        # Link MPI libraries (if they exist)
        for mpi_lib in /usr/lib/x86_64-linux-gnu/openmpi /usr/lib/openmpi; do
            if [ -d "$mpi_lib" ]; then
                find "$mpi_lib" -maxdepth 1 -type f -name "*.so*" -exec ln -sf {} /usr/local/ompi/lib/ \\; 2>/dev/null || true
                break
            fi
        done
        
        echo "MPI symlink structure created at /usr/local/ompi"
        
        # Build NVSHMEM
        # Strategy: Use /usr/local/ompi as MPI_HOME (where we created symlinks)
        # This ensures Makefile finds mpicc at the expected location
        export PATH="/usr/local/ompi/bin:$MPI_PREFIX/bin:$PATH"
        export MPICC="/usr/local/ompi/bin/mpicc"
        if [ -n "$MPICXX" ]; then
            export MPICXX="/usr/local/ompi/bin/mpicxx"
        fi
        
        # Build NVSHMEM - conditionally enable SHMEM support
        BUILD_CMD="make -j$(nproc) install PREFIX=/usr/local/nvshmem \
            NVSHMEM_BUILD_IBGDA=1 \
            NVSHMEM_ENABLE_ALL_DEVICE_INLINING=1 \
            NVSHMEM_USE_GDRCOPY=0 \
            NVSHMEM_MPI_SUPPORT=1 \
            NVSHMEM_BUILD_EXAMPLES=0 \
            NVSHMEM_BUILD_TESTS=0 \
            MPI_HOME=\"/usr/local/ompi\" \
            OMPI_HOME=\"/usr/local/ompi\" \
            MPICC=\"/usr/local/ompi/bin/mpicc\""
        
        if [ "$ENABLE_SHMEM" = "1" ]; then
            BUILD_CMD="$BUILD_CMD NVSHMEM_SHMEM_SUPPORT=1"
            echo "Building with SHMEM support enabled"
        else
            BUILD_CMD="$BUILD_CMD NVSHMEM_SHMEM_SUPPORT=0"
            echo "Building with SHMEM support disabled (MPI only)"
        fi
        
        # Execute build command
        eval $BUILD_CMD
        """
    ).strip()
    
    with open("/tmp/nvshmem_build.sh", "w") as f:
        f.write(build_script)
    
    r = _run("bash /tmp/nvshmem_build.sh", timeout=1800)  # up to 30 minutes just in case
    if r.returncode == 0:
        # Verify installation
        check_result = _run("bash -lc 'ls -la /usr/local/nvshmem/lib* 2>/dev/null | head -20'", timeout=10)
        if check_result.stdout:
            print("  Installed files:")
            for line in check_result.stdout.strip().split('\n')[:10]:
                print(f"    {line}")
        
        # Post-install: 复制 nvshmrun 启动器和头文件
        print("  Post-install fixes...")
        post_fix = _run(
            "bash -lc '"
            "mkdir -p /usr/local/nvshmem/bin && "
            # 从多个位置查找 nvshmrun: 源码树、构建目录、安装目录
            "NVSHMRUN=\"\" && "
            "for loc in /tmp/nvshmem_src_2.9.0-2/src/launch/nvshmrun "
            "/tmp/nvshmem_src_2.9.0-2/build/bin/nvshmrun "
            "/tmp/nvshmem_src_2.9.0-2/usr/bin/nvshmrun "
            "/usr/local/nvshmem/bin/nvshmrun "
            "$(find /tmp/nvshmem_src_2.9.0-2 -name nvshmrun -type f 2>/dev/null | head -1); "
            "do "
            "  if [ -f \"$loc\" ] && [ -x \"$loc\" ]; then "
            "    NVSHMRUN=\"$loc\"; break; "
            "  fi; "
            "done && "
            # 如果找到，复制到安装目录
            "if [ -n \"$NVSHMRUN\" ]; then "
            "  cp \"$NVSHMRUN\" /usr/local/nvshmem/bin/ && "
            "  chmod +x /usr/local/nvshmem/bin/nvshmrun && "
            "  echo \"    ✓ nvshmrun installed from $NVSHMRUN\"; "
            "else "
            "  echo \"    (nvshmrun not found - creating wrapper script)\"; "
            "fi"
            "'",
            timeout=30
        )
        
        # If nvshmrun not found, create a wrapper script using Python
        nvshmrun_wrapper_check = _run(
            "bash -lc 'test -x /usr/local/nvshmem/bin/nvshmrun && echo FOUND || echo MISS'",
            timeout=5
        )
        if "MISS" in (nvshmrun_wrapper_check.stdout or ""):
            print("  Creating nvshmrun wrapper script...")
            wrapper_script = """#!/bin/bash
# NVSHMEM launcher wrapper (using mpirun)
# This is a fallback wrapper when nvshmrun is not available in the NVSHMEM build

export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-512M}
export LD_LIBRARY_PATH=/usr/local/nvshmem/lib:/usr/local/nvshmem/lib64:$LD_LIBRARY_PATH

# Parse nvshmrun-style arguments: -np N --ppn M command args...
NP=""
PPN=""
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -np)
            NP="$2"
            shift 2
            ;;
        --ppn)
            PPN="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Use mpirun with parsed arguments
if [ -n "$NP" ]; then
    exec mpirun -np "$NP" "${ARGS[@]}"
else
    exec mpirun "${ARGS[@]}"
fi
"""
            with open("/tmp/nvshmrun_wrapper.sh", "w") as f:
                f.write(wrapper_script)
            _run(
                "bash -lc 'cp /tmp/nvshmrun_wrapper.sh /usr/local/nvshmem/bin/nvshmrun && chmod +x /usr/local/nvshmem/bin/nvshmrun && echo \"    ✓ nvshmrun wrapper created\"'",
                timeout=10
            )
        
        # Continue with other post-install fixes
        post_fix2 = _run(
            "bash -lc '"
            # 创建 libnvshmem.so 兼容软链（如果只有 _host 版本）
            "cd /usr/local/nvshmem/lib && "
            "if [ ! -f libnvshmem.so ] && [ -f libnvshmem_host.so ]; then ln -sf libnvshmem_host.so libnvshmem.so && echo \"    ✓ libnvshmem.so -> libnvshmem_host.so\"; fi && "
            # 创建 lib64 -> lib 软链
            "cd /usr/local/nvshmem && if [ ! -e lib64 ]; then ln -sf lib lib64 && echo \"    ✓ lib64 -> lib\"; fi"
            "'",
            timeout=10
        )
        if post_fix2.stdout:
            print(post_fix2.stdout)
        
        if post_fix.stdout:
            print(post_fix.stdout)
        
        # Verify nvshmrun installation
        nvshmrun_check = _run("bash -lc 'test -x /usr/local/nvshmem/bin/nvshmrun && echo nvshmrun:OK || echo nvshmrun:MISS'", timeout=10)
        if nvshmrun_check.stdout:
            print(f"  {nvshmrun_check.stdout.strip()}")
        
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
def build_extension(enable_nvshmem=False):
    """Build the PyTorch CUDA extension in-place."""
    # rpath 避免运行期再 export LD_LIBRARY_PATH
    # Support both lib and lib64
    os.environ["LD_LIBRARY_PATH"] = (
        f"/usr/local/nvshmem/lib:/usr/local/nvshmem/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )
    os.environ["CPATH"] = f"/usr/local/nvshmem/include:{os.environ.get('CPATH', '')}"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;9.0")
    
    # Force use g++ (PyTorch looks for clang++ by default on some systems)
    os.environ["CXX"] = "g++"
    os.environ["CC"] = "gcc"
    
    # Set NVSHMEM flag
    if enable_nvshmem:
        os.environ["USE_NVSHMEM"] = "1"
        print("🔥 Building with NVSHMEM ENABLED (full RDMA support)")
    else:
        os.environ["USE_NVSHMEM"] = "0"
        print("Building with core structures only (NVSHMEM disabled)")

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
def run_single_gpu():
    """
    Single-GPU smoke test: Verify compilation and basic functionality (without NVSHMEM).
    """
    os.chdir("/root/flashmoe")

    print("=" * 80)
    print("FlashMoE RDMA Fused Kernel - Single-GPU Smoke Test")
    print("=" * 80)

    print("\n[1/2] Build FlashMoE Extension (Core Structures Only)")
    print("-" * 80)
    print("NOTE: Building WITHOUT NVSHMEM for single-GPU stability")
    print("      Use multi-GPU mode for full RDMA support")
    # Build without NVSHMEM for single-GPU stability
    if not build_extension(enable_nvshmem=False):
        print("Build failed; aborting.")
        return
    print()

    print("[2/2] Run Single-GPU Smoke Test")
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
        print("NVSHMEM support:", getattr(mod, "has_nvshmem", lambda: "False (disabled for single-GPU)")())

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
    Quick path: Build extension (no NVSHMEM for fast test).
    """
    os.chdir("/root/flashmoe")

    print("FlashMoE Quick Build Test (Core Structures Only)")
    print("=" * 80)

    if build_extension(enable_nvshmem=False):
        print("✓ Module built OK, try importing ...")
        r = _run(
            "bash -lc \"python -c 'import sys; sys.path.insert(0,\\\"/root/flashmoe\\\"); "
            "import flashmoe_rdma_cuda as m; "
            "print(m.get_version() if hasattr(m, \\\"get_version\\\") else \\\"no_version\\\")'\"",
            timeout=30
        )
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
@app.function(image=flashmoe_image, gpu="A100-40GB:2", timeout=3600)
def run():
    """
    Multi-GPU test: Full Paper-aligned RDMA with 2 GPUs.
    """
    os.chdir("/root/flashmoe")
    os.environ["NVSHMEM_SYMMETRIC_SIZE"] = "768M"  # Increased for multi-PE

    print("=" * 80)
    print("FlashMoE RDMA - MULTI-GPU (Full Paper Alignment)")
    print("=" * 80)

    print("\n[1/5] NVSHMEM Setup")
    print("-" * 80)
    if not install_nvshmem():
        print("NVSHMEM is not available; aborting.")
        return
    print()

    print("[2/5] Verify NVSHMEM Installation")
    print("-" * 80)
    # Run detection test
    detect_result = _run("bash -lc 'python test_nvshmem_detection.py'", timeout=30)
    print(detect_result.stdout if detect_result.stdout else "(no output)")
    print()
    
    print("[3/5] Build FlashMoE Extension with NVSHMEM")
    print("-" * 80)
    if not build_extension(enable_nvshmem=True):
        print("Build failed; aborting.")
        return
    print()

    print("[4/5] Check GPU Configuration")
    print("-" * 80)
    gpu_check = _run("nvidia-smi -L", timeout=30)
    print(gpu_check.stdout)
    print()

    print("[5/5] Run Multi-GPU RDMA Kernel")
    print("-" * 80)
    
    # Create multi-GPU test script
    test_code = dedent(
        r"""
        import torch
        import os
        import sys
        
        sys.path.insert(0, '/root/flashmoe')
        
        print("=" * 80)
        print("Multi-GPU NVSHMEM Test")
        print("=" * 80)
        
        # Get MPI rank (will correspond to PE ID after nvshmemx_init_attr)
        # PE information is set during kernel initialization via nvshmemx_init_attr
        # Use OpenMPI environment variables (mpi4py not installed in image)
        my_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
        n_ranks = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
        
        # Check available GPUs
        n_gpus = torch.cuda.device_count()
        print(f"\nMPI Rank: {my_rank}/{n_ranks} (will become PE {my_rank}/{n_ranks} after NVSHMEM init)")
        print(f"Available GPUs: {n_gpus}")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Use MPI rank as device ID (one PE per GPU)
        # Kernel will set device via cudaSetDevice(mype_node) during init
        device_id = my_rank % n_gpus
        print(f"\n✓ Rank {my_rank} will use GPU {device_id} for FlashMoE RDMA")
        
        print()
        
        # Import module
        import flashmoe_rdma_cuda as mod
        print(f"Module version: {mod.get_version()}")
        print(f"NVSHMEM support: {mod.has_nvshmem()}")
        
        if not mod.has_nvshmem():
            print("\n⚠ Module compiled without NVSHMEM support")
            print("  Set USE_NVSHMEM=1 to enable")
            sys.exit(0)
        
        print()
        
        # Create test tensors
        batch_size, seq_len, hidden_dim = 8, 256, 1024
        num_experts, intermediate_dim = 16, 4096
        num_devices = n_ranks  # Use actual number of MPI ranks (PEs)
        top_k = 2
        
        device = torch.device(f"cuda:{device_id}")
        
        print(f"\nTest configuration:")
        print(f"  Batch: {batch_size}, Seq: {seq_len}, Hidden: {hidden_dim}")
        print(f"  Experts: {num_experts}, Top-K: {top_k}")
        print(f"  PEs: {n_ranks}, Devices: {num_devices}, Rank {my_rank} on GPU {device_id}")
        print()
        
        # Input tensors
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        router_weight = torch.randn(hidden_dim, num_experts, device=device)
        expert_gate = torch.randn(num_experts, intermediate_dim, hidden_dim, device=device)
        expert_up = torch.randn(num_experts, intermediate_dim, hidden_dim, device=device)
        expert_down = torch.randn(num_experts, hidden_dim, intermediate_dim, device=device)
        
        print("Running FlashMoE RDMA kernel...")
        try:
            output = mod.flashmoe_rdma_forward(
                hidden_states, router_weight,
                expert_gate, expert_up, expert_down,
                top_k, num_experts, num_devices
            )
            print(f"✓ Rank {my_rank} (PE {my_rank}): Multi-GPU RDMA kernel executed successfully!")
            print(f"  Output shape: {output.shape}")
            print(f"  Output dtype: {output.dtype}")
            print(f"  Output device: {output.device}")
            print()
            if my_rank == 0:  # Only print once from Rank 0 (PE 0)
                print("=" * 80)
                print("✅ Full Paper-Aligned FlashMoE with NVSHMEM RDMA (Multi-PE)")
                print("=" * 80)
        except Exception as e:
            print(f"✗ Kernel execution failed: {e}")
            import traceback
            traceback.print_exc()
        """
    ).strip()

    with open("/tmp/test_multi_gpu.py", "w") as f:
        f.write(test_code)

    # Check GPU count (use PyTorch which is more reliable)
    gpu_count_check = _run("python -c 'import torch; print(torch.cuda.device_count())'", timeout=10)
    n_gpus = int(gpu_count_check.stdout.strip()) if gpu_count_check.returncode == 0 else 1
    
    print(f"Detected {n_gpus} GPU(s) via PyTorch")
    print()
    
    # NVSHMEM environment variables (best practices)
    nvshmem_home = "/usr/local/nvshmem"
    nvshmem_symmetric_size = "768M"
    
    # Set OpenMPI allow root execution (required for Modal root user)
    os.environ["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    os.environ["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    
    # Use mpirun directly (nvshmemx_init_attr with MPI_COMM_WORLD handles PE initialization)
    # OpenMPI automatically sets OMPI_COMM_WORLD_LOCAL_RANK for strict PE-GPU binding
    if n_gpus >= 2:
        np = min(n_gpus, 2)
        print(f"Using mpirun for multi-PE execution ({np} PEs, one per GPU)...")
        print("  (nvshmemx_init_attr with MPI_COMM_WORLD will create PE: 0/2, 1/2)")
        print("  (OMPI_COMM_WORLD_LOCAL_RANK will be auto-set by mpirun for strict PE-GPU binding)")
        
        # Build CUDA_VISIBLE_DEVICES list (0,1 for 2 GPUs)
        cuda_visible = ",".join(str(i) for i in range(np))
        
        # Build LD_LIBRARY_PATH in parent process (expand $LD_LIBRARY_PATH first)
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        ld_library_path = f"{nvshmem_home}/lib:{nvshmem_home}/lib64"
        if current_ld_path:
            ld_library_path = f"{ld_library_path}:{current_ld_path}"
        
        # Use mpirun with proper environment variable passing
        # --bind-to none: allow processes to use different CPUs/GPUs
        # -x: pass environment variables to MPI processes (does NOT expand $VAR)
        # --allow-run-as-root: explicit flag (double insurance)
        cmd = (
            f"mpirun -np {np} --bind-to none --allow-run-as-root "
            f"-x OMPI_ALLOW_RUN_AS_ROOT=1 "
            f"-x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 "
            f"-x NVSHMEM_HOME={nvshmem_home} "
            f"-x LD_LIBRARY_PATH={ld_library_path} "
            f"-x NVSHMEM_BOOTSTRAP=MPI "
            f"-x NVSHMEM_SYMMETRIC_SIZE={nvshmem_symmetric_size} "
            f"-x NVSHMEM_UCX_TLS=rc,sm,cuda_copy,cuda_ipc "
            f"-x CUDA_VISIBLE_DEVICES={cuda_visible} "
            f"python /tmp/test_multi_gpu.py"
        )
    else:
        print("Single GPU detected, running without mpirun...")
        # Single GPU: set environment variables directly
        env_setup = (
            f"export NVSHMEM_HOME={nvshmem_home} && "
            f"export LD_LIBRARY_PATH={nvshmem_home}/lib:{nvshmem_home}/lib64:$LD_LIBRARY_PATH && "
            f"export NVSHMEM_BOOTSTRAP=MPI && "
            f"export NVSHMEM_SYMMETRIC_SIZE={nvshmem_symmetric_size} && "
            f"export NVSHMEM_UCX_TLS=rc,sm,cuda_copy,cuda_ipc"
        )
        cmd = f"{env_setup} && python /tmp/test_multi_gpu.py"
    
    r = _run(f"bash -lc '{cmd}'", timeout=1200)
    print(r.stdout)
    if r.returncode != 0:
        print("\nTest stderr:")
        _print_tail("stderr", r.stderr, 120)
    print("=" * 80)
    print("Multi-GPU test completed.")
    print("=" * 80)


@app.local_entrypoint()
def main(mode: str = "multi"):
    """
    Run FlashMoE RDMA kernel.
    
    Args:
        mode: "single" for single-GPU test, "multi" for multi-GPU (default)
    """
    if mode == "single":
        print("Running single-GPU smoke test...")
        run_single_gpu.remote()
    else:
        print("Running multi-GPU for full Paper alignment...")
        run.remote()
