import modal
import os
import sys

# HPC-X configuration for CUDA-aware MPI
HPCX_VER = "v2.18"  
HPCX_TBZ = f"hpcx-{HPCX_VER}-gcc-mlnx_ofed-ubuntu22.04-cuda12-x86_64.tbz"
HPCX_DIR = f"/opt/hpcx-{HPCX_VER}-gcc-mlnx_ofed-ubuntu22.04-cuda12-x86_64"
URLS = [
    f"https://content.mellanox.com/hpc/hpc-x/{HPCX_VER}/{HPCX_TBZ}",
    "https://content.mellanox.com/hpc/hpc-x/v2.17/hpcx-v2.17-gcc-mlnx_ofed-ubuntu22.04-cuda12-x86_64.tbz",
    "https://content.mellanox.com/hpc/hpc-x/v2.16/hpcx-v2.16-gcc-mlnx_ofed-ubuntu22.04-cuda12-x86_64.tbz",
]

image = (
    modal.Image.from_registry(f"nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
        .apt_install("wget", "curl", "ca-certificates", "bzip2", "libnuma1", "libnuma-dev")
        .run_commands(
        # Download and extract HPC-X for CUDA-aware MPI
            "set -eux"
        " && cd /opt"
        " && success=0; for u in " + " ".join(f"'{u}'" for u in URLS) + "; do "
        f"      (wget -t 2 -T 20 \"$u\" -O '{HPCX_TBZ}' || "
        f"       curl -fL --retry 2 --retry-connrefused -o '{HPCX_TBZ}' \"$u\") && success=1 && break || true; "
        "   done"
        " && [ $success -eq 1 ]"
        f" && tar -xjf '{HPCX_TBZ}'"
        # Create a fixed /opt/hpcx symlink for easier finding later
        f" && ln -s '{HPCX_DIR}' /opt/hpcx || true"
        # Self-check mpicxx exists during build
        f" && test -x {HPCX_DIR}/ompi/bin/mpicxx || (echo 'HPC-X mpicxx not found'; ls -al /opt; ls -al {HPCX_DIR} || true; exit 1)"
    )
        .env({
            "NCCL_DEBUG": "INFO",
        })
        .add_local_dir("util", remote_path="/root/util")
        .add_local_dir("week_01", remote_path="/root/week_01")
        .add_local_dir("week_02", remote_path="/root/week_02")
        .add_local_dir("week_03", remote_path="/root/week_03")
        .add_local_dir("week_04", remote_path="/root/week_04")
        .add_local_dir("week_05", remote_path="/root/week_05")
        .add_local_dir("week_07", remote_path="/root/week_07")
)
app = modal.App("nvcc")

@app.function(image=image, gpu="A100-40gb:8", timeout=300)
def compile_and_run_cuda(code_path: str, np: int = 4):
    import subprocess
    import os

    # First, let's check what files are available
    print("Current directory contents:")
    subprocess.run(["ls", "-la"], text=True)
    
    # Check if this is an MPI-enabled CUDA file
    is_mpi_file = "mpi" in code_path.lower()
    
    if is_mpi_file:
        HPCX_DIR = "/opt/hpcx-v2.18-gcc-mlnx_ofed-ubuntu22.04-cuda12-x86_64"

        # 1) 先构造基础环境（关键：OPAL_PREFIX 指向 ompi）
        base_env = os.environ.copy()
        base_env.update({
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
            # CUDA-aware / 单机 UCX
            "OMPI_MCA_opal_cuda_support": "true",
            "OMPI_MCA_pml": "ucx",
            "OMPI_MCA_plm": "isolated",
            "UCX_TLS": "cuda_copy,cuda_ipc,sm,self",
            "UCX_MEMTYPE_CACHE": "n",
            #"UCX_SOCKADDR_CM_ENABLE": "n", 
            #"UCX_NET_DEVICES": "lo", 
            "UCX_WARN_UNUSED_ENV_VARS": "n",
            "UCX_LOG_LEVEL": "fatal",
            "OMPI_MCA_btl": "self,vader",
            # 单机模式配置 - 避免SSH/RSH依赖
            "OMPI_MCA_rmaps_base_mapping_policy": "slot",
            "OMPI_MCA_rmaps_base_ranking_policy": "slot:fill",
            "OMPI_MCA_btl_vader_single_copy_mechanism": "none",
        })

        # 2) 用 bash -lc 真正加载 HPC-X（source + hpcx_load），然后把环境带回 Python
        def hpcx_loaded_env(extra_env):
            cmd = f"source {HPCX_DIR}/hpcx-init.sh; hpcx_load; env"
            out = subprocess.run(["bash", "-lc", cmd], text=True, capture_output=True, check=True, env=extra_env).stdout
            env = dict(extra_env)
            for line in out.splitlines():
                if "=" in line and not line.startswith("_"):
                    k, v = line.split("=", 1)
                    env[k] = v
            # 再追加我们关心的库路径（有些镜像里 hpcx_load 不一定把 ucx/hcoll lib 都并上）
            env["LD_LIBRARY_PATH"] = (
                f"{HPCX_DIR}/ompi/lib:{HPCX_DIR}/ucx/lib:{HPCX_DIR}/hcoll/lib:"
                + env.get("LD_LIBRARY_PATH", "")
            )
            return env

        run_env = hpcx_loaded_env(base_env)

        # 3) 用 HPC-X 的 mpicxx 做 host 编译器（避免 ABI/路径错配）
        mpicxx = f"{HPCX_DIR}/ompi/bin/mpicxx"
        print("======== Compiling with NVCC + HPC-X mpicxx ========")
        subprocess.run(
            [
                "nvcc", "-O2", "-std=c++17", "-arch=sm_80",
                f"-ccbin={mpicxx}",
                code_path,
                "-o", "output.bin",
            ],
            check=True, text=True, env=run_env
        )

        # 4) 用 HPC-X 的 mpirun，且在"已加载 HPC-X 的 shell"里执行
        mpirun = f"{HPCX_DIR}/ompi/bin/mpirun"

        desired_np = max(1, min(int(np), 8))
        try:
            gpus = subprocess.run(["bash", "-lc", "nvidia-smi --list-gpus | wc -l"],
                                text=True, capture_output=True, check=True)
            ngpus = int(gpus.stdout.strip() or "1")
            desired_np = min(desired_np, max(1, ngpus))
        except Exception:
            pass

        # 若未显式限定可见卡，则默认按前 N 张卡限制
        if "CUDA_VISIBLE_DEVICES" not in run_env:
            run_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(desired_np))


        print("======== Running with CUDA-aware HPC-X mpirun ========")
        print(f"Using np = {desired_np}")

        # 单节点映射 & 不绑核；isolated 启动器避免 ssh/rsh；UCX 配置在环境中已给定
        cmd = (
            f"source {HPCX_DIR}/hpcx-init.sh; "
            f"hpcx_load; "
            f"{mpirun} -np {desired_np} --map-by slot --bind-to none ./output.bin"
        )
        subprocess.run(["bash", "-lc", cmd], text=True, check=True, env=run_env)
        
    else:
        # Regular CUDA compilation (non-MPI)
        print("======== Compiling with NVCC (regular CUDA) ========")
        subprocess.run(["nvcc", "-DCUDA=1", "-g", "-G", "-rdc=true", "-arch=sm_80", code_path, "-o", "output.bin"],
                       text=True, check=True)
        print("======== Running CUDA program ========")
        subprocess.run(["./output.bin"], text=True, check=True)