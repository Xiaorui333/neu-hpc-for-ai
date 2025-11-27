from setuptools import setup
from torch.utils import cpp_extension as ce
import os
import sys

# --- NVSHMEM HOME ---
NVSHMEM_HOME = os.environ.get("NVSHMEM_HOME", "/usr/local/nvshmem")
print("Using NVSHMEM_HOME =", NVSHMEM_HOME)

# 基本存在性检查（更早失败、更好排错）
need = [
    os.path.join(NVSHMEM_HOME, "include", "nvshmem.h"),
]
missing = [p for p in need if not os.path.exists(p)]
if missing:
    sys.stderr.write("[setup] NVSHMEM headers not found: " + ", ".join(missing) + "\n")

# --- CUDA 架构处理 ---
# 优先尊重 TORCH_CUDA_ARCH_LIST（例如 "8.0;9.0"），否则给默认 A100/H100
arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0;9.0").replace(" ", "")
gencodes = []
for arch in arch_list.replace(";", ",").split(","):
    if not arch:
        continue
    # 允许 "8.0+PTX" 这类写法
    base = arch.replace("+PTX", "")
    gencodes.append(f"-gencode=arch=compute_{base.replace('.', '')},code=sm_{base.replace('.', '')}")
    if arch.endswith("+PTX"):
        gencodes.append(f"-gencode=arch=compute_{base.replace('.', '')},code=compute_{base.replace('.', '')}")

# --- include / library 目录（注意 flatten） ---
include_dirs = ce.include_paths() + [
    os.path.dirname(os.path.abspath(__file__)),
    os.path.join(NVSHMEM_HOME, "include"),
]

library_dirs = [
    os.path.join(NVSHMEM_HOME, "lib"),
]

# --- 链接库 ---
# 一般只需 nvshmem_host + nvshmem；cuda/cudart 由 PyTorch/runtime 解决，保守起见保留
libraries = [
    "nvshmem_host",
    "nvshmem",
    "cudart",
    "cuda",
]

# --- NVSHMEM 开关（可选）---
# 设置为 True 启用 NVSHMEM，False 则只用核心结构
USE_NVSHMEM_FLAG = os.environ.get("USE_NVSHMEM", "0") == "1"

if USE_NVSHMEM_FLAG:
    print("[INFO] NVSHMEM support ENABLED")
else:
    print("[INFO] NVSHMEM support DISABLED (core structures only)")

# --- 额外编译/链接参数 ---
extra_cflags = [
    "-O3", "-std=c++17", "-fPIC",
]

extra_nvcc = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "-Xcompiler=-fPIC",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
] + gencodes

# 如果启用 NVSHMEM，添加相关定义和RDC
if USE_NVSHMEM_FLAG:
    extra_cflags.append("-DUSE_NVSHMEM")
    extra_nvcc.append("-DNVSHMEM_HAVE_CUDA")
    extra_nvcc.append("-rdc=true")  # Only needed for NVSHMEM device calls
    print("[INFO] RDC (Relocatable Device Code) enabled for NVSHMEM")
else:
    print("[INFO] RDC disabled for simpler linking")

# 运行期无需再配 LD_LIBRARY_PATH
extra_link_args = [
    f"-L{os.path.join(NVSHMEM_HOME, 'lib')}",
    f"-Wl,-rpath,{os.path.join(NVSHMEM_HOME, 'lib')}",
]

ext = ce.CUDAExtension(
    name="flashmoe_rdma_cuda",
    sources=[
        "flashmoe_bindings_rdma.cpp",
        "flashmoe_kernel_no_nvshmem.cu",  # Full structure, NVSHMEM optional
    ],
    include_dirs=include_dirs,
    library_dirs=library_dirs if USE_NVSHMEM_FLAG else [],
    libraries=libraries if USE_NVSHMEM_FLAG else ["cudart", "cuda"],
    extra_compile_args={
        "cxx": extra_cflags,
        "nvcc": extra_nvcc,
    },
    extra_link_args=extra_link_args if USE_NVSHMEM_FLAG else [],
)

setup(
    name="flashmoe_rdma",
    version="0.1.0",
    ext_modules=[ext],
    cmdclass={"build_ext": ce.BuildExtension},
)
