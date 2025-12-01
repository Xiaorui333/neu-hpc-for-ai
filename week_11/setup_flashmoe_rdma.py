from setuptools import setup
from torch.utils import cpp_extension as ce
import os
import sys
import glob

# --- NVSHMEM HOME ---
NVSHMEM_HOME = os.environ.get("NVSHMEM_HOME", "/usr/local/nvshmem")

# --- NVSHMEM 开关（可选）---
# 设置为 True 启用 NVSHMEM，False 则只用核心结构
USE_NVSHMEM_FLAG = os.environ.get("USE_NVSHMEM", "0") == "1"

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

# --- include / library 目录 ---
include_dirs = ce.include_paths() + [
    os.path.dirname(os.path.abspath(__file__)),
]

# 只在启用 NVSHMEM 时添加 NVSHMEM 路径
library_dirs = []
if USE_NVSHMEM_FLAG:
    include_dirs.append(os.path.join(NVSHMEM_HOME, "include"))
    
    # Glob 自动发现所有 lib* 目录
    for libdir_pattern in [os.path.join(NVSHMEM_HOME, d) for d in ["lib", "lib64", "lib32"]]:
        if os.path.isdir(libdir_pattern):
            library_dirs.append(libdir_pattern)

# --- 链接库 ---
# NVSHMEM 2.9.0 使用 libnvshmem_host.so 作为主要动态库
# 不启用时不链接 CUDA（PyTorch 会自动处理，避免符号冲突）
if USE_NVSHMEM_FLAG:
    libraries = [
        "nvshmem_host",  # 主要的 NVSHMEM 动态库
    ]
    # 如果存在 libnvshmem.so，也链接它
    nvshmem_so = os.path.join(NVSHMEM_HOME, "lib", "libnvshmem.so")
    if os.path.exists(nvshmem_so):
        libraries.append("nvshmem")
else:
    libraries = []  # PyTorch 自动处理 CUDA 库

# --- 启用时检查 NVSHMEM 路径 ---
if USE_NVSHMEM_FLAG:
    print(f"[INFO] NVSHMEM support ENABLED")
    print(f"[INFO] Using NVSHMEM_HOME = {NVSHMEM_HOME}")
    
    # 容错检查：使用 glob 自动发现库路径
    header = os.path.join(NVSHMEM_HOME, "include", "nvshmem.h")
    if not os.path.exists(header):
        sys.stderr.write(f"[ERROR] NVSHMEM header not found: {header}\n")
        sys.stderr.write("Please install NVSHMEM or set NVSHMEM_HOME correctly.\n")
        sys.exit(1)
    
    # Glob 查找 libnvshmem*.so（支持 libnvshmem.so 和 libnvshmem_host.so）
    lib_patterns = [
        os.path.join(NVSHMEM_HOME, "lib*", "libnvshmem.so*"),
        os.path.join(NVSHMEM_HOME, "lib*", "libnvshmem_host.so*"),
    ]
    found_libs = []
    for pattern in lib_patterns:
        found_libs.extend(glob.glob(pattern))
    
    if not found_libs:
        sys.stderr.write("[ERROR] NVSHMEM library not found.\n")
        sys.stderr.write("Searched patterns:\n")
        for pattern in lib_patterns:
            sys.stderr.write(f"  - {pattern}\n")
        sys.stderr.write("Please install NVSHMEM or set NVSHMEM_HOME correctly.\n")
        sys.exit(1)
    
    # 自动选择第一个找到的库目录
    detected_lib_dirs = sorted(set(os.path.dirname(lib) for lib in found_libs))
    print(f"[INFO] Auto-detected NVSHMEM library directories:")
    for lib_dir in detected_lib_dirs:
        print(f"      - {lib_dir}")
    
    print("[INFO] Device-initiated RDMA calls will be enabled")
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

# 如果启用 NVSHMEM，添加相关定义
if USE_NVSHMEM_FLAG:
    extra_cflags.append("-DUSE_NVSHMEM")
    extra_nvcc.append("-DNVSHMEM_HAVE_CUDA")
    
    # Note: No -rdc=true needed because we built NVSHMEM with
    # NVSHMEM_ENABLE_ALL_DEVICE_INLINING=1, which makes all device
    # functions inline (no separate device linking required)

# 运行期无需再配 LD_LIBRARY_PATH
# 自动添加所有检测到的 lib 目录到 rpath
extra_link_args = []
if USE_NVSHMEM_FLAG and library_dirs:
    for libpath in library_dirs:
        extra_link_args.extend([
            f"-L{libpath}",
            f"-Wl,-rpath,{libpath}",
        ])

ext = ce.CUDAExtension(
    name="flashmoe_rdma_cuda",
    sources=[
        "flashmoe_bindings_rdma.cpp",
        "flashmoe_kernel_no_nvshmem.cu",  # Full structure, NVSHMEM optional
    ],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,  # PyTorch handles CUDA if NVSHMEM not enabled
    extra_compile_args={
        "cxx": extra_cflags,
        "nvcc": extra_nvcc,
    },
    extra_link_args=extra_link_args,  # Already conditional above
)

setup(
    name="flashmoe_rdma",
    version="0.1.0",
    ext_modules=[ext],
    cmdclass={"build_ext": ce.BuildExtension},
)
