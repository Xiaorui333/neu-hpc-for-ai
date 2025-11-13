# modal_deepseek_simple.py
import modal

# 最简单可用：CUDA 12.8 开发镜像 + PyTorch cu124 轮子 + 必要构建工具
# modal_deepseek_simple.py（修正后的关键片段）
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .env({
        "PYTHONUNBUFFERED": "1",
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "8.0",  # A100
        "CC": "gcc",            
        "CXX": "g++", 
    })
    .apt_install(
        "build-essential", "gcc", "g++", "python3-dev",
        "cmake", "ninja-build", "pkg-config", "git",
    )
    # 1) 先装带 CUDA 的 PyTorch（包名放位置参数，索引用 index_url）
    .pip_install("torch==2.4.1+cu124", index_url="https://download.pytorch.org/whl/cu124")
    # 2) 再装构建依赖
    .pip_install("numpy", "setuptools", "wheel", "ninja", "pybind11", "packaging")
    # 3) 同步源码
    .add_local_dir(".", remote_path="/root/deepseek_moe")
)


app = modal.App("deepseek-moe-min")

@app.function(image=image, gpu="A100-40gb", timeout=1800)
def run():
    import os, subprocess
    os.chdir("/root/deepseek_moe")

    # 1) 编译安装 CUDA 扩展（开发模式，改代码可立即生效）
    subprocess.run(["python", "setup.py", "build_ext", "--inplace"], check=True)

    # 2) 运行你的自测入口
    subprocess.run(["python", "deepseek_moe.py"], check=True)
