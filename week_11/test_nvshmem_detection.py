#!/usr/bin/env python3
"""
测试 NVSHMEM 库路径自动检测逻辑
"""
import os
import glob
import sys

NVSHMEM_HOME = os.environ.get("NVSHMEM_HOME", "/usr/local/nvshmem")

print("=" * 80)
print("NVSHMEM Auto-Detection Test")
print("=" * 80)
print(f"\nNVSHMEM_HOME = {NVSHMEM_HOME}")

# 1. 检查头文件
header = os.path.join(NVSHMEM_HOME, "include", "nvshmem.h")
print(f"\n[1] Header file:")
print(f"    Path: {header}")
print(f"    Exists: {os.path.exists(header)}")

# 2. Glob 查找库文件（多种模式）
patterns = [
    os.path.join(NVSHMEM_HOME, "lib*", "libnvshmem.so*"),
    os.path.join(NVSHMEM_HOME, "usr", "lib*", "libnvshmem.so*"),
]
print(f"\n[2] Library search patterns:")
found_libs = []
for pattern in patterns:
    print(f"    Pattern: {pattern}")
    matches = glob.glob(pattern)
    if matches:
        print(f"      → Found {len(matches)} file(s)")
        found_libs.extend(matches)
    else:
        print(f"      → No matches")

print(f"\n    Total found: {len(found_libs)} library file(s)")
for lib in sorted(set(found_libs)):
    print(f"      - {lib}")

# 3. 检测库目录
if found_libs:
    detected_lib_dirs = sorted(set(os.path.dirname(lib) for lib in found_libs))
    print(f"\n[3] Auto-detected library directories:")
    for lib_dir in detected_lib_dirs:
        print(f"      - {lib_dir}")
        # 列出目录内容
        try:
            files = os.listdir(lib_dir)
            nvshmem_files = [f for f in files if "nvshmem" in f.lower()][:5]
            if nvshmem_files:
                print(f"        NVSHMEM files: {', '.join(nvshmem_files)}")
        except Exception as e:
            print(f"        Error listing: {e}")
else:
    print("\n[3] ❌ No library directories detected!")
    print("    Searched in:")
    for libdir in ["lib", "lib64", "lib32"]:
        path = os.path.join(NVSHMEM_HOME, libdir)
        exists = os.path.isdir(path)
        print(f"      - {path} {'✓' if exists else '✗'}")

# 4. 手动检查常见位置
print(f"\n[4] Manual check of common locations:")
for libdir in ["lib", "lib64", "lib32"]:
    path = os.path.join(NVSHMEM_HOME, libdir)
    
    # 检查是否是符号链接
    is_symlink = os.path.islink(path)
    exists = os.path.isdir(path)
    
    if is_symlink:
        target = os.readlink(path)
        status = f"✓ SYMLINK -> {target}"
    elif exists:
        status = "✓ EXISTS"
    else:
        status = "✗ NOT FOUND"
    
    print(f"    {status}: {path}")
    
    if exists:
        # 查找 libnvshmem.so
        lib_files = glob.glob(os.path.join(path, "libnvshmem.so*"))
        if lib_files:
            print(f"              Contains: {', '.join(os.path.basename(f) for f in lib_files)}")
            
            # 显示符号链接详情
            for lib_file in sorted(lib_files):
                if os.path.islink(lib_file):
                    link_target = os.readlink(lib_file)
                    basename = os.path.basename(lib_file)
                    print(f"                {basename} -> {link_target}")

# 5. 使用 find 命令深度搜索（fallback）
if not found_libs:
    print(f"\n[5] Deep search using find command:")
    import subprocess
    try:
        result = subprocess.run(
            ["find", NVSHMEM_HOME, "-name", "libnvshmem.so*", "-type", "f"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.stdout.strip():
            deep_found = result.stdout.strip().split('\n')
            print(f"    Found {len(deep_found)} file(s) via find:")
            for f in deep_found[:10]:
                print(f"      - {f}")
        else:
            print("    No files found via find")
    except Exception as e:
        print(f"    Error running find: {e}")

print("\n" + "=" * 80)
print("✅ Test complete!" if found_libs else "❌ Detection failed!")
print("=" * 80)

