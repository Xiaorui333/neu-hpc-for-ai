#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <path/to/file[.cu]>"
  exit 1
fi

src="$1"

# 规范化：若没有 .cu 后缀则补上；有的话就保持不动
if [[ "$src" == *.cu ]]; then
  dst="$src"
else
  dst="${src}.cu"
fi

# 只有在 dst != src 时才做复制；避免把源文件删了
if [[ "$dst" != "$src" ]]; then
  rm -f "$dst"
  cp "$src" "$dst"
fi

# 切到仓库根目录（脚本在 scripts/ 下）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "+ python3.11 -m modal run scripts/modal_nvcc.py --code-path \"$dst\""
python3.11 -m modal run scripts/modal_nvcc.py --code-path "$dst"

