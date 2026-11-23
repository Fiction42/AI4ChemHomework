#!/usr/bin/env bash
# 在 Apple Silicon 上搭建可加载 pip 二进制扩展的 Python 环境。
# 背景：DSH/其他运行时自带的 Python 带 hardened runtime + 库验证（Team ID 固定），
#       直接 pip install 的二进制扩展（torch/rdkit/...）会因签名不一致而 dlopen 失败。
# 做法：复制解释器，授予 disable-library-validation 权限后重新签名，并用 --target 把依赖装到 pylibs/。
set -euo pipefail

RUNTIME_PY="${RUNTIME_PY:-/Users/rongronghehe/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python}"
BASE="$RUNTIME_PY/bin/python3.12"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$ROOT/pyrenv" "$ROOT/pylibs"
cp "$BASE" "$ROOT/pyrenv/python3"
cat > "$ROOT/pyrenv/ent.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>com.apple.security.cs.disable-library-validation</key><true/>
</dict></plist>
PLIST
codesign --force --sign - --entitlements "$ROOT/pyrenv/ent.plist" --options=runtime "$ROOT/pyrenv/python3"

# 注意：工作目录路径可能包含冒号，PYTHONPATH 会被冒号切分，故用 /tmp 下无冒号符号链接
ln -sfn "$ROOT/pylibs" /tmp/ch3libs 2>/dev/null || true

PYTHONHOME="$RUNTIME_PY" "$ROOT/pyrenv/python3" -m pip install --target "$ROOT/pylibs" \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  autograd sympy mpmath torch rdkit 'transformers==4.44.2' lmdb timm pyarrow tensorboard

echo "done. 运行方式："
echo "  PYTHONHOME=$RUNTIME_PY PYTHONPATH=/tmp/ch3libs $ROOT/pyrenv/python3 <script>"
