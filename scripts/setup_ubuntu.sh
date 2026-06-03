#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
LIBTINFO5_DEB_URL="${LIBTINFO5_DEB_URL:-http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb}"
INSTALL_COMPILER_GYM="${INSTALL_COMPILER_GYM:-1}"

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  clang \
  cmake \
  llvm \
  python3-dev \
  python3-pip \
  python3.12-venv \
  wget

if ! ldconfig -p | grep -q "libtinfo.so.5"; then
  tmp_deb="$(mktemp --suffix=.deb)"
  wget -O "$tmp_deb" "$LIBTINFO5_DEB_URL"
  sudo dpkg -i "$tmp_deb"
  rm -f "$tmp_deb"
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -e ".[dev]"

if [[ "$INSTALL_COMPILER_GYM" == "1" ]]; then
  "$VENV_DIR/bin/python" -m pip install compiler_gym==0.2.5 --no-deps
  "$VENV_DIR/bin/python" -m pip install \
    absl-py \
    deprecated \
    docker \
    fasteners \
    grpcio \
    gym==0.26.2 \
    humanize \
    loop-tool-py \
    "networkx<3" \
    "numpy<2" \
    protobuf==3.20.3 \
    pydantic \
    requests \
    tabulate
fi

"$VENV_DIR/bin/python" -m unittest discover -s tests -v
"$VENV_DIR/bin/python" - <<'PY_CHECK'
import shutil
missing = [tool for tool in ("opt", "llc", "llvm-size", "llvm-dis", "llvm-extract", "llvm-as") if shutil.which(tool) is None]
if missing:
    raise SystemExit("Missing LLVM tools: " + ", ".join(missing))
print("LLVM tools: ok")
try:
    import compiler_gym
    env = compiler_gym.make("llvm-v0", disable_env_checker=True)
    print(f"CompilerGym llvm-v0: ok ({len(env.datasets)} datasets)")
    env.close()
except ImportError:
    print("CompilerGym: skipped")
PY_CHECK
