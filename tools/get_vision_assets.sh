#!/usr/bin/env bash
# Fetch + build the offline vision stack: llama.cpp (CPU) + SmolVLM2-500M GGUF.
# Everything lands inside the repo; after this script no network is needed.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x third_party/llama.cpp/build/bin/llama-mtmd-cli ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp third_party/llama.cpp || true
  cmake -S third_party/llama.cpp -B third_party/llama.cpp/build \
        -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build third_party/llama.cpp/build --target llama-mtmd-cli -j "$(nproc)"
fi

mkdir -p models
BASE=https://huggingface.co/ggml-org/SmolVLM2-500M-Video-Instruct-GGUF/resolve/main
[ -f models/SmolVLM2-500M-Video-Instruct-Q8_0.gguf ] || \
  wget -q -O models/SmolVLM2-500M-Video-Instruct-Q8_0.gguf "$BASE/SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
[ -f models/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf ] || \
  wget -q -O models/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf "$BASE/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf"

echo "vision assets ready:"
ls -la models/*.gguf third_party/llama.cpp/build/bin/llama-mtmd-cli
