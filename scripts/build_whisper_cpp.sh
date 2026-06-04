#!/usr/bin/env bash
# scripts/build_whisper_cpp.sh
# 编译 vendored 的 whisper.cpp，产出：
#   - native 二进制  → bin/   whisper-quantize : 导出/量化【必需】，ggml_export.py 调它做 q5_0
#                            whisper-cli      : 【可选】验证工具，跑导出的 ggml 模型抽查转录质量
#                                               （仅 ggml_export.py --test_audio 用，非生产路径）
#   - WASM 产物      → static/wasm/    (libmain.js, libmain.wasm, helpers.js；浏览器端推理用)
#
# 两个一起编：whisper-cli 几乎零额外成本（重头的 libggml/libwhisper 是 quantize 本来就要编的，
# cli 只是多链一个 cli.cpp）。想精简也可只 --target whisper-quantize。
#
# 用法:
#   scripts/build_whisper_cpp.sh                 # native + WASM 都编
#   scripts/build_whisper_cpp.sh --native-only   # 只编 native（采集/导出机用，无需 emsdk）
#   scripts/build_whisper_cpp.sh --wasm-only      # 只编 WASM
#
# 固化了 P0-1 / P0-2 踩过的坑（见 TODO_WHISPER_CPP_WASM.md）：
#   - Ubuntu 24.04 的 libsdl2-dev cmake config 损坏 → **不传** -DWHISPER_SDL2=ON
#   - 新版 emcc 5.x 下 WHISPER_WASM_SINGLE_FILE=ON 会 truncate 内嵌 wasm → 必须 OFF
#   - WASM 的真 CMake target 名是 libmain（whisper.wasm 带点不是真 target）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WHISPER_CPP_DIR="$PROJECT_ROOT/vendor/whisper.cpp"
BIN_OUT="$PROJECT_ROOT/bin"
WASM_OUT="$PROJECT_ROOT/static/wasm"

DO_NATIVE=1
DO_WASM=1
case "${1:-}" in
  --native-only) DO_WASM=0 ;;
  --wasm-only)   DO_NATIVE=0 ;;
  ""|--all)      ;;
  *) echo "用法: $0 [--all|--native-only|--wasm-only]"; exit 2 ;;
esac

if [ ! -f "$WHISPER_CPP_DIR/CMakeLists.txt" ]; then
  echo "❌ 找不到 $WHISPER_CPP_DIR —— 先拉 submodule:"
  echo "     git submodule update --init --recursive"
  exit 1
fi

# ------------------------- native: whisper-quantize（必需）+ whisper-cli（可选验证）------------
if [ "$DO_NATIVE" = 1 ]; then
  echo "🔨 [native] 配置 + 编译 (whisper-quantize 必需 / whisper-cli 可选验证)..."
  # 不传 -DWHISPER_SDL2=ON（Ubuntu 24.04 libsdl2-dev cmake config 损坏，且我们不需要 SDL2）
  cmake -B "$WHISPER_CPP_DIR/build" -S "$WHISPER_CPP_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$WHISPER_CPP_DIR/build" -j --config Release --target whisper-cli whisper-quantize
  mkdir -p "$BIN_OUT"
  cp "$WHISPER_CPP_DIR/build/bin/whisper-cli"      "$BIN_OUT/"
  cp "$WHISPER_CPP_DIR/build/bin/whisper-quantize" "$BIN_OUT/"
  echo "✅ [native] → $BIN_OUT/ (whisper-cli, whisper-quantize)"
fi

# ------------------------- WASM: libmain (非流式 whisper.wasm) -------------------------
if [ "$DO_WASM" = 1 ]; then
  # 找 emcc：优先 PATH，其次自动 source ~/emsdk
  if ! command -v emcc >/dev/null 2>&1 && [ -f "$HOME/emsdk/emsdk_env.sh" ]; then
    echo "ℹ️ 未在 PATH 找到 emcc，自动 source ~/emsdk/emsdk_env.sh"
    # shellcheck disable=SC1091
    source "$HOME/emsdk/emsdk_env.sh" >/dev/null 2>&1 || true
  fi

  if ! command -v emcc >/dev/null 2>&1; then
    echo "⚠️ [wasm] 跳过：找不到 emcc。先装并激活 Emscripten:"
    echo "     git clone https://github.com/emscripten-core/emsdk.git ~/emsdk"
    echo "     cd ~/emsdk && ./emsdk install latest && ./emsdk activate latest && source ./emsdk_env.sh"
    echo "     然后重跑: $0 --wasm-only"
  else
    echo "🔨 [wasm] emcmake 配置 + 编译 libmain (SINGLE_FILE=OFF)..."
    emcmake cmake -B "$WHISPER_CPP_DIR/build-em" -S "$WHISPER_CPP_DIR" \
      -DWHISPER_WASM_SINGLE_FILE=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$WHISPER_CPP_DIR/build-em" -j --target libmain
    mkdir -p "$WASM_OUT"
    # 产物默认在 build-em/bin/；用 find 兜底不同 emcc 版本的输出位置
    js="$(find "$WHISPER_CPP_DIR/build-em" -name 'libmain.js' | head -1)"
    wasm="$(find "$WHISPER_CPP_DIR/build-em" -name 'libmain.wasm' | head -1)"
    if [ -z "$js" ] || [ -z "$wasm" ]; then
      echo "❌ [wasm] 没找到 libmain.js / libmain.wasm，检查编译输出"; exit 1
    fi
    cp "$js"   "$WASM_OUT/libmain.js"
    cp "$wasm" "$WASM_OUT/libmain.wasm"
    # helpers.js（loadRemote / storeFS，P2-2 前端会用）
    [ -f "$WHISPER_CPP_DIR/examples/helpers.js" ] && cp "$WHISPER_CPP_DIR/examples/helpers.js" "$WASM_OUT/helpers.js"
    echo "✅ [wasm] → $WASM_OUT/ (libmain.js, libmain.wasm, helpers.js)"
  fi
fi

echo "🎉 build_whisper_cpp 完成。"
