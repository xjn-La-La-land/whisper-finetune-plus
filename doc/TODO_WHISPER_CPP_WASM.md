# Whisper.cpp + WASM 浏览器端 CPU 推理 To-Do

> 本文档汇总了把"服务端 GPU 推理"迁移到"浏览器 WASM 本地 CPU 推理"的实施路径，
> 按 **可行性验证 → 服务端导出 → 客户端集成 → 体验优化 → 收尾** 分阶段推进，
> 每项包含：**任务目标**、**相关文件**、**实施步骤**、**预估工作量**、**完成状态**。
>
> 强制要求：**Phase 0 全部跑通 + 关键指标可接受后才进入 Phase 1**，避免做完发现
> WASM 性能不够 / 量化后 WER 崩塌等架构级风险。
>
> 设计动机：
> - 把推理从服务端 GPU 移到用户浏览器的 CPU，腾出 GPU 给训练 / 多用户使用
> - 浏览器 WASM 方案零安装、跨平台、音频不离开用户机器（隐私更好）
> - 复用 whisper.cpp 现成的 `examples/whisper.wasm/`（**非流式文件模式**）：用户录完
>   整段 / 上传整段 → 一次性 `full_default()` 出结果（2026-05-31 从流式收敛而来，见下）

---

## 方向调整（2026-05-31）：流式 → 非流式

P0-1/P0-2 跑通后做了一次方向收敛，**正式定调**（替代原"实时流式"路线）：

- **客户端 WASM 本地 CPU 推理保留**（原始动机不变：腾 GPU + 隐私 + 零安装）。曾考虑
  "整段上传服务端 GPU"，但那会把腾 GPU / 隐私两个动机退回去，故否决。
- **从"实时流式"改为"非流式整段识别"**：用户录完整段 / 上传整段 → 一次性喂给 WASM。
  - 流式 `stream.wasm` 每 3s 切窗，而 whisper 架构固定吃 30s mel 窗口 → 每 3s 重跑一次
    30s encode 还反复处理重叠音频，N 秒音频要 ~N/3 次 encode；非流式按 30s 连续切窗，
    ~N/30 次 encode，**算力差一个数量级**。
  - 整段不从中间切句，识别更准。
  - small 当初被否决的唯一理由是"流式跟不上"（0.41× 余量）；改非流式后该理由消失，
    small q5_0 单次 ~7.3s "录完即出" 可用。tiny/base 的 WASM 损耗实测仅 ~1.5×（0.75s vs
    原生 0.5s，吃到 SIMD + 8 线程），完全够用。
- **目标产物从 `stream.wasm`/`libstream` 换成 `whisper.wasm`/`libmain`**。JS 接口：
  `Module.init(path_model)` 拿 context index，`Module.full_default(idx, pcmF32, 'zh', nthreads, false)`
  整段一次性推理。注意构建 target 名是 **`libmain`**（带点的 `whisper.wasm` 不是真 target，
  `make libmain`），产物 `libmain.js` + `libmain.wasm`。
- **影响范围**：Phase 1 服务端 ggml 导出**完全不变**（导出链路与流式/非流式无关）；
  Phase 2 大幅简化——删掉 ring buffer / 滑窗 / AudioWorklet 流式 / VAD 防空转，P2-3
  缩成"录音 Blob/上传文件 → 16k 单声道 PCM → 一次 `full_default`"。
- **备选（未采纳，已知可行）**：localhost 原生 `whisper-server`（P0-1 已编出，6.2× 满速、
  零 WASM 损耗，但牺牲零安装，要用户跑一个本地小服务）；WebGPU（打 GPU 不打 CPU、
  ONNX/transformers.js 生态，与现有 ggml 链路不兼容，等于换框架）。

---

## 阶段速览

| 阶段                              | 项数 | 进入下一阶段的硬条件                              |
| --------------------------------- | ---- | ------------------------------------------------- |
| Phase 0 可行性验证                | 4    | tiny 中文模型在目标浏览器达到 ≥1.5× 实时；WER 可接受 |
| Phase 1 服务端 ggml 导出          | 6    | 微调后的模型能正确导出 + 下载                     |
| Phase 2 客户端 WASM 集成          | 6    | 端到端跑通：登录 → 下载模型 → 整段录音/上传识别    |
| Phase 3 体验优化                  | 4    | 真实用户可用的体验                                |
| Phase 4 收尾 / 退役旧路径         | 3    | 文档与代码一致                                    |

---

# 🔵 Phase 0 可行性验证（先验证再大改）

## [x] P0-1 在 Linux 上编译 whisper.cpp 原生二进制

**任务目标**
- 验证 whisper.cpp 在用户机器上能从源码编译通过
- 拿到 `whisper-cli` / `whisper-stream` / `quantize` 三个二进制，后续 Phase 1 服务端导出会用到
- 顺便验证 ffmpeg 集成正常

**相关文件**
- `/home/xiejianan/whisper.cpp/CMakeLists.txt`
- `/home/xiejianan/whisper.cpp/examples/stream/README.md`

**实施步骤**
1. 装编译依赖：`sudo apt install build-essential cmake libsdl2-dev`
2. CPU 模式编译：
   ```bash
   cd /home/xiejianan/whisper.cpp
   cmake -B build -DWHISPER_SDL2=ON
   cmake --build build -j --config Release
   ```
3. 检查产物：`build/bin/whisper-cli`、`build/bin/whisper-stream`、`build/bin/quantize`
4. 下载 tiny 中文模型：`bash models/download-ggml-model.sh tiny`
5. 跑一个 sample：`./build/bin/whisper-cli -m models/ggml-tiny.bin -f samples/jfk.wav -l zh -oj`
6. 记录 tiny 在本机的推理速度（音频时长 / 转录时长）

**预估工作量** 1 小时

**验收**：CLI 能成功转录英文+中文 sample 音频，记录基准延迟数据。

✅ 已完成 @ 2026-05-25
   方案：去掉 `-DWHISPER_SDL2=ON` 规避 Ubuntu 24.04 libsdl2-dev 的 cmake config bug，
   只编 whisper-cli / whisper-quantize / whisper-server。native whisper-stream 由
   WASM 版替代（见 P0-2），不需要再编 SDL2 链路。

   **环境**：
     * conda env `whisper` (Python 3.11)
     * cmake 3.28.3 / gcc 13.3.0 / build-essential / libsdl2-dev 都已装
     * libsdl2-dev 的 cmake config 损坏：`/usr/include/SDL2/SDL_config.h` 引用
       不存在的 `_real_SDL_config.h`（Ubuntu 多架构 include 路径未暴露），
       开启 `-DWHISPER_SDL2=ON` 时编译失败。绕过办法：不传该选项

   **构建**：
     ```
     cmake -B build                          # 注意：不加 -DWHISPER_SDL2=ON
     cmake --build build -j --config Release
     ```

   **产物** (`/home/xiejianan/whisper.cpp/build/bin/`)：
     * `whisper-cli`     775 KB —— 主推理 CLI
     * `whisper-quantize` 204 KB —— ggml 量化工具（q5_0 / q4_0 等）
     * `whisper-server`   1.1 MB —— HTTP 推理服务（备选）
     * `whisper-bench`    24 KB
     * `whisper-vad-speech-segments` 487 KB —— VAD 切片
     * 注意：`main` / `bench` / `stream` (19 KB) 是 deprecation 占位，不是真二进制

   **转换链路顺手打通**（提前完成 P0-3 的一部分）：
     * 用现有 `whisper-base-models/whisper-tiny`（HF safetensors）直接转 ggml
     * `convert-h5-to-ggml.py` 第 102 行依赖 openai/whisper repo 的
       `whisper/assets/mel_filters.npz` (4.3 KB)
     * 解决方案：单独 curl 下载这一个文件到临时目录，不装 openai-whisper 整包
       （避免拉一堆依赖污染 env）
     * 产物：`/tmp/ggml-out/ggml-model.bin` = 78 MB (fp16)，与 [models/README.md](../whisper.cpp/models/README.md) 标称大小一致

   **关键性能数据**（tiny native，默认 4 线程）：
     * 测试音频：3.12 秒中文"明天会下雨吗"
     * 总耗时：**500.99 ms**（encoder 313 ms 占大头，decoder 2 ms）
     * **速度 ≈ 6.2× 实时**
     * CPU 占用 305%（3-4 核）；内存 buffer 合计 ~130 MB（conv+encode+cross+decode）
     * 转录结果："明天會下雨嗎"（繁体输出，与现有 [inference_controller.py:298](inference_controller.py#L298)
       一样可走 `zhconv.convert(text, "zh-cn")` 后处理）

   **对 Go/No-Go 的影响**：native 6.2× 实时是个**很强的指标**。WASM 即便保守按
   50% 性能损失，tiny 也有 ~3× 实时，**远超 1.5× 阈值**。但 small 模型计算量
   约 5-7 倍，WASM 上 small 可能落到 0.5-1× 实时——P0-2 必须实测确认。

   **遗留 / 影响后续步骤**：
     * P1-2 `ggml_export.py` 应**自带** `mel_filters.npz`（直接拷进
       `vendor/whisper.cpp/` 或单独 `assets/` 目录），避免依赖外部 repo
     * Native `whisper-stream` 没编 —— 不影响项目，P0-2 走 WASM
     * P4-2 部署文档要写明：Ubuntu 24.04+ 用 `cmake -B build`（不带 `-DWHISPER_SDL2=ON`）

---

## [x] P0-2 编译 whisper.cpp WASM 产物 + 跑通官方 stream demo

**任务目标**
- 验证 Emscripten 工具链能在本机编出 WASM
- 用 ggml-tiny 中文模型在浏览器里跑一段流式识别，**拿到真实可感的延迟数据**
- 这是整个方案的核心赌点：WASM 性能必须够，否则后面全白做

**相关文件**
- `/home/xiejianan/whisper.cpp/examples/stream.wasm/README.md`
- `/home/xiejianan/whisper.cpp/examples/stream.wasm/index-tmpl.html`

**实施步骤**
1. 装 Emscripten（推荐 emsdk 管理）：
   ```bash
   git clone https://github.com/emscripten-core/emsdk.git ~/emsdk
   cd ~/emsdk && ./emsdk install latest && ./emsdk activate latest
   source ~/emsdk/emsdk_env.sh
   ```
2. 编 WASM：
   ```bash
   cd /home/xiejianan/whisper.cpp
   mkdir -p build-em && cd build-em
   emcmake cmake ..
   make -j stream.wasm
   ```
3. 本地起 HTTP 服务：`python3 examples/server.py`（默认 8000）
4. 浏览器打开 `http://localhost:8000/stream.wasm`
5. 加载 tiny 中文模型（页面上点 "load model" 上传 ggml-tiny.bin）
6. 点 "Start"，对着麦克风说一段中文，记录：
   - **首句出字延迟**（开口到第一段文字出现的间隔）
   - **持续吞吐**（说 10 秒 → 转录耗时多少）
   - **CPU 占用率**（Chrome devtools 看）
   - **内存占用**

**预估工作量** 2-3 小时（emsdk 装一次比较慢）

**验收**：tiny 模型在 Chrome 上首句出字 ≤2s，持续吞吐 ≥1.5× 实时，浏览器不卡。

**风险点**
- 如果 tiny 都达不到 1.5× 实时 → 方案否决，回 GPU 路径
- 如果 small 太慢但 tiny 够用 → 限制只支持 tiny / base / small-q5_0

✅ 已完成 @ 2026-05-25
   方案：`-DWHISPER_WASM_SINGLE_FILE=OFF` 编出分离的 `.wasm` 文件（默认 ON 嵌入
   base64 在新版 emcc 5.x 下会 truncate），手动把 `libstream.js` / `libstream.wasm`
   拷到 `bin/stream.wasm/` 部署目录，并修补 demo HTML 以支持自定义模型上传。

   **环境**：
     * Emscripten emcc 5.0.7（emsdk latest，~370 MB 一次性下载）
     * Node v22.16.0（emsdk 自带）
     * 浏览器：Chrome / Edge on Windows，访问 WSL2 上的 localhost:8000
     * SharedArrayBuffer 可用 —— 实测 WASM 用了 **8 线程**多线程跑

   **构建**（在 P0-1 已有 build 之外，单独搞一个 build-em）：
   ```
   git clone https://github.com/emscripten-core/emsdk.git ~/emsdk
   cd ~/emsdk && ./emsdk install latest && ./emsdk activate latest
   export PATH="$HOME/emsdk:$HOME/emsdk/upstream/emscripten:$PATH"

   cd /home/xiejianan/whisper.cpp
   mkdir -p build-em && cd build-em
   emcmake cmake .. -DWHISPER_WASM_SINGLE_FILE=OFF   # 关键：必须 OFF
   make -j libstream                                  # 注意目标名是 libstream
   # 手动部署到 stream.wasm/ 子目录：
   cp ../build-em/bin/libstream.js   stream.wasm/stream.js
   cp ../build-em/bin/libstream.wasm stream.wasm/libstream.wasm
   ```

   **产物** (`build-em/bin/stream.wasm/`)：
     * `stream.js`        97 KB —— Emscripten 生成的 JS loader
     * `libstream.wasm`   1.2 MB —— 真正的 WASM 字节码
     * `index.html`       21 KB —— demo 页（需打补丁，见下）
     * `helpers.js`       6.5 KB —— IndexedDB 缓存工具

   **跑测服务**：`python3 examples/server.py`（whisper.cpp 自带，默认 8000）
   **URL**：`http://localhost:8000/whisper.cpp/stream.wasm/`（末尾路径必须带 `/stream.wasm/`，
   server.py 的 `/whisper.cpp/` 根路径默认重定向到 `whisper.wasm` 那个 demo，不是 stream）

   **遇到的坑（5 个，都要写进部署文档）**：
     1. `-DWHISPER_SDL2=ON` 在 Ubuntu 24.04 libsdl2-dev 包 cmake config 损坏，
        编译失败 —— 不需要 SDL2 时直接去掉这个选项（P0-1 同样的坑）
     2. `make stream.wasm` 报 "No rule to make target" —— 因为 CMake target 名带点
        会被 make 当成文件路径。必须 `make libstream`（实际的库 target）
     3. `WHISPER_WASM_SINGLE_FILE=ON`（默认）在新版 emcc 5.x 下嵌入的 WASM 会被
        truncate，浏览器报 `CompileError: section was shorter than expected`。
        必须 `=OFF`，生成独立 `.wasm` 文件
     4. CMake 的 post-build 复制只更新 `.js`，**不复制新生成的 `.wasm`**。改 OFF
        重编后要手动 cp `libstream.wasm` 到 `stream.wasm/` 子目录（文件名必须叫
        `libstream.wasm`，因为 JS loader 里 hardcode 了 `findWasmBinary()` → `locateFile("libstream.wasm")`）
     5. `stream.wasm/index.html` 里 `<input type="file">` 被 HTML 注释包起来（demo
        故意只暴露预设按钮），且 **`loadFile()` 函数压根没定义**。要：
        a) 解开注释；b) 自己写一个 loadFile，里面读文件、设 `model_whisper = '...'`
        （这步漏了 Start 按钮永远灰着）、调 `storeFS(fname, buf)` 写进 WASM FS

   **关键性能数据**（WASM 8 线程，CPU only，stream.wasm 默认 step_ms=3000 / length_ms=10000）：

   | 模型 | 文件大小 | WASM 内存峰值 | whisper_full() 单次 | 流式余量（step/单次）| 流式可用？ |
   |---|---|---|---|---|---|
   | **tiny** (HF→ggml fp16) | 78 MB | ~140 MB | **0.75 s** | **4×** | ✅ 流畅 |
   | base (官方多语言 fp16)  | 147 MB | ~250 MB | 2.00 s | 1.5× | ✅ 临界 |
   | **small fp16** (HF→ggml) | 488 MB | ~720 MB (预估) | — | — | ❌ 加载即 OOM |
   | **small q5_0** (quantize) | 175 MB | ~330 MB | **7.30 s** | **0.41×** | ❌ 跑得动但跟不上 |

   **流式余量公式**：step_ms / whisper_full() 耗时。≥1× 才不积压；≥1.5× 留 CPU 抖动余量。

   **small fp16 OOM 内存账本**（OOM 在 compute (cross) 分配后炸）：
     * 模型权重 487 MB + KV (self/cross/pad) 80 MB + compute buffers (conv/encode/cross/decode)
       约 155 MB ≈ **720 MB**。Emscripten 默认 MAXIMUM_MEMORY=2GB，加上堆碎片实际 ~1.5GB，
       720MB 理论够但实际分大块连续内存时碰碎片就炸。**架构上不能上 fp16 small**。

   **识别质量观察**（同一段中文输入，未微调）：
     * tiny：`北京官營您` / `我們一堅去哪裡玩呢` —— 错字多 + 全繁
     * base：`北京寬迎迎` / `今天是星期幾` —— 准确度好一些但仍全繁
     * **small q5_0**：`您好,北京欢迎您` / `明天我们去哪里玩` —— **明显更准 + 简体**
     * 所有模型都有静音幻觉（`(字幕君)` / `(拍攝)` / token repetition `你你你你`），
       与 WASM 无关，是 Whisper 模型本身问题。需要 VAD + 后处理

   **重要结论**：
     1. **WASM 推流方案可行**：tiny / base 都能流式跑，tiny 余量充裕（4×）
     2. **生产强烈推荐 tiny**：4× 余量 + 78MB 下载快 + 对 LoRA 微调最敏感
     3. **small fp16 完全不可用**：在 WASM 默认配置下加载即 OOM
     4. **small q5_0 不适合流式**（0.41× 余量），但**质量明显胜出**，可作为
        "非流式高质量模式"备选（例如：录完即出，能接受 7s 延迟的场景）
     5. **双轨潜在方案**：
        * 实时流式 → 微调后的 tiny（边说边出）
        * 高质量批量 → small q5_0（whisper.wasm 文件模式，录完即出）

   **后续影响**：
     * P1-1 集成脚本要自动处理 5 个坑（特别是 SINGLE_FILE=OFF + 手动拷 .wasm + 改名）
     * P1-2 ggml_export 量化默认 q5_0 仍合理（即便不用于流式，也可用于批量模式）
     * P2-1 静态部署时记得把 `libstream.wasm` 放在跟 `stream.js` 同目录
     * P2-2 前端集成时要写一个完整的 loadFile（在 demo HTML 漏的基础上加）
     * P3-3 移动端 small 几乎不能用（连桌面 fp16 都 OOM），默认只暴露 tiny
     * P4-2 部署文档要写明：访问 URL 是 `/stream.wasm/` 不是根路径
     * 关键风险登记表里把 "WASM 在中文 small 上太慢" 升级为 "small fp16 OOM；
       q5_0 流式跑不动；只能 tiny/base 流式"

   **Phase 0 Go/No-Go 倾向**：**强 Go**。WASM tiny 性能远超阈值，唯一限制（small
   不能流式）跟你项目的实际场景（tiny + LoRA 微调）天然匹配。最后等 P0-3 LoRA
   端到端验证通过后，正式进 Phase 1。

---

## [x] P0-3 端到端验证：LoRA → ggml 转换链路

**任务目标**
- 验证现有 LoRA 微调产物能成功转 ggml 格式
- 用现有 [tflite_export.py:107](tflite_export.py#L107) 的 LoRA merge 逻辑作起点
- 找到 transformers 版本兼容性的坑（whisper.cpp 的 `convert-h5-to-ggml.py` 对 transformers 版本有要求）

**相关文件**
- `/home/xiejianan/whisper.cpp/models/convert-h5-to-ggml.py`
- `/home/xiejianan/whisper-finetune-plus/tflite_export.py:107` (LoRA merge 参考实现)
- `/home/xiejianan/whisper-finetune-plus/output/<某用户>/<某模型>/checkpoint-final` (现成的 LoRA 产物)

**实施步骤**
1. 选一个现有的 LoRA checkpoint
2. 临时 Python 脚本：
   ```python
   from transformers import WhisperForConditionalGeneration
   from peft import PeftModel
   base = WhisperForConditionalGeneration.from_pretrained("whisper-base-models/whisper-tiny")
   merged = PeftModel.from_pretrained(base, "<lora_path>").merge_and_unload()
   merged.save_pretrained("/tmp/merged_hf")
   ```
3. 调用 whisper.cpp 转换脚本：
   ```bash
   python /home/xiejianan/whisper.cpp/models/convert-h5-to-ggml.py /tmp/merged_hf ./whisper.cpp /tmp/ggml-out
   ```
4. 用 `whisper-cli` 推理转出来的 .bin，对照原 transformers pipeline 的转录结果
5. 量化测试：`./build/bin/quantize /tmp/ggml-out/ggml-model.bin /tmp/ggml-out/ggml-q5_0.bin q5_0`
6. 量化前后用 [whisper-finetune-plus/uploads/](uploads/) 里同一段中文音频跑推理，对比 CER

**预估工作量** 半天

**验收**：
- 转换流程无错
- LoRA 微调效果保留：转完的 ggml 模型在同一段音频上的转录与原 PyTorch 推理一致或差距 ≤2% CER
- q5_0 量化后 CER 退化 ≤5%（如果超过，量化方案降级或干脆不量化）

✅ 已完成 @ 2026-06-04
   做法：**直接产品化为 [ggml_export.py](ggml_export.py)（即 P1-2），用它跑现成
   checkpoint 完成 P0-3 验证**——P0-3 与 P1-2 是同一份代码，合并完成。

   **链路**（`run_ggml_export`）：
     1. `WhisperForConditionalGeneration.from_pretrained(base)` + `PeftModel.from_pretrained(ckpt).merge_and_unload()`
     2. `merged.save_pretrained(tmp)` + **`WhisperProcessor.save_pretrained(tmp)`**
        —— 关键坑：convert 脚本要读 `vocab.json`/`added_tokens.json`，而 `save_pretrained`
        只存权重+config，必须把 processor 也存进去，否则 convert 报缺文件。
     3. subprocess 调 `whisper.cpp/models/convert-h5-to-ggml.py` → `ggml-model.bin` (fp16)
     4. subprocess 调 `build/bin/whisper-quantize <fp16> <out> q5_0`

   **mel_filters.npz 坑已闭环**（P0-1 遗留项）：convert 脚本第 102 行硬依赖
   `<dir_whisper>/whisper/assets/mel_filters.npz`，已 vendored 进
   [vendor/whisper_convert_assets/whisper/assets/mel_filters.npz](vendor/whisper_convert_assets/)（4.3KB），
   脚本自包含，不再依赖 `/tmp` 或外部 openai/whisper repo。

   **环境**：conda `whisper`（transformers 4.51.3 / peft 0.15.2 / torch 2.10.0+cu130）。
   convert + quantize 在这套版本上跑通，无版本兼容问题（P0-3 担心的点已排除）。

   **验证数据**（checkpoint：`output/小满/whisper-small-lora-v1/checkpoint-final`，
   音频：`uploads/admin/task_804.wav`）：

   | 路径 | 产物大小 | 识别结果 |
   |---|---|---|
   | PyTorch（合并 LoRA） | — | `明天会下雨吗` |
   | ggml **fp16** | 465.0 MB | `明天会下雨吗` |
   | ggml **q5_0** | 167.1 MB | `明天会下雨吗` |

   **三路完全一致**：PyTorch == fp16 == q5_0，该样本 CER 差距 = 0%。LoRA 效果完整保留、
   q5_0 量化无损；且输出**直接是简体**（small 微调模型，无需再 zhconv）。文件大小与
   P0-2 标称（small fp16 ~488MB / q5_0 ~175MB）吻合。

   **遗留（不阻塞进 Phase 1，但属正式验收的补充）**：
     * 本次只测了 1 段音频（定性 3 路一致）。严格 CER ≤2%/≤5% 阈值应跑一个多样本集
       （建议用 `uploads/` 已有标注音频批量对比）。核心风险已排除，量化退化补测可放到 P1-2 收尾。
     * CLI 用法：`python ggml_export.py --base_model whisper-base-models/whisper-small
       --checkpoint <ckpt> --output <out>.bin --quantize q5_0 --test_audio <wav>`

---

## [ ] P0-4 关键性能数据汇总 + Go/No-Go 决策

**任务目标**
- 把 P0-1 / P0-2 / P0-3 的数据汇总成一个表
- 跟用户对齐"基于这些数据，是否进入 Phase 1"
- 如果 No-Go，回到 GPU 路径或考虑 native 二进制方案

**汇总指标**
| 指标                       | tiny | small | small-q5_0 | 可接受阈值       |
| -------------------------- | ---- | ----- | ---------- | ---------------- |
| Native CLI 推理速度（×RT） | ?    | ?     | ?          | ≥3× 实时         |
| WASM 推理速度（×RT）       | ?    | ?     | ?          | ≥1.5× 实时       |
| WASM 内存占用（MB）        | ?    | ?     | ?          | ≤2GB             |
| 中文 CER（vs PyTorch）     | ?    | ?     | ?          | ≤5% 绝对差距     |
| 首句延迟（s）              | ?    | ?     | ?          | ≤2s              |
| 模型下载大小（MB）         | 75   | 466   | ~180       | small-q5_0 可接受 |

**预估工作量** 1 小时（汇总 + 讨论）

**验收**：用户明确批准进入 Phase 1。

✅ 决策已定 @ 2026-05-31：**Go**，但架构从"实时流式"收敛为"非流式整段识别"（见顶部
   「方向调整」一节）。仍保留客户端 WASM 本地 CPU 推理，目标产物换成 `whisper.wasm`/`libmain`。
   P0-3（LoRA → ggml 链路）与流式/非流式无关，仍需先跑通再正式进 Phase 1。

---

# 🟢 Phase 1 服务端 ggml 导出流程

## [x] P1-1 把 whisper.cpp 集成进项目目录

**任务目标**
- 让 whisper.cpp 跟项目代码一起部署，部署文档里有清晰的编译步骤

**相关文件**
- `/home/xiejianan/whisper-finetune-plus/`（新增子模块或子目录）

**实施步骤**
1. 决策：git submodule vs. 拷贝源码 vs. apt 包
   - 推荐 **git submodule**：升级 whisper.cpp 主线时一条命令搞定
   - 替代方案：把 whisper.cpp 拷进 `vendor/whisper.cpp/`，钉一个 commit
2. 在项目根目录加：
   ```bash
   git submodule add https://github.com/ggml-org/whisper.cpp.git vendor/whisper.cpp
   git submodule update --init --recursive
   ```
3. 写一个 `scripts/build_whisper_cpp.sh`：
   - 自动 cmake + make
   - 拷贝产物到 `bin/`（whisper-cli, quantize）和 `static/wasm/`（libstream.js, *.wasm）
4. 部署文档加一节"首次部署：编译 whisper.cpp"
5. `.gitignore` 加 `vendor/whisper.cpp/build*/`、`bin/`、`static/wasm/*.wasm`

**预估工作量** 1-2 小时

✅ 已完成 @ 2026-06-04

   **集成方式：git submodule（采纳文档推荐），钉死在验证过的 commit**
     * `vendor/whisper.cpp` = `ggml-org/whisper.cpp` @ `0ccd896f`（v1.8.4-328，即 P0-1/2/3
       用的同一 commit），见 `.gitmodules`。升级主线只需进 submodule `git checkout <新commit>`
       后在父仓库 `git add vendor/whisper.cpp`。
     * 新克隆务必 `git clone --recursive`（或 `git submodule update --init --recursive`），
       README「快速开始」第 1 步已改。

   **构建脚本 [scripts/build_whisper_cpp.sh](scripts/build_whisper_cpp.sh)**（native + WASM 都已实测通过）：
     * `--native-only`：cmake 编 `whisper-cli` + `whisper-quantize` → 拷到 `bin/`
     * `--wasm-only`：emcmake 编 `libmain` → 拷 `libmain.js`/`libmain.wasm` + `helpers.js` 到 `static/wasm/`
       （脚本会自动 source `~/emsdk/emsdk_env.sh`）
     * 默认两个都编。P0-1/P0-2 的坑（不传 `-DWHISPER_SDL2=ON`、`SINGLE_FILE=OFF`、target 名
       `libmain`）已固化进脚本注释。
     * 实测产物：`bin/whisper-cli`(998K) / `bin/whisper-quantize`(208K)；
       `static/wasm/libmain.js`(99K) / `libmain.wasm`(1.2M) / `helpers.js`(6.5K)。

   **与原 step 5 的偏差（已和用户确认）：WASM 产物入库，native 二进制 gitignore**
     * `static/wasm/*`（libmain.js/.wasm/helpers.js，~1.3MB）**提交进库**——WASM 平台无关，
       入库后部署机免装 emsdk(370MB)、免重编，`git clone --recursive` 完即可服务。
     * `bin/`（native 二进制，平台相关）**gitignore**。
     * 故 `.gitignore` 实际只加了 `/bin/`，**没有**加 `static/wasm/*.wasm`。
     * 部署影响：纯采集机/普通用户无需碰 whisper.cpp（WASM 随仓库走、浏览器跑）；只有
       「导出/量化模型」的 GPU 机要 `scripts/build_whisper_cpp.sh --native-only`（供
       [ggml_export.py](ggml_export.py) 调 `bin/whisper-quantize` + `vendor/whisper.cpp` 的 convert 脚本）。

   **ggml_export.py 路径已切换**：优先 `vendor/whisper.cpp` + `bin/`，回退 `~/whisper.cpp`，
   构建前后都能用（全 env 变量可覆盖）。

   **遗留**：P1-1 step4 原指「部署文档加一节」——production runbook(`deploy/README.md`)
   的更新留到 P4-2（WASM 尚未接进前端，现在写进生产手册会误导）；开发向说明已写进 README。

---

## [ ] P1-2 新增 `ggml_export.py`：LoRA → 合并 → 转 ggml → 量化

**任务目标**
- 复制 [tflite_export.py](tflite_export.py) 的结构，但产物是 .bin 而非 .tflite
- 输入：base_model_path + checkpoint_path + 输出路径
- 输出：`whisper_model.bin`（fp16）+ 可选 `whisper_model_q5_0.bin`（量化）

**相关文件**
- `/home/xiejianan/whisper-finetune-plus/tflite_export.py`（参考结构）
- `/home/xiejianan/whisper-finetune-plus/ggml_export.py`（新建）
- `vendor/whisper.cpp/models/convert-h5-to-ggml.py`（调用）

**实施步骤**
1. 新建 `ggml_export.py`，结构对齐 tflite_export：
   ```python
   def run_ggml_export(
       base_model_path: str,
       checkpoint_path: Optional[str],
       output_path: str,
       quantize: str = "q5_0"  # None / "q5_0" / "q4_0"
   ) -> Tuple[bool, str]:
       # 1. 复用 tflite_export.py:107 的 LoRA merge 逻辑
       # 2. save_pretrained 到 tmp 目录
       # 3. subprocess 调 convert-h5-to-ggml.py
       # 4. 可选 subprocess 调 quantize 二进制
       # 5. 把产物复制到 output_path
   ```
2. CLI 入口同 tflite_export：`python ggml_export.py --base_model ... --checkpoint ... --output ... --quantize q5_0`
3. 错误处理：和 tflite_export 一样返 (success, error_msg) 元组，前端能透传

**预估工作量** 0.5 天

**验收**：
- 命令行单独跑通
- 产物用 whisper-cli 推理结果与原 PyTorch 一致（CER 差距 ≤5%）

---

## [x] P1-3 ggml 导出接到微调发布流程

**任务目标**
- 用户点"发布模型"时，除了现有的 tflite 导出，再额外导出 ggml（供浏览器 WASM 用）

**相关文件**
- `/home/xiejianan/whisper-finetune-plus/finetune_controller.py`（publish_model 钩子）
- `/home/xiejianan/whisper-finetune-plus/ggml_export.py`

**实施步骤**
1. 找到 finetune_controller.py 里现有的 tflite 导出调用点（P3-5 提到要异步化，可一并做）
2. 在同样位置增加 ggml 导出调用：
   ```python
   ggml_ok, ggml_err = run_ggml_export(base, ckpt, ggml_out, quantize="q5_0")
   ```
3. 失败策略：
   - tflite 失败 → 仍发布（用于 Android）
   - ggml 失败 → 仍发布（用户可用旧 GPU 推理路径，降级）
   - 两个都失败 → 整体发布失败
4. 把 ggml 产物路径加入数据库 models 表

**预估工作量** 2 小时

✅ 已完成 @ 2026-06-05（含 P1-4）

   **关键澄清：导出发生在「训练成功后自动导出」，不在 publish_model**。
   `publish_model` 只把已存在的 tflite 传 ModelScope + 标记发布；真正的 tflite 导出在
   [finetune_controller.py](finetune_controller.py) 的 `run_finetune_process()`（训练
   `returncode==0` 之后）。所以 ggml 导出也加在那同一处，与 tflite 并列。

   **改动**：
     * `finetune_controller.py`：tflite 自动导出块之后新增 ggml 自动导出块，调
       `run_ggml_export(base, checkpoint-final, output/<u>/<m>/ggml/whisper.bin, quantize="q5_0")`，
       产物 `ggml/whisper.bin`(fp16) + `ggml/whisper_q5_0.bin`（浏览器实际加载这个）。
     * **失败策略与 tflite 一致：只记日志、不影响训练成果落库**（用户仍可走旧 GPU 推理降级）。
       文档原写的“两个都失败→整体发布失败”针对的是 publish 语义；此处是训练后自动导出，
       训练已成功，导出失败不该回滚，故只记日志。

   **前置依赖**：导出机（GPU 机）需先 `scripts/build_whisper_cpp.sh --native-only` 编出
   `bin/whisper-quantize`；没编时 `run_ggml_export` 返回 `(False, "找不到 whisper-quantize…")`，
   被 except 记日志跳过，不崩。

   **未做（按设计）**：导出同步跑在 `run_finetune_process` 内（占 GPU 锁到结束），与 tflite
   一样；异步化是 P3-5 的事，本次不动。fp16 中间产物（465MB，浏览器用不上、只 native 可用）
   暂未清理，留待需要时再加 `keep_fp16=False` 之类。

---

## [x] P1-4 数据库 schema：models 表加 `has_ggml` 字段

**任务目标**
- 让 [resolve_user_model_path()](inference_controller.py#L74) 同时返回 has_tflite / has_ggml 状态
- 前端能据此决定走哪条推理路径

**相关文件**
- `/home/xiejianan/whisper-finetune-plus/data/tasks.db`
- `/home/xiejianan/whisper-finetune-plus/utils/db.py` 或 init_db
- `/home/xiejianan/whisper-finetune-plus/inference_controller.py:74-109`

**实施步骤**
1. init_db 给 models 表加 `has_ggml INTEGER DEFAULT 0` 字段（兼容旧行）
2. publish_model 成功导出 ggml 后 UPDATE has_ggml = 1
3. `resolve_user_model_path` 返回字典加入 `has_ggml` 字段（参考现有 `has_tflite` 的实现）

**预估工作量** 30 分钟

✅ 已完成 @ 2026-06-05（随 P1-3 一并）

   **达成方式与文档预设不同（更一致）**：P1-4 真实目标是「`resolve_user_model_path()` 返回
   has_ggml」。但现有 `has_tflite` **并非 DB 列**，而是 inference_controller 读时
   `os.path.exists()` 现查（[inference_controller.py](inference_controller.py)）。为保持一致，
   has_ggml 也按文件存在性现查（判据 = `ggml/whisper_q5_0.bin` 是否存在），**不加 DB 列**。
   `/api/user_models` 现已为每个模型返回 has_ggml。
     * 好处：免 schema 迁移、免在 publish 时 UPDATE、删模型时不会残留脏状态。
     * 已用现成的 `output/小满/whisper-small-lora-v1`（早先 P0-3 产出的 q5_0）验证 has_ggml=True。

---

## [x] P1-5 新增下载接口 `/api/download_published_ggml_model`

**任务目标**
- 跟现有 `/api/download_published_model`（tflite）并列，但返回 ggml .bin
- 前端 WASM 加载流程会 fetch 这个接口

**相关文件**
- `/home/xiejianan/whisper-finetune-plus/inference_controller.py:146`（仿 download_published_model）

**实施步骤**
1. 仿照 [inference_controller.py:146](inference_controller.py#L146) 写一个新 router
2. 路径：`GET /api/download_published_ggml`，鉴权同 `get_current_user`
3. 返回 FileResponse，正确设置 Content-Type=`application/octet-stream` 和 Content-Length
4. **重要**：响应头加 `Cache-Control: public, max-age=31536000`，配合前端 IndexedDB 缓存
5. 加 ETag（用文件 mtime + size），方便前端校验"模型有没有更新"
6. 新增 `/api/published_ggml_info`：仿 [latest_model_info](inference_controller.py#L117)，返回 size / version / hash，前端用来判断要不要重新下载

**预估工作量** 1-2 小时

✅ 已完成 @ 2026-06-05

   两个接口都加在 [inference_controller.py](inference_controller.py)，与 tflite 的
   `latest_model_info` / `download_published_model` 并列，服务
   `output/<user>/<model>/ggml/whisper_q5_0.bin`：
     * `GET /api/published_ggml_info` → `{has_published, model_name, version_tag, file_size, etag}`
     * `GET /api/download_published_ggml` → FileResponse（filename `whisper_<model>_q5_0.bin`，
       `application/octet-stream`，Content-Length 由 FileResponse 自动给）
     * 共用 helper `_resolve_published_ggml`（查 is_published=1 → safe_resolve_under 定位文件）
       和 `_ggml_etag`（`"{int(mtime)}-{size}"`，两端一致便于前端比对）。

   **缓存策略：纠正了文档原话的一个隐患**。原 step4 让下载 URL 挂 `max-age=31536000`，
   但该 URL 固定、内容会随“发布了另一个模型”变化 → 固定 URL + 长缓存会取到旧文件。改为：
     * 带 `?v=<version_tag>`（前端从 info 接口取）→ URL 随版本唯一 → `public, max-age=31536000, immutable`（安全）；
     * 不带 `v` → `no-cache` 兜底。
   真正的缓存本来就靠前端 IndexedDB（P2-4），HTTP 缓存只是锦上添花。ETag/Last-Modified 由
   FileResponse + 我们的 ETag 提供。

   **接口命名**：用了 step2/step6 的 `/api/download_published_ggml` + `/api/published_ggml_info`
   （比标题里的 `..._ggml_model` 更简洁、与 info 接口对仗）。

   **验证**：路由注册 ✅；`_ggml_etag` 真文件 ✅；`_resolve_published_ggml` 未发布→None ✅；
   临时置 is_published=1（dev 库，测完已还原）happy path → info 返回 has_published=True +
   file_size 175209680 + etag ✅。

---

## [x] P1-6 兼容基础模型的 ggml 版本

**任务目标**
- 没微调过的用户也能用 WASM 推理：直接用 base 模型的 ggml 版本
- 现有 `whisper-base-models/{whisper-tiny, whisper-small}` 需要导出对应的 ggml 版本

**相关文件**
- `/home/xiejianan/whisper-finetune-plus/whisper-base-models/`
- `/home/xiejianan/whisper-finetune-plus/download_whisper_models.py`（参考下载基础模型的脚本）

**实施步骤**
1. 写一个 `scripts/build_base_ggml.py`：
   - 遍历 `whisper-base-models/*`
   - 对每个跑 convert-h5-to-ggml.py
   - 输出到 `whisper-base-models/ggml/{tiny,small}.bin` 和量化版本
2. 部署文档加一步"首次部署：生成基础模型的 ggml 版本"
3. `inference_controller.resolve_base_model_paths()` 增加 ggml 路径解析
4. `/api/download_base_ggml?model=tiny` 接口，无认证（基础模型公开）

**预估工作量** 2 小时

✅ 已完成 @ 2026-06-05

   **不写独立 `build_base_ggml.py`，而是集成进 [ggml_export.py](ggml_export.py)（采纳用户建议）**：
   `run_ggml_export(checkpoint_path=None)` 本就支持「纯基座、不融合 LoRA」，所以基座导出
   不用重写任何 convert/quantize 逻辑，只加了一层薄封装：
     * `run_base_ggml_export(names=None, quantize="q5_0")` 遍历基座、各自调 `run_ggml_export`，
       输出 `whisper-base-models/ggml/<name>(_q5_0).bin`。
     * CLI 新增 `--base-models [NAME...]`：`python ggml_export.py --base-models`（默认
       `DEFAULT_BASE_MODELS = tiny/base/small`；**medium 跳过**——q5_0 ~500MB / fp16 ~1.5GB，
       浏览器 WASM 2GB 上限装不下）。

   **服务端（[inference_controller.py](inference_controller.py)）**：
     * `GET /api/download_base_ggml?model=whisper-tiny` —— **保留 JWT 鉴权**（与全站一致，
       前端本就登录态；未采纳文档"无认证"）。基座内容恒定 → 直接 `immutable` 长缓存，
       无需 P1-5 那样的 `?v=`。
     * `_resolve_base_ggml`：白名单校验（必须是真实基座目录名）+ `safe_resolve_under` 防穿越。
     * **关键坑**：基座 ggml 输出到 `whisper-base-models/ggml/`，会被 `resolve_base_model_paths()`
       和 `_list_base_model_dirs()` 的 `os.scandir` 误当成一个基座。已在**两个扫描器**里
       `entry.name != "ggml"` 跳过。

   **前端可见（[finetune_controller.py](finetune_controller.py)）**：`/api/base_models` 的每个
   基座加 `has_ggml`（= `ggml/<name>_q5_0.bin` 是否存在），前端 P2-6 据此决定能否走 WASM。

   **部署文档**：README「快速开始」加第 4b 步 `python ggml_export.py --base-models`。
   `whisper-base-models/` 已 gitignore → base ggml 不入库、部署时现生成（需先有 `bin/whisper-quantize`）。

   **验证**：批量导出 tiny/base/small 成功（q5_0 30M/55M/175M）；两个扫描器不再含 "ggml"；
   `_resolve_base_ggml`：whisper-tiny→路径✅ / ggml→None✅ / `../etc/passwd`→None✅；
   `/api/base_models` has_ggml = base/tiny/small True、medium/large False ✅。

   **遗留**：同 P1-3，fp16 中间产物（base 三个共 ~714MB）暂留未清，浏览器只用 q5_0；
   要省盘可给 ggml_export 加 `keep_fp16=False`。

---

# 🟡 Phase 2 客户端 WASM 集成

## [x] P2-1 WASM 产物部署到 static/wasm/

**任务目标**
- 把编译好的 **whisper.wasm（非流式）** 产物放到能被前端加载的位置
- 同时考虑产物体积 + gzip 压缩

**相关文件**
- `/home/xiejianan/whisper.cpp/build-em/bin/whisper.wasm/`（注意不是 stream.wasm）
- `/home/xiejianan/whisper-finetune-plus/static/wasm/`（新建）

**实施步骤**
1. 编译目标是 **`libmain`**（`whisper.wasm` 带点不是真 target，会被 make 当文件路径）：
   ```bash
   cd /home/xiejianan/whisper.cpp/build-em
   emcmake cmake .. -DWHISPER_WASM_SINGLE_FILE=OFF   # 同 P0-2：必须 OFF
   make -j libmain
   ```
2. P1-1 的构建脚本处理拷贝，确认产物（沿用 P0-2 的 5 个坑，只是文件名 stream→main）：
   - `static/wasm/libmain.js`
   - `static/wasm/libmain.worker.js`（如果 emcc < 3.1.58）
   - `static/wasm/libmain.wasm`（`-DWHISPER_WASM_SINGLE_FILE=OFF` 时独立文件；JS loader 里
     hardcode 了 `locateFile("libmain.wasm")`，文件名必须一致）
3. 推荐 `-DWHISPER_WASM_SINGLE_FILE=OFF` 分离文件 —— WASM 字节流可以走 streaming compilation，启动更快
4. nginx / uvicorn 静态服务确认 `application/wasm` MIME type 正确
5. 加 `Cross-Origin-Embedder-Policy: require-corp` 和 `Cross-Origin-Opener-Policy: same-origin` 响应头 —— SharedArrayBuffer / 多线程 WASM 需要
   - 注意：这两个头会影响隧道（现生产走 Cloudflare named tunnel），要实测确认能透传

**预估工作量** 1-2 小时

✅ 已完成 @ 2026-06-05（产物部署在 P1-1 已做，本次补 MIME + 隔离头）

   [main.py](main.py)：
     * `mimetypes.add_type("application/wasm", ".wasm")` —— 保证 StaticFiles 用
       `application/wasm` 发 libmain.wasm（流式编译 + COEP 下 MIME 必须正确）。
     * 全局中间件 `add_cross_origin_isolation`：所有响应加
       `COOP: same-origin` + `COEP: require-corp` → 浏览器 `crossOriginIsolated` 生效，
       SharedArrayBuffer / pthread 多线程 WASM 可用。
     * 已核对 templates/index.html 全同源/本地（echarts/tailwind/vue 都 vendored，唯一
       "外部"是 favicon 的 data: URL）→ require-corp 不会拦掉任何现有子资源。

   **本地验证**（httpx ASGITransport 直连 app）：`GET /` → COOP/COEP ✅；
   `libmain.wasm` → content-type application/wasm + COEP ✅。

   **✅ 部署已验证（de-risk 闸门已过）@ 2026-06-06**：生产代码 sync + 重启后，在
   https://app.carespeechai.cn 浏览器控制台实测 `crossOriginIsolated === true`——Cloudflare
   具名隧道**原样透传** COOP/COEP，多线程 WASM 在线上可用，无需 coi-serviceworker / 单线程兜底。

   **未做（可选优化）**：gzip/br 预压缩（libmain.wasm 1.2MB；uvicorn 静态默认不压，
   生产可由 Cloudflare 边缘压缩，暂不在应用层做）。

---

## [x] P2-2 改造 `inference_panel.js`：WASM 模型加载 + 非流式推理

**任务目标**
- 现有 [inference_panel.js](static/js/inference_panel.js) 是"整段录音/上传 → POST /api/recognition"模式
- 改造为"加载 WASM → 下载模型到 IndexedDB → 整段音频 → WASM 一次性 `full_default` 推理"
- **好消息**：现有前端已经是"录完整段才发"（[inference_panel.js:210](static/js/inference_panel.js#L210)
  `MediaRecorder.onstop` 才出 Blob），交互骨架可直接复用，只把"发给后端"换成"喂给 WASM"

**相关文件**
- `/home/xiejianan/whisper-finetune-plus/static/js/inference_panel.js`
- `/home/xiejianan/whisper-finetune-plus/templates/index.html`（推理面板部分）
- `/home/xiejianan/whisper.cpp/examples/helpers.js`（直接拷过来 `loadRemote` / `storeFS`）
- `/home/xiejianan/whisper.cpp/examples/whisper.wasm/index-tmpl.html`（**非流式**参考逻辑，不是 stream.wasm）

**实施步骤**
1. 把 [helpers.js](whisper.cpp/examples/helpers.js) 拷到 `static/js/wasm_helpers.js`，做 ESM 化（如果项目用 ESM）
2. inference_panel 加模型 lifecycle：
   - 用户选模型 → 调 `/api/published_ggml_info` 拿 etag / size
   - 调 `loadRemote(url='/api/download_published_ggml', cacheKey=etag)` → `storeFS('whisper.bin', buf)`
   - 进度条 UI：`cbProgress` 回调更新百分比
   - 加载完调 `const idx = Module.init('whisper.bin')`（返回 context index，缓存起来复用）
3. 推理（非流式，一次性）：
   - 录音 `onstop` 出的整段 Blob，或 `<input type="file">` 上传的整文件
   - → 用 `AudioContext.decodeAudioData` 解码 → 取单声道 → 重采样到 16kHz Float32 PCM（见 P2-3）
   - → `Module.full_default(idx, pcmF32, 'zh', nthreads, false)` 一次返回
   - 转录文本通过 demo 的 `print` 回调累积取出，再走 `zhconv` 简繁转换 + 去标点（复用现有后处理）
4. **关键 UX**：模型下载 tiny ~78MB / small q5_0 ~175MB，需要明确的进度条 + "已缓存"标识

**预估工作量** 1-2 天（比原流式方案省，无 mic 流式 / 滑窗逻辑）

✅ 代码完成 @ 2026-06-06（⏳ 待浏览器验收。本批同时覆盖 P2-3/P2-4/P2-5/P2-6 + 简繁）

   **没拷 helpers.js 改 ESM，而是自写更干净的 [static/js/wasm_whisper.js](static/js/wasm_whisper.js)**
   （helpers.js 的 loadRemote 带 confirm() 弹窗 + 硬编码 dbName，不适合）。该模块导出：
     * `detectCapabilities()` —— WASM/SIMD/IndexedDB/crossOriginIsolated 探测（P2-5）。
     * `ensureRuntime()` —— 注入 libmain.js、配 `window.Module`（print 捕获 + locateFile→/static/wasm/）。
     * `ensureModel({cacheKey,url,onProgress})` —— IndexedDB 缓存 → 命中直接用，未命中 fetch（带进度）
       → storeFS 写 WASM FS → `Module.init`。切模型先 `Module.free` 旧 context（P2-4）。
     * `transcribe(pcm,'zh')` —— `full_default` 跑后台线程，**解析 print 流**：段落行
       `[ts --> ts] 文本` 收集（时间戳转秒，与服务端 results 一致），见 `whisper_print_timings: total time`
       判完成 resolve；带 5min 超时。
     * `decodeToPcm16k(blob)` —— P2-3，见下。
     * `evictModel` / `isCached` / `getLoadedKey`。

   **inference_panel.js 整文件重写**（[static/js/inference_panel.js](static/js/inference_panel.js)）：
     * **分段开关 本地/服务端**（Option B，与用户敲定的设计）；浏览器不支持本地则灰掉 + 默认服务端。
     * **模型卡片状态机**：unavailable / need-download / downloading(进度条) / loading / ready(⚡已缓存)。
     * 路由：选中模型 `has_ggml` 且浏览器支持 → 本地 WASM；否则回退 `/api/recognition`（P2-5）。
       本地下载走 P1-5 的 `/api/download_published_ggml?v=` 或 base 的 `/api/download_base_ggml`。
     * **参数面板砍掉**（用户要求）：BEAMS / 简体 / 去标点三个控件移除，焊死
       `num_beams=1 + 转简体 + 保留标点`（目标用户是小朋友/家长，零配置）。

   **简繁转换**：服务端用 Python `zhconv`；本地 WASM 在浏览器用 [static/js/zhconv_lite.js](static/js/zhconv_lite.js)
   复刻 zhconv 的「前缀集 + 最长匹配」算法 + 从同一个 zhconv 导出的表
   [static/vendor/zh2cn.json](static/vendor/) (8039 条/132KB，lazy-load)。**已与 zhconv 做一致性
   测试**（含 1500 字 torture 串全一致）。base 模型输出繁体，靠它转简体。

   **后端配合（已在 P2-1 / P1-x 就位）**：COOP/COEP + wasm MIME；user/base 的 has_ggml 与下载接口。

   **静态校验**：3 个 JS `node --check` 通过；模块导出齐全；zhconv 一致性测试通过。
   **未验证（需浏览器）**：libmain 运行时 init / full_default print 解析 / IndexedDB 缓存 /
   音频 decode→transcribe 端到端 / Vue 模板渲染。验收步骤见交付说明。

---

## [x] P2-3 整段音频解码 → 16kHz PCM → 一次性喂 WASM

> 原"麦克风流式滑窗"已随非流式方向取消（无 ring buffer / 滑窗 / AudioWorklet 流式 /
> VAD 防空转）。本项只剩"把整段录音/上传文件转成 whisper 要的 16k 单声道 Float32 PCM"。

**任务目标**
- 整段 Blob/文件 → 16kHz 单声道 Float32 PCM → `Module.full_default(idx, pcm, 'zh', nthreads, false)` 一次返回

**相关文件**
- `/home/xiejianan/whisper.cpp/examples/whisper.wasm/emscripten.cpp`（`full_default` 接口定义：`(index, audioVal, lang, nthreads, translate)`）
- `/home/xiejianan/whisper-finetune-plus/static/js/inference_panel.js`

**实施步骤**
1. 拿到整段音频：录音 `MediaRecorder.onstop` 的 Blob，或 `<input type="file">` 上传的文件
2. `const ab = await blob.arrayBuffer()` → `await audioCtx.decodeAudioData(ab)` 解码为 AudioBuffer
   （能吃 webm/opus、wav、mp3 等，浏览器原生解码，无需自己处理容器格式）
3. 取单声道：`audioBuffer.getChannelData(0)`（多声道则混下来）
4. 重采样到 16kHz：用 `OfflineAudioContext(1, ceil(duration*16000), 16000)` 跑一遍，
   或简单线性插值（MediaRecorder 默认 48kHz → 16kHz）
5. 把 Float32 PCM 传给 `Module.full_default(idx, pcm, 'zh', nthreads, false)`，一次性返回
6. 结果经 demo 的 `print` 回调收齐 → `zhconv` 简繁 + 去标点（复用现有后处理）

**预估工作量** 0.5-1 天（比原流式方案大幅缩水）

**风险点**
- 长音频（>30s）由 whisper.cpp 内部按 30s 连续切窗处理，无需前端切；但 small q5_0
  单窗 ~7.3s，长录音的总耗时要给用户明确的"处理中"进度反馈（不能像流式那样边说边出）
- `decodeAudioData` 对个别浏览器的 webm/opus 兼容性，必要时 `MediaRecorder` 指定 `audio/wav`

✅ 代码完成 @ 2026-06-06（⏳ 待浏览器验收）：`wasm_whisper.decodeToPcm16k(blob)` ——
   `blob.arrayBuffer()` → `decodeAudioData`（吃 webm/opus/wav/mp3）→ `OfflineAudioContext(1, …, 16000)`
   重采样为 16k 单声道 → `getChannelData(0)` Float32 喂 `full_default`。处理中 UI：状态栏 spinner +
   「本地识别中…已用时 Xs」。

---

## [x] P2-4 模型缓存策略：IndexedDB + 版本管理

**任务目标**
- 用户首次访问下载模型（~180MB），之后从 IndexedDB 秒加载
- 用户重新发布模型后，浏览器能感知并下载新版本

**相关文件**
- `/home/xiejianan/whisper-finetune-plus/static/js/wasm_helpers.js`（loadRemote）
- inference_controller.py 的 `/api/published_ggml_info`（P1-5）

**实施步骤**
1. IndexedDB key 设计：`ggml_{username}_{version_tag}_{etag}`
2. 加载流程：
   ```
   GET /api/published_ggml_info → 拿到 latest_etag
   IndexedDB.has(latest_etag)?
     yes → 直接读出来 → WASM
     no  → fetch /api/download_published_ggml → 进度条 → 存 IndexedDB → WASM
   ```
3. 旧版本清理：每次成功加载新版本后，删 IndexedDB 里 ≥30 天未访问的旧版本（避免缓存膨胀）
4. 量化模型与 fp16 模型分两个 key（用户可以切换）

**预估工作量** 1 天

✅ 代码完成 @ 2026-06-06（⏳ 待浏览器验收）：IndexedDB 缓存在 `wasm_whisper.js`
   （库 `whisper-ggml-cache`/store `models`）。缓存键由 inference_panel 给：用户模型
   `pub:{user}:{version_tag}:{etag}`、基座 `base:{name}`（基座不变，免版本）——重发布换
   version_tag/etag 即换键、自动重下。`isCached`/`evictModel`（重新下载）已实现。
   **未做**：≥30 天旧版本自动清理（暂靠浏览器配额；后续再加）。

---

## [x] P2-5 浏览器兼容性 + 降级路径

**任务目标**
- 不支持 WASM SIMD 的浏览器（旧 Edge / 老移动浏览器）能继续用现有的 GPU 推理路径
- 浏览器检测 + 友好提示

**相关文件**
- `/home/xiejianan/whisper-finetune-plus/static/js/inference_panel.js`
- `/home/xiejianan/whisper-finetune-plus/inference_controller.py:243` （现有 `/api/recognition` 不能删）

✅ 代码完成 @ 2026-06-06（⏳ 待浏览器验收）：`detectCapabilities()` 探测
   WASM/SIMD/IndexedDB/crossOriginIsolated，不支持 → 分段开关「本地」灰掉 + 黄条提示原因，
   默认落到「服务端 GPU」走 `/api/recognition`（保留未删）。单个模型无 ggml 时也提示切服务端。

**实施步骤**
1. JS 启动时检测：
   ```js
   const hasWasm = typeof WebAssembly !== 'undefined';
   const hasSIMD = await WebAssembly.validate(simd_test_bytes);
   const hasIndexedDB = 'indexedDB' in window;
   ```
2. 不支持 → 在 UI 上显示"您的浏览器不支持本地推理，将使用服务端推理（速度较慢）"
3. 降级走现有的 `/api/recognition` 路径（不删）
4. 推荐浏览器：Chrome 91+ / Firefox 89+ / Edge 91+ / Safari 16.4+

**预估工作量** 半天

---

## [x] P2-6 模型选择 UI：基础模型 + 用户微调模型

✅ 代码完成 @ 2026-06-06（⏳ 待浏览器验收）：下拉合并 base + 用户模型（本地模式标注「无本地版」），
   下方「模型卡片」显示大小/缓存状态/下载按钮；切模型时 `ensureModel` 自动 `Module.free` 旧
   context 再 init 新的。注：本地可用的用户模型 = 已发布⭐那一个（受 /api/download_published_ggml
   只服务发布模型所限）；非发布用户模型本地不可用，走服务端。

**任务目标**
- 用户能在前端切换"用 base tiny / base small / 我的微调模型"
- 切换时清空旧 WASM 实例，加载新模型

**相关文件**
- `/home/xiejianan/whisper-finetune-plus/templates/index.html`
- `/home/xiejianan/whisper-finetune-plus/static/js/inference_panel.js`

**实施步骤**
1. 模型 selector 列表来自 `/api/user_models` + `/api/base_models` 合并
2. 切换模型时调 WASM `Module.free()` 释放内存，再 `Module.init()` 加载新模型
3. UI 显示当前模型的 size / 版本 / 缓存状态

**预估工作量** 2-3 小时

---

# 🟠 Phase 3 体验优化

## [x] P3-1 模型下载进度条 + 离线可用提示

**任务目标**
- 首次下载 ~180MB 的等待体验要明确
- 已缓存的模型给"⚡ 已缓存，可离线使用"标识

**预估工作量** 半天

✅ 已完成 @ 2026-06-06（实为 P2-2 顺带做完，此处补记验收）

   下载进度 + 缓存徽章在 P2-2 大改时已实现，本项无新增代码，仅核对确认：
     * **进度条**：[inference_panel.js](static/js/inference_panel.js) `downloadModel()` 把
       `wasm.ensureModel` 的 `onProgress(recv,total)` 映射到 `downloadPct`/`downloadRecvMB`；
       模板 [templates/index.html](templates/index.html) 本地模型卡片有进度条 DOM +
       「⬇ 下载中 {pct}%（{MB}MB）」文字。状态机 `unavailable/need-download/downloading/loading/ready` 齐全。
     * **已缓存/离线徽章**：卡片 ready 态显示「⚡ 已缓存 · 可离线」；缓存判定走
       IndexedDB（`wasm.isCached`），命中即标 ready。
   **未做（加分项，非本项定义范围）**：`navigator.storage.estimate()` 配额预警、
   `navigator.onLine` 真离线探测（现"可离线"是基于"已缓存"推断）。需要时再补。

---

## [x] P3-2 音频质量实时反馈

**任务目标**
- 复用 [P2-3 音频质量校验](TODO_CODE_REVIEW.md#p2-3-音频质量校验缺失) 的能量算法
- 录音时实时显示 dB 表 / VAD 状态，告诉用户"麦克风有没有收到声音"

**相关文件**
- `/home/xiejianan/whisper-finetune-plus/data_collector.py:validate_audio_quality`（已有的 RMS 算法可移植到前端）

**预估工作量** 半天

✅ 代码完成 @ 2026-06-06（⏳ 待浏览器验收）

   **实现**（[inference_panel.js](static/js/inference_panel.js) + [templates/index.html](templates/index.html)）：
     * 录音开始时 `startMeter(stream)` 在 MediaRecorder 用的同一条流上挂 `AnalyserNode`
       （fftSize 2048，**只 connect 不接 destination**，避免回放啸叫），`requestAnimationFrame`
       逐帧 `getFloatTimeDomainData` 算 RMS。
     * **阈值与服务端对齐**：`MIC_FLOOR = 0.002` = `data_collector.MIN_AUDIO_RMS`，所以
       「电平表显示在收音（绿）」⇔「上传不会被 `validate_audio_quality` 判空拒绝」，两端一致。
     * **电平映射**：dBFS = 20·log10(rms)，-60dB→0% / 0dB→100%（正常说话 ~-30dB ≈ 50%）。
     * **UI（按用户意见集成到按钮，不用独立电平条）**：录音时「停止识别」按钮按质量变色——
       收音→绿（emerald）/ 静音→黄（amber）/ 过载（level>80%）→红，配 `transition duration-300`
       平滑过渡；按钮内附一行小字（🎤 正在收音 / ⚠ 没听到声音 / 🔊 声音过大）兼顾色觉无障碍。
       由 `micQuality`/`micHint`/`recordBtnClass` 三个 computed 驱动。
     * **防抖**：EMA（`smRms=0.8*smRms+0.2*rms`）+ 收音判定带回差（开 0.002 / 关 0.0016），
       避免临界处按钮颜色乱闪。`stopMeter()` 在 `onstop` 取消 rAF / 断 source / 关 AudioContext / 归零。
     * 电平表异常被 try/catch 吞掉，**不影响录音与识别本身**。

   **设计演进**：初版做的是状态栏下方独立电平条；按用户意见改为**把反馈集成进录音按钮**（颜色 + 小字），
   去掉独立条，UI 更紧凑、贴合零配置定位。

   **范围说明**：只做录音路径的实时反馈（上传文件无实时流，不适用）；未做独立 VAD 自动切段
   （非流式整段识别不需要）。`AudioContext` 在点击「开始录音」的用户手势内创建，不受自动播放策略限制。

   **缓存版本链已 bump**：`inference_panel.js?v=1.8`（app.js 内引用）、`app.js?v=2.2`（index.html）。
   **静态校验**：`node --check` 通过；模板引用的 `recordBtnClass/micHint` 已 setup 导出，已确认无残留电平条引用。
   **未验证（需浏览器）**：真实麦克风下按钮变色跟随、静音/过载文案切换、stop 后资源释放。

---

## [x] P3-3 移动端浏览器适配

**任务目标**
- iOS Safari 16.4+ / Android Chrome 上能跑
- AudioWorklet 兼容性兜底

**风险点**
- 移动端内存限制：small 模型（~180MB）可能在某些手机上挂掉，需要默认 tiny

**预估工作量** 1 天

✅ 代码完成 @ 2026-06-06（⏳ 待真机验收）

   **「AudioWorklet 兜底」已作废**：非流式方案用 `decodeAudioData` + `OfflineAudioContext`，
   全程不碰 AudioWorklet，原任务点随流式路线一起取消。本项聚焦真正的移动端问题：**内存** + iOS 细节。

   **内存保守策略（核心，[wasm_whisper.js](static/js/wasm_whisper.js) + [inference_panel.js](static/js/inference_panel.js)）**：
     * `detectCapabilities()` 新增 `isMobile`（`userAgentData.mobile` 优先，回退 UA 正则 +
       iPadOS「伪装成 Mac + maxTouchPoints>1」补判）和 `deviceMemory`（Chrome/Android 有，iOS 为 null）。
     * **默认模型按端选**：手机默认 `whisper-tiny`（最小），桌面仍默认 `whisper-base`
       （`ensureSelectedModel` 内分支）。
     * **大模型下载前内存警告**：手机上选「已知 ≥120MB（small 基座）/ 大小未知的微调模型」时，
       `downloadModel` 先弹 `dialog.confirm`（「仍要下载 / 改用 tiny」），选改用 tiny 自动切到 tiny 基座。
       卡片再常驻一行黄字「📱 手机内存有限…建议改用 tiny」。
     * **与 P3-4 协同**：万一仍 OOM，P3-4 的崩溃捕获 + 回退横幅会兜住，引导切服务端。

   **iOS 细节**：录音电平表的 `AudioContext` 在创建后 `resume()`（iOS 手势后才允许）；
   `AudioContext`/`OfflineAudioContext` 已带 `webkit` 前缀回退（P2-3）。**旧 iOS（<16.4，无
   crossOriginIsolated）**本就被 P2-5 能力闸门拦下、自动落服务端，无需额外处理。

   **响应式布局**：模板原本已全程用断点（`grid-cols-1 lg:grid-cols-5` / `sm:grid-cols-2` /
   `h-[40vh] md:h-[46vh]` / vh 高度），窄屏自动单列堆叠；未做投机性 CSS 改动，留真机核对。

   **缓存版本链已 bump**：`wasm_whisper.js?v=1.3`、`inference_panel.js?v=1.9`、`app.js?v=2.3`。
   **静态校验**：3 个 JS `node --check` 通过；模板 `mobileMemWarn` 绑定有效；新增模板类
   （text-amber-600 / leading-snug）本就在 vendored CSS 里，无需重建 tailwind。
   **未验证（需真机）**：iPhone/Android 实际内存下 small 是否 OOM、电平表 iOS 跟随、UA 判定准确性、窄屏排版。

---

## [x] P3-4 WASM 推理失败的 Fallback UI

**任务目标**
- WASM crash / OOM 等异常时，引导用户切到服务端推理
- 不要白屏，不要 console error 然后无任何 UI 反馈

**预估工作量** 2 小时

✅ 代码完成 @ 2026-06-06（⏳ 待浏览器验收）

   **分两层：运行时把崩溃变成「可感知的快速失败」，UI 把失败变成「一键回退」。**

   **运行时层（[static/js/wasm_whisper.js](static/js/wasm_whisper.js)）**：
     * 新增 `_aborted` 标志 + `isRuntimeBroken()` 导出：`Module.onAbort`（OOM/abort 的总入口）
       触发时置位，本页内本地推理即判为不可恢复（需刷新）。
     * `_markAborted()`：崩溃时**让在飞的 `transcribe` 立刻 reject**（经新增的 `_activeReject`
       钩子），不再干等 5min 超时；同时放掉 `_busy` 互斥、清 context，避免卡死后续操作。
     * `transcribe` 重写为单次触发的 `finish()` 汇聚点（完成/超时/异常/崩溃只触发一次、状态清干净），
       原先 resolve 后 timer 仍在、或异常路径漏清 `_busy` 的隐患一并消除。
     * 崩溃用 `_fatalError()`（带 `.fatal=true`）区分「致命/不可重试」与普通失败；`ensureModel`/
       `transcribe` 在 `_aborted` 时直接快速失败。

   **UI 层（[static/js/inference_panel.js](static/js/inference_panel.js) + [templates/index.html](templates/index.html)）**：
     * `runLocalInference` catch 里：弹**红色回退横幅**（`localError`），不再只是一行错误状态。
     * 横幅两个动作，**用同一段音频、不必重录**（`lastBlob` 缓存最近一次捕获）：
       「☁ 改用服务端」→ 切 mode=server 重跑 `/api/recognition`；「↻ 重试本地」→ 重跑本地。
     * **致命崩溃**（`e.fatal || isRuntimeBroken()`）时额外灰掉「本地」开关 + 提示「刷新页面可重试本地」，
       且隐藏「重试本地」按钮（重试必然再崩）——只留「改用服务端」。
     * 普通失败（解码错误/超时，运行时仍健康）保留「重试本地」。横幅可手动 ✕ 关闭；切模式/新一次识别自动清。

   **缓存版本链已 bump**：`wasm_whisper.js?v=1.2`（inference_panel 内引用）、
   `inference_panel.js?v=1.6`（app.js 内引用）、`app.js?v=2.0`（index.html）。

   **静态校验**：3 个 JS `node --check` 通过；模板引用的 `localError/retryLocal/fallbackToServer/
   dismissFallback` 均已 setup 导出。
   **✅ 浏览器已验收 @ 2026-06-06**：笔记本浏览器实测——用临时调试钩子（`window.__simLocalFail`，
   仅非生产域名生效）分别模拟「普通失败」「致命崩溃」两条路径，横幅渲染、一键「改用服务端」端到端、
   致命态灰掉本地开关均正常。验收后调试钩子已删除（inference_panel.js?v=1.11 / app.js?v=2.5）。
   注：真实 OOM（如手机加载 small）触发的 `onAbort` 路径仍未在真机上跑过；pthread 后台线程内的
   OOM 是否一定回调主线程 `onAbort` 不确定，故保留 5min 超时作兜底——两条路径最终都会落到回退横幅。

---

# ⚫ Phase 4 收尾

## [x] P4-1 旧 GPU 推理路径 deprecation 策略

**任务目标**
- `/api/recognition` 不删，但用法变更：仅作为降级路径，不再是主路径
- 文档明确说明：默认走 WASM，不支持 WASM 才走 GPU
- GPU_LOCK 的用法不变（训练独占），WASM 推理无需 GPU 锁，多用户并发推理彻底解锁

**相关文件**
- `/home/xiejianan/whisper-finetune-plus/inference_controller.py`
- `/home/xiejianan/whisper-finetune-plus/shared_state.py`

**预估工作量** 1 小时（主要是文档）

✅ 已完成 @ 2026-06-06（纯文档/注释，无行为改动）

   **行为本就到位**：前端 P2-5 已是「支持→默认本地 WASM，不支持→落 `/api/recognition`」，
   接口保留未删。WASM 在浏览器跑、不碰服务端 → 天然不占 `GPU_LOCK`，多用户并发识别无需改任何锁逻辑。
   本项只补「说明」：
     * [inference_controller.py](inference_controller.py) `api_recognition` 加 docstring，
       注明它现在是**降级路径**、默认走浏览器 WASM、仍受 GPU_LOCK 约束且与 WASM 路径互不影响。
     * [README.md](README.md)「核心特点」新增第 4 点「浏览器端本地推理（默认推理路径）」，
       并点明不支持时降级到 `/api/recognition`。
   `shared_state.py` 无需改（GPU_LOCK 用法不变，已在注释里说清）。

---

## [x] P4-2 部署文档更新

**任务目标**
- 部署文档加"编译 whisper.cpp + WASM"一节
- 用户能照着新文档从零部署成功

**相关文件**
- ~~`Whisper语音识别项目网页端部署文档.md`~~（**该文件不存在**，路径过时；真实文档是下面两个）
- `/home/xiejianan/whisper-finetune-plus/README.md`
- `/home/xiejianan/whisper-finetune-plus/deploy/README.md`

**预估工作量** 半天

✅ 文档完成 + 生产闸门已过 @ 2026-06-06

   **README.md 早已覆盖编译/导出步骤**（第 4 步 `build_whisper_cpp.sh --native-only`、
   4b 步 `ggml_export.py --base-models`、static/wasm 随仓库走无需重编、emsdk 说明），本次未动。
   **本次补的是生产 runbook [deploy/README.md](deploy/README.md)**（P1-1 当时把它推迟到这里）：
     * 新增「③ 浏览器端 WASM 本地推理（COOP/COEP 透传，上线必查）」一节——讲清默认走 WASM、
       依赖 `crossOriginIsolated`、两个隔离头由 main.py 加但**隧道可能剥掉**的风险，给出
       `crossOriginIsolated` / `curl -I … grep cross-origin` 验证法 + 剥掉后的退路；并说明
       模型来源（WASM 入库免重编、基座 ggml 用 `ggml_export.py --base-models`）、采集机不适用。
     * 「验证」节加一行指到 ③。

   **✅ 运行时闸门已过 @ 2026-06-06**：生产 sync + 重启后，在 https://app.carespeechai.cn
   控制台实测 `crossOriginIsolated === true`，Cloudflare 隧道透传 COOP/COEP 确认无误。
   风险登记表里「隧道不透传 COEP/COOP」一条就此关闭。P4-2 全部完成。

---

## [ ] P4-3 监控埋点 + 灰度

**任务目标**
- 记录每个推理请求走的是 WASM 还是 GPU 降级，方便观察用户分布
- WASM 失败率超过阈值时报警

**实施步骤**
1. 客户端调用 `/api/log_inference_metric`（新增）上报：
   - 走的路径（wasm / gpu fallback）
   - 模型名 / 版本
   - 推理耗时
   - 浏览器 / OS
2. 这些数据存 DB，写一个 `/api/admin/metrics` 接口供调优用

**预估工作量** 半天

---

# 实施顺序建议

**第 1 周（可行性 + 服务端打底）**
1. P0-1 编译 native 二进制
2. P0-2 编译 + 跑 WASM stream demo（**最关键的赌点**）
3. P0-3 LoRA → ggml 链路打通
4. P0-4 Go/No-Go 决策点
5. P1-1 集成 whisper.cpp 到项目
6. P1-2 ggml_export.py
7. P1-3/4/5 接到发布流程 + 数据库 + 下载接口
8. P1-6 基础模型 ggml 版本

**第 2 周（客户端集成主战场）**
9. P2-1 WASM 产物部署
10. P2-2 inference_panel 改造（核心工作量）
11. P2-3 整段音频 → 16k PCM → 一次性推理
12. P2-4 IndexedDB 缓存
13. P2-5 浏览器兼容 + 降级
14. P2-6 模型选择 UI

**第 3 周（体验打磨 + 收尾）**
15. P3-1/2/3/4 体验优化
16. P4-1/2/3 文档 + 灰度

---

# 验收清单

每修完一项，请在该行的 `[ ]` 改为 `[x]`，并在条目末尾追加一行：

```
✅ 已完成 by <名字> @ <日期>
   - PR / commit: <链接>
   - 验证方式: <命令 / 操作步骤 / 关键指标数据>
```

---

# 关键风险登记

| 风险                                         | 影响       | 缓解措施                                    |
| -------------------------------------------- | ---------- | ------------------------------------------- |
| WASM 在中文 small 模型上太慢                 | 🟢 已缓解   | 改非流式后已解：small q5_0 单次 ~7.3s"录完即出"可用；tiny/base 损耗仅 ~1.5× |
| 量化后中文 WER 大幅退化                      | 🟠 体验差   | P0-3 测；不行就走 fp16（下载更大）          |
| 浏览器 IndexedDB 配额不够                    | 🟠 用户报错 | 监测 `navigator.storage.estimate()`，提示用户 |
| 移动端内存不足跑挂                           | 🟠 部分设备不可用 | 默认走 tiny；small 仅桌面浏览器             |
| Emscripten 工具链编译失败 / 版本兼容性问题   | 🟡 部署门槛高 | 部署文档钉死 emsdk 版本                     |
| 隧道不透传 COEP/COOP 头（现走 Cloudflare named tunnel） | 🟢 已关闭 | ✅ 2026-06-06 生产实测 `crossOriginIsolated===true`，Cloudflare 隧道原样透传，无需兜底 |
