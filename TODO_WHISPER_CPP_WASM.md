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
> - 复用 whisper.cpp 现成的 `examples/stream.wasm/`，支持真流式（边说边出字）

---

## 阶段速览

| 阶段                              | 项数 | 进入下一阶段的硬条件                              |
| --------------------------------- | ---- | ------------------------------------------------- |
| Phase 0 可行性验证                | 4    | tiny 中文模型在目标浏览器达到 ≥1.5× 实时；WER 可接受 |
| Phase 1 服务端 ggml 导出          | 6    | 微调后的模型能正确导出 + 下载                     |
| Phase 2 客户端 WASM 集成          | 6    | 端到端跑通：登录 → 下载模型 → 麦克风流式识别       |
| Phase 3 体验优化                  | 4    | 真实用户可用的体验                                |
| Phase 4 收尾 / 退役旧路径         | 3    | 文档与代码一致                                    |

---

# 🔵 Phase 0 可行性验证（先验证再大改）

## [ ] P0-1 在 Linux 上编译 whisper.cpp 原生二进制

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

---

## [ ] P0-2 编译 whisper.cpp WASM 产物 + 跑通官方 stream demo

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

---

## [ ] P0-3 端到端验证：LoRA → ggml 转换链路

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

---

# 🟢 Phase 1 服务端 ggml 导出流程

## [ ] P1-1 把 whisper.cpp 集成进项目目录

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

## [ ] P1-3 ggml 导出接到微调发布流程

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

---

## [ ] P1-4 数据库 schema：models 表加 `has_ggml` 字段

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

---

## [ ] P1-5 新增下载接口 `/api/download_published_ggml_model`

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

---

## [ ] P1-6 兼容基础模型的 ggml 版本

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

---

# 🟡 Phase 2 客户端 WASM 集成

## [ ] P2-1 WASM 产物部署到 static/wasm/

**任务目标**
- 把编译好的 stream.wasm 产物放到能被前端加载的位置
- 同时考虑产物体积 + gzip 压缩

**相关文件**
- `/home/xiejianan/whisper.cpp/build-em/bin/stream.wasm/`
- `/home/xiejianan/whisper-finetune-plus/static/wasm/`（新建）

**实施步骤**
1. P1-1 的构建脚本已经处理拷贝，确认产物：
   - `static/wasm/libstream.js`
   - `static/wasm/libstream.worker.js`（如果 emcc < 3.1.58）
   - `static/wasm/stream.wasm`（如果用 `-DWHISPER_WASM_SINGLE_FILE=OFF`，否则嵌在 js 里）
2. 推荐 `-DWHISPER_WASM_SINGLE_FILE=OFF` 分离文件 —— WASM 字节流可以走 streaming compilation，启动更快
3. nginx / uvicorn 静态服务确认 `application/wasm` MIME type 正确
4. 加 `Cross-Origin-Embedder-Policy: require-corp` 和 `Cross-Origin-Opener-Policy: same-origin` 响应头 —— SharedArrayBuffer / 多线程 WASM 需要
   - 注意：这两个头会影响 cpolar 隧道，要测一下

**预估工作量** 1-2 小时

---

## [ ] P2-2 改造 `inference_panel.js`：WASM 模型加载流程

**任务目标**
- 现有 [inference_panel.js](static/js/inference_panel.js) 是"调 /api/recognition"模式
- 改造为"加载 WASM → 下载模型到 IndexedDB → 麦克风 → WASM 直接推理"

**相关文件**
- `/home/xiejianan/whisper-finetune-plus/static/js/inference_panel.js`
- `/home/xiejianan/whisper-finetune-plus/templates/index.html`（推理面板部分）
- `/home/xiejianan/whisper.cpp/examples/helpers.js`（直接拷过来 `loadRemote` / `storeFS`）
- `/home/xiejianan/whisper.cpp/examples/stream.wasm/index-tmpl.html`（参考逻辑）

**实施步骤**
1. 把 [helpers.js](whisper.cpp/examples/helpers.js) 拷到 `static/js/wasm_helpers.js`，做 ESM 化（如果项目用 ESM）
2. inference_panel 加 lifecycle：
   - 用户选模型 → 调 `/api/published_ggml_info` 拿 etag / size
   - 调 `loadRemote(url='/api/download_published_ggml', cacheKey=etag)`
   - 进度条 UI：`cbProgress` 回调更新百分比
   - 加载完调 `Module.init('whisper.bin', 'zh')`
3. 推理开始：
   - 用 `<input type="file">` 让用户传 wav → `Module.full(...)` （文件模式）
   - 麦克风模式：参考 stream.wasm 的 SDL2 模拟层，从 `getUserMedia()` 取流喂给 WASM
4. **关键 UX**：模型下载 ~180MB，需要明确的进度条 + "已缓存"标识

**预估工作量** 2-3 天（前端工作量最大的部分）

---

## [ ] P2-3 麦克风流式输入接到 WASM

**任务目标**
- 实现"边说边出字"：MediaStream → 16kHz PCM → WASM `whisper_full` 滑动窗口调用

**相关文件**
- `/home/xiejianan/whisper.cpp/examples/stream.wasm/emscripten.cpp`（参考调用方式）
- `/home/xiejianan/whisper-finetune-plus/static/js/inference_panel.js`

**实施步骤**
1. `navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } })`
2. `AudioContext` + `AudioWorkletProcessor`（或 ScriptProcessorNode 兜底）从 MediaStream 拿到 Float32 PCM
3. ring buffer：每 500ms 切一段，10s 窗口（参考 [stream.cpp:20-22](whisper.cpp/examples/stream/stream.cpp#L20-L22) 的 `step_ms=3000` / `length_ms=10000` / `keep_ms=200`）
4. 调用 WASM 的推理接口，结果实时显示
5. VAD 集成（可选）：[whisper.cpp 内置 Silero-VAD](whisper.cpp/models/convert-silero-vad-to-ggml.py) 改善体验，避免静音段空转

**预估工作量** 1-2 天

**风险点**
- AudioWorklet 在某些浏览器（特别是移动 Safari）支持不一致，可能需要 ScriptProcessorNode 兜底
- 采样率转换：MediaStream 默认是 48kHz，需要降采样到 16kHz

---

## [ ] P2-4 模型缓存策略：IndexedDB + 版本管理

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

---

## [ ] P2-5 浏览器兼容性 + 降级路径

**任务目标**
- 不支持 WASM SIMD 的浏览器（旧 Edge / 老移动浏览器）能继续用现有的 GPU 推理路径
- 浏览器检测 + 友好提示

**相关文件**
- `/home/xiejianan/whisper-finetune-plus/static/js/inference_panel.js`
- `/home/xiejianan/whisper-finetune-plus/inference_controller.py:243` （现有 `/api/recognition` 不能删）

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

## [ ] P2-6 模型选择 UI：基础模型 + 用户微调模型

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

## [ ] P3-1 模型下载进度条 + 离线可用提示

**任务目标**
- 首次下载 ~180MB 的等待体验要明确
- 已缓存的模型给"⚡ 已缓存，可离线使用"标识

**预估工作量** 半天

---

## [ ] P3-2 音频质量实时反馈

**任务目标**
- 复用 [P2-3 音频质量校验](TODO_CODE_REVIEW.md#p2-3-音频质量校验缺失) 的能量算法
- 录音时实时显示 dB 表 / VAD 状态，告诉用户"麦克风有没有收到声音"

**相关文件**
- `/home/xiejianan/whisper-finetune-plus/data_collector.py:validate_audio_quality`（已有的 RMS 算法可移植到前端）

**预估工作量** 半天

---

## [ ] P3-3 移动端浏览器适配

**任务目标**
- iOS Safari 16.4+ / Android Chrome 上能跑
- AudioWorklet 兼容性兜底

**风险点**
- 移动端内存限制：small 模型（~180MB）可能在某些手机上挂掉，需要默认 tiny

**预估工作量** 1 天

---

## [ ] P3-4 WASM 推理失败的 Fallback UI

**任务目标**
- WASM crash / OOM 等异常时，引导用户切到服务端推理
- 不要白屏，不要 console error 然后无任何 UI 反馈

**预估工作量** 2 小时

---

# ⚫ Phase 4 收尾

## [ ] P4-1 旧 GPU 推理路径 deprecation 策略

**任务目标**
- `/api/recognition` 不删，但用法变更：仅作为降级路径，不再是主路径
- 文档明确说明：默认走 WASM，不支持 WASM 才走 GPU
- GPU_LOCK 的用法不变（训练独占），WASM 推理无需 GPU 锁，多用户并发推理彻底解锁

**相关文件**
- `/home/xiejianan/whisper-finetune-plus/inference_controller.py`
- `/home/xiejianan/whisper-finetune-plus/shared_state.py`

**预估工作量** 1 小时（主要是文档）

---

## [ ] P4-2 部署文档更新

**任务目标**
- 部署文档加"编译 whisper.cpp + WASM"一节
- 用户能照着新文档从零部署成功

**相关文件**
- `/home/xiejianan/whisper-finetune-plus/Whisper语音识别项目网页端部署文档.md`
- `/home/xiejianan/whisper-finetune-plus/README.md`

**预估工作量** 半天

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
11. P2-3 麦克风流式
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
| WASM 在中文 small 模型上太慢                 | 🔴 方案否决 | P0-2 提前测；不行就只支持 tiny + base       |
| 量化后中文 WER 大幅退化                      | 🟠 体验差   | P0-3 测；不行就走 fp16（下载更大）          |
| 浏览器 IndexedDB 配额不够                    | 🟠 用户报错 | 监测 `navigator.storage.estimate()`，提示用户 |
| 移动端内存不足跑挂                           | 🟠 部分设备不可用 | 默认走 tiny；small 仅桌面浏览器             |
| Emscripten 工具链编译失败 / 版本兼容性问题   | 🟡 部署门槛高 | 部署文档钉死 emsdk 版本                     |
| cpolar 隧道不支持 COEP/COOP 头               | 🟡 多线程 WASM 不可用 | 单线程 WASM 兜底（性能损失 ~30%）           |
