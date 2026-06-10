# 第一梯队前端修复 — 设计文档

- 日期：2026-06-10
- 范围：前端 UX 审计中标记为"第一梯队（高影响）"的 5 个问题
- 验证方式：无前端测试框架（无构建步骤、Vue 全局版纯 JS），统一靠"启动应用 + 浏览器驱动"验证

## 背景

`whisper-finetune-plus` 是 Vue 3（全局版，无打包器）+ Tailwind（预编译 CSS）+ ECharts 的单页应用，中文 UI。
经过 12 维度 UX 审计 + 逐项对抗式核验，确认这 5 个为最高优先级。其中 **4 个为纯前端**（后端已返回所需的一切），
**仅"停止训练"需要后端改动**。

## 已敲定的决策（用户确认）

1. **停止训练后的 checkpoint**：直接丢弃（`rmtree` 输出目录），模型名立即可复用。不保留续训。
2. **停止权限**：仅本人（发起训练的用户）可停止。
3. **无障碍范围**：仅焦点/键盘管理（贴合文档记录的脑瘫儿童+家长/运动障碍人群），**不做** aria-live 屏幕阅读器播报。

## 实施顺序

`⑤ errors.js（共享依赖）` → `① 采集写操作` → `④ 微调表单分层` → `③ 焦点/键盘` → `② 停止训练（后端，风险最高，最后验证）`

理由：⑤ 提供错误文案，① 与微调的 catch 块都经它产出消息；② 改后端、且本机无法完整验证 kill，放最后。

---

## ⑤ 错误文案助手（先做，被 ① 复用）

**问题**：catch 块直接把 `Failed to fetch`、裸 422 detail 等原始英文/后端文本甩给用户；只说"哪里错"不说"怎么办"。

**方案**：新建 `static/js/errors.js`（ES module，按现有约定以 `./errors.js?v=1` 引入；不引入打包器/CDN）。导出：

- `friendlyHttpError(res, fallback?) -> Promise<string>`：按 **HTTP 状态码** 映射（不解析 detail 文本）：
  - `401` → 返回空串，交给 `api.js` 已有的全局 `whisper:unauthorized` 处理（避免与 ① 重复弹窗）
  - `413` → `"文件太大，无法上传。请录制更短的音频后重试。"`（只认状态码，不解析其 HTML body —— 当前应用无体积限制，413 只可能来自 Cloudflare 隧道）
  - `422` → `"提交的内容不符合要求，请检查后重试。"`（仅当 detail 为中文时附加）
  - `423 / 409 / 400` → 回显后端 detail（这些消息后端已用中文精心写过且可操作，如 GPU 占用、模型名规则）
  - `5xx` → `"服务器开小差了，请稍后重试；若多次失败请联系管理员。"`
- `friendlyNetworkError(err) -> string`：**绝不**插入 `err.message`（那是 "Failed to fetch" 泄漏处），固定返回 `"无法连接服务器，请检查网络后重试。"`
- `isChinese(s)` / `safeDetail(s)` 守卫：保证泛化兜底永不回显原始英文；默认兜底 `"操作失败，请稍后重试。"`

**关键点**：助手只产出**字符串**，渲染层不变（`dialog.alert` 与 inference 的 `typeStatus` 两个出口都能用）。
SSE 错误帧（`finetune_panel.js` 的 `data.message`）是已解析的事件负载、非 Response，单独处理或直接用后端中文消息。

**涉及文件**：`static/js/errors.js`（新建）

**风险**：中文检测对"中文+英文路径名"混排会判为中文从而部分泄漏路径（可接受）；务必保留各调用点的 `!res.ok` 分支（`apiFetch` 对 HTTP 错误不抛异常）。

---

## ① 采集页写操作：错误吞掉 + 无 loading 态

**问题**：录音上传 `uploadAudio`（[audio_collector.js:138](../../../static/js/audio_collector.js)）已正确处理，是模板。其余 5 个写操作
（TXT 导入、手动加句、内联编辑、删除、清空）fire-and-forget：不查 `res.ok`、无 try/catch、且在确认成功**前**就清空输入（失败时静默丢句）。

**方案**：照搬 `uploadAudio` 模式（无新依赖、无构建、复用 `dialog.js`）：

1. `setup()` 增加 per-action in-flight ref：`busyImport`/`busyAdd`/`savingTaskId`/`deletingTaskId`/`clearing`，并加入 return 对象。
2. 每个写操作包 `try { if(busy) return; busy=true; const res=await apiFetch(...); const data=await res.json().catch(()=>({})); if(!res.ok) throw new Error(await friendlyHttpError(res)); ...success... } catch(e){ if(e.message) dialog.alert(e.message, {variant:'danger'}); } finally { busy=false }`。
   - 注意 `friendlyHttpError` 对 401 返回空串（交全局 `whisper:unauthorized` 处理），故 catch 内 **仅当 `e.message` 非空才弹窗**，避免弹出空白 danger 框、也避免与全局会话过期提示重复。
3. **取消乐观清空**：`txtInput.value=''` / `newTaskText.value=''` / `editingTaskId.value=null` 移到 `res.ok` 之后（失败保留用户输入）。
4. TXT 导入成功：用后端返回的条数 `dialog.alert(data.message, {variant:'success'})` 显示"成功导入 N 条"。其余写操作**成功保持静默**（列表自动刷新即可，避免给家长/儿童频繁弹窗），**仅失败弹窗**。
5. 模板：对应按钮绑 `:disabled` + 复用文件里已有的 `animate-spin` SVG 标记（从 index.html:642-644 / 754-757 精确复制）。
6. `fetchTasks()` 留在 `finally` 让列表无论成败都重新同步。

**涉及文件**：`static/js/audio_collector.js`、`templates/index.html`

**后端**：无需改动。已知契约：所有 `HTTPException` body 统一 `{detail:...}`；`upload_txt` 返回 `{message:'成功导入 N 条...'}`。
（已知局限：`delete_task` 即使 not-found 也返回 200，故新 `!res.ok` 检查无法识别该逻辑失败 —— 本修复只改善传输/5xx/校验类错误，不动后端。）

**风险**：`saveEdit` 已先 `await dialog.confirm`，in-flight ref 必须在 confirm 解析为 true **之后**才置位，否则用户取消会卡死；新 `:disabled` 绑定的 ref 必须真的从 `setup()` 返回，否则 Vue 静默当 undefined（永不禁用）。

---

## ④ 微调表单：ML 黑话分层（渐进式披露）

**问题**：`AdaLora`/`FP16`/`Eval Loss`/`LR`/`epochs`/`batch` 直接铺给非技术家长，看不懂且易误配。

**方案**：纯前端。默认值已在三层（`finetune_panel.js:139-151` == `FinetuneRequest` == `finetune.py` argparse）严格对齐且合理，**不改默认值**，只做披露：

1. `finetune_panel.js` 加 `const showAdvanced = ref(false)` 并加入 return。
2. index.html step-2 卡片：**简版常显**只留 模型名称 + 基础模型 + 训练轮数（标签改白话，如"AI 要学多少遍，越多越久"）。
3. `learning_rate`/`batch_size`/`accumulation_steps`/三个 checkbox/音频时长范围 全部移入 `<div v-show="showAdvanced">`（用 `v-show` 而非 `v-if`，避免双滑块重新挂载丢值/布局抖动）。
4. 简↔高之间插入披露开关：`⚙️ 高级设置（保持默认即可）` + 旋转 ▼，复用现有 Tailwind 按钮样式。
5. 高级块内 **保留英文术语**（工程师需要），每项补一行中文 helper（如 AdaLora "动态分配学习容量，通常无需开启"；FP16 "半精度加速，默认开启"）。
6. **栅格注意**：step-2 用 `grid grid-cols-1 md:grid-cols-2`，含若干 `col-span-2`。高级块需自成一个 grid 容器，或逐项复核 col-span，否则列错位。

**涉及文件**：`templates/index.html`、`static/js/finetune_panel.js`。后端无需改动。

**Out of scope（记录为后续）**：loss 图例/坐标轴的去黑话（`训练集 Loss`/`Eval Loss`/`Step`）。

---

## ③ 焦点/键盘管理（不做 aria-live）

**问题**：弹窗不锁/不还焦点、无 dialog 语义；沉浸式录音的方向键/Esc 是全局监听、绕过焦点管理，键盘用户进不去；登录框用 placeholder 当 label、不在 `<form>` 内、用户名框回车无反应。

**方案**（贴合运动障碍/键盘用户，不做屏幕阅读器播报）：

1. **弹窗焦点**：在 `app.js` 已集中处理 Enter/Esc 的地方（约 169-178）扩展：打开时把焦点移入弹窗、关闭时还原到触发元素；加最小焦点陷阱（Tab 在弹窗内循环）。顺带给弹窗加 `role="dialog"` + `aria-modal="true"`（实现正确焦点陷阱的标准配套，零额外成本）。作用域限定在 `dialogState.visible`，避免 tab 切换时抢焦点。
2. **登录页**：输入框包进 `<form>`、补 `<label>`（或 `aria-label`），用户名框回车提交。
3. **沉浸式录音灯箱**：方向键/Esc 与焦点管理打通，键盘可打开并操作；打开移焦进浮层、关闭还原。

**涉及文件**：`static/js/app.js`、`templates/index.html`、`static/js/inference_panel.js`（沉浸式键盘）

**Out of scope（用户已确认）**：aria-live 屏幕阅读器播报区、`.sr-only` 工具类、char-by-char 镜像播报区。

**风险**：焦点陷阱与 app.js 既有全局 Enter/Esc、keep-alive tab 组件交互，作用域不收紧会在 tab 过场时抢焦点。

---

## ② 停止训练（唯一需要后端）

**问题**：训练跑起来后**没有任何**停止手段（UI 或 API）。epochs 最高 200，loss 跑飞也只能干等几小时；单 GPU 的 423 锁会阻塞后面所有人。

**后端现状**：无停止端点、无取消标志，子进程句柄是函数局部（[finetune_controller.py:218](../../../finetune_controller.py)），外部够不着。
GPU 锁在单一 `finally` 释放（:285-292）；`log_generator` 仅凭 `_is_user_training()` 在 EOF 时发 `finished`（:476-479）。
约束：`GPU_STATE` 为进程内全局，**仅在 `uvicorn --workers 1` 下正确**（TODO_CODE_REVIEW.md P1-3），停止端点必须与持有句柄的进程同体。

**方案**：

后端 `finetune_controller.py`：
1. 模块全局 `TRAINING_PROC = {"proc": None, "pgid": None, "user": None, "cancelled": False}`。
2. 用 `asyncio.create_subprocess_exec(..., start_new_session=True)` 让子进程成为**进程组组长**（这样能连带 finetune.py 派生的 dataloader worker 孙进程一起杀，避免孤儿）。拉起后存 `proc` 与 `os.getpgid(proc.pid)`；`start_finetune` 抢锁时重置 `cancelled=False`、记 `user`。
3. 新增 `@router.post("/api/stop_finetune")`（`Depends(get_current_user)`）：抢 `GPU_LOCK` 校验 `status==TRAINING` 且 `current_user==caller`（复用 `_is_user_training`）；他人占用 → 423，未在训练 → 409/400。校验通过：`TRAINING_PROC["cancelled"]=True`，`os.killpg(pgid, SIGTERM)`；起一个 `asyncio.create_task`：睡 ~10s，若 `proc.returncode is None` 则 `os.killpg(pgid, SIGKILL)`（torch/CUDA 可能不响应 SIGTERM 的兜底，防止锁永久卡 TRAINING）。**不**在端点里重置 GPU_STATE。
4. `run_finetune_process`：`communicate()` 返回后，若 `cancelled` → 跳过 tflite/ggml 导出与入库（不 upsert 模型），并 `shutil.rmtree(output_dir)`（线程池里跑，复用 inference_controller 的模式），模型名即可复用。Linux 下 SSE 读取端的已打开文件句柄在 `rmtree`（unlink）后仍有效，读到 EOF 不崩。
5. 单一 `finally` 仍负责重置 GPU_STATE→IDLE（保住 P1-2 单点释放不变式）；`cancelled` 不在此清，留给下次 `start_finetune` 清。
6. `log_generator` 的两个 not-training 分支（:448 文件没生成就被杀 / :476 正常 EOF）：若 `TRAINING_PROC["cancelled"] and TRAINING_PROC["user"]==username` → 发 `{"status":"stopped"}` 否则 `finished`。

前端 `finetune_panel.js` + `index.html`：
7. `handleStopFinetune`（仿 `handleStartFinetune`）：先 `await dialog.confirm('确定要停止当前训练吗？已训练的进度将不会保存。', {variant:'danger', title:'停止训练', confirmText:'停止训练'})`；确认后 `apiFetch('/api/stop_finetune', {method:'POST'})`；423/409 用 `dialog.alert`（经 ⑤）。**不**在 JS 里强行 `isTraining=false`，交给 SSE 帧翻转（单一真相源）。
8. SSE `onmessage` 加 `stopped` 分支（仿 `error` 分支）：关闭 SSE、提示"训练已停止"、**不**弹发布弹窗。
9. index.html：在禁用的绿色启动按钮处加红色停止按钮 `v-if="isTraining"`，接 `handleStopFinetune`，复用现有 Tailwind 红色样式；`handleStopFinetune` 加入 setup return。
10. SSE `eventSource.onerror`（训练中途断连）：经 ⑤ 给一次 `friendlyNetworkError` 的可见提示，门控避免正常 `finished` 关闭时误报。

**新增 import**：`signal`、`shutil`（如未引入）；`os` 已在用。

**风险与验证局限**：
- 必须保持单 worker。
- 本机无法完整验证：kill 链路（尤其 dataloader 孙进程的 `killpg` 是否彻底、CUDA 显存是否及时释放、SIGKILL 兜底是否触发）需在**真实 GPU 主机**上跑一次真训练来 smoke-test。
- 不在端点重置 GPU_STATE；若日后有人在 `/api/stop_finetune` 里也重置，会重新引入 P1-2 双重释放竞态。
- 被否决的替代方案：协作式停止（让训练循环轮询 `should_training_stop`）对显存更友好，但需跨进程 IPC，复杂度高，故选信号/进程组方案。

**涉及文件**：`finetune_controller.py`、`static/js/finetune_panel.js`、`templates/index.html`

---

## 整体 Out of scope（明确不做）

- aria-live / 屏幕阅读器播报（用户已选"仅焦点/键盘"）
- 微调 loss 图的去黑话（记为后续）
- 后端给上传加体积限制以产出结构化 413（413 仅来自隧道，按状态码兜底即可）
- `delete_task` 改返回 404（纯前端错误暴露用不到；若日后改了，新 try/catch 会自动受益）
- 微调表单默认值任何改动（必须与 `FinetuneRequest` / argparse 锁步）

## 验证计划

1. `conda activate whisper`，`uvicorn main:app` 启动，浏览器（chromium-cli / Playwright）驱动。
2. ①：断网/造一个 5xx，确认每个写操作弹中文可操作提示、输入不丢、按钮禁用+转圈；TXT 导入显示条数。
3. ④：微调页默认只见 3 项简版字段，展开"高级设置"见全部参数 + helper，栅格不错位，提交 payload 不变。
4. ③：仅键盘可打开/操作弹窗与沉浸式灯箱，焦点开合正确还原，登录页回车可提交。
5. ⑤：各类错误均为中文、无 "Failed to fetch" 泄漏。
6. ②：**在 GPU 主机上**起一次真训练 → 点停止 → 确认进程组（含 worker）退出、显存释放、输出目录被清、模型名可复用、UI 显示"训练已停止"且不弹发布、GPU 锁回到 IDLE 可再次训练。

## 发布

按仓库约定（用户 2026-06-06 指示）：commit/push 直接推 `main`，不开功能分支。是否提交由用户在 review 后决定。
