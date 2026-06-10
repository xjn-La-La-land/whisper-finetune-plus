# 第一梯队前端修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复前端审计第一梯队的 5 个高影响问题：采集写操作错误吞掉、训练无法停止、弹窗/键盘焦点缺失、微调黑话、错误文案泄漏。

**Architecture:** 4 个纯前端修复 + 1 个需后端（停止训练）。新增 `static/js/errors.js` 作为统一错误文案源；其余复用现有 `dialog.js` / `apiFetch` 模式，无新依赖、无构建步骤、无 CDN。停止训练用「进程组 + SIGTERM→SIGKILL 兜底 + cancelled 标志让 SSE 发 stopped」，保住现有单点 GPU 锁释放不变式。

**Tech Stack:** Vue 3 全局版（无打包器）、Tailwind 预编译、ECharts、FastAPI、asyncio subprocess。

**无测试框架说明：** 本前端没有单元测试框架（无构建、Vue 全局版纯 JS）。因此每个 Task 的验证 = 启动应用 + 浏览器/curl 实测，而非 `pytest`/`jest`。验证统一用：
```bash
# 启动（端口若被占用换一个）
cd /home/xiejianan/whisper-finetune-plus && conda run -n whisper uvicorn main:app --host 127.0.0.1 --port 8011 > /tmp/uvicorn_test.log 2>&1 &
```
浏览器用 chromium-cli / Playwright 打开 `http://127.0.0.1:8011`，硬刷新（禁用缓存）以避开 ES module 缓存。

**缓存版本号约定（重要）：** 编辑某个 JS 模块后，必须在引用它的地方 bump `?v=` 查询串，否则浏览器可能加载旧模块。关键：**改 `app.js` 的内容（包括只改它内部 import 的 `?v=` 字符串）都必须在 `index.html` 同步 bump `app.js?v=`**，否则浏览器拿的还是旧 app.js（连带旧的面板版本）。本计划的 bump 表：
- `index.html` 的 `app.js?v=`：2.6 → 2.7(T2) → 2.8(T3) → 2.9(T4) → 2.10(T5) → 2.11(T7)
- `app.js` 内面板版本：`audio_collector.js` 1.2→1.3(T2)→1.4(T5)；`finetune_panel.js` 1.7→1.8(T4)→1.9(T7)；`inference_panel.js` 1.12→1.13(T3)
- 新模块 `errors.js` 内容只在 T1 创建、之后不变，一律以 `./errors.js?v=1` 引入
- 验证时 devtools 勾选 Disable cache 可绕过以上；`?v=` bump 是给生产用户的

---

## Task 1: 新增 errors.js 错误文案助手（FIX ⑤ 核心）

**Files:**
- Create: `static/js/errors.js`

- [ ] **Step 1: 创建 `static/js/errors.js`**

```javascript
// static/js/errors.js
// 统一的错误文案助手：把任何非 2xx Response / 抛出的异常映射成「可操作的中文」。
//
// 设计要点：
//   * 按 HTTP 状态码映射，不去解析 detail 的语义；
//   * 后端已写好中文的 detail（423/409/部分 400）原样回显，其余用固定中文兜底；
//   * 401 返回空串 —— 交给 api.js 的全局 whisper:unauthorized 处理，调用方据此跳过弹窗；
//   * 网络错误绝不回显 err.message（那是 "Failed to fetch" 泄漏处）。

// 含至少一个 CJK 字符即认为可直接展示给中文用户。
function isChinese(s) {
    return typeof s === 'string' && /[一-鿿]/.test(s);
}

// 从已读出的 body 里取后端文案（detail 优先，其次 message —— 登录/注册接口用 message）。
function pickDetail(body) {
    if (!body || typeof body !== 'object') return '';
    const d = body.detail != null ? body.detail : body.message;
    return typeof d === 'string' ? d : '';
}

/**
 * 把一个非 2xx 的 Response 映射成可操作的中文提示。
 * @param {Response} res
 * @param {string} [fallback]
 * @returns {Promise<string>} 提示文案；401 返回 ''（交全局处理，调用方应据此跳过弹窗）
 */
export async function friendlyHttpError(res, fallback = '操作失败，请稍后重试。') {
    const status = res ? res.status : 0;

    // 401：会话过期，已由 api.js 广播 whisper:unauthorized 全局接管，这里不重复提示。
    if (status === 401) return '';
    // 413：请求体过大。应用层无体积限制，413 只来自前置隧道/代理（body 多为 HTML），按状态码兜底。
    if (status === 413) return '文件太大，无法上传。请录制更短的音频后重试。';

    // 读 body（413/SSE/空体可能不是 JSON；clone 以免与调用方的 res.json() 抢 body）
    let body = null;
    try { body = await res.clone().json(); } catch (e) { body = null; }
    const detail = pickDetail(body);

    // 423 GPU 占用 / 409 冲突：后端写的就是可操作的中文，原样回显。
    if (status === 423 || status === 409) return isChinese(detail) ? detail : fallback;
    // 400：多为校验类，后端中文 detail 直接给（如模型名规则），否则兜底。
    if (status === 400) return isChinese(detail) ? detail : '请求无法处理，请检查输入后重试。';
    // 422：FastAPI 参数校验，detail 常是结构化/英文，避免泄漏，给固定中文。
    if (status === 422) return '提交的内容不符合要求，请检查后重试。';
    // 5xx：服务端错误。
    if (status >= 500) return '服务器开小差了，请稍后重试；若多次失败请联系管理员。';

    return isChinese(detail) ? detail : fallback;
}

/**
 * 把 fetch reject / 其它异常映射成固定中文网络提示。绝不回显 err.message。
 * @returns {string}
 */
export function friendlyNetworkError() {
    return '无法连接服务器，请检查网络后重试。';
}
```

- [ ] **Step 2: 启动应用并验证模块语法**

Run（启动见顶部）。浏览器控制台执行：
```js
const m = await import('/static/js/errors.js?v=1');
console.log(await m.friendlyHttpError({status:413}), '|', await m.friendlyHttpError({status:401}), '|', m.friendlyNetworkError());
```
Expected: 打印 `文件太大，无法上传。请录制更短的音频后重试。 |  | 无法连接服务器，请检查网络后重试。`（401 为空串），且无语法报错。

- [ ] **Step 3: Commit**

```bash
cd /home/xiejianan/whisper-finetune-plus
git add static/js/errors.js docs/superpowers/plans/2026-06-10-tier1-frontend-fixes.md docs/superpowers/specs/2026-06-10-tier1-frontend-fixes-design.md
git commit -m "feat(frontend): add errors.js friendly error-message helper (tier-1 fix 5)"
```

---

## Task 2: 采集页写操作错误暴露 + loading 态（FIX ①，复用 errors.js）

**Files:**
- Modify: `static/js/audio_collector.js`（import、in-flight refs、5 个写函数、return）
- Modify: `templates/index.html`（5 个按钮加 `:disabled`/spinner）

- [ ] **Step 1: 在 `audio_collector.js` 顶部加 errors.js import**

把 `static/js/audio_collector.js:3` 这一行：
```javascript
import { apiFetch, sseUrl } from './api.js?v=1.2';
```
改为（其下新增一行）：
```javascript
import { apiFetch, sseUrl } from './api.js?v=1.2';
import { friendlyHttpError, friendlyNetworkError } from './errors.js?v=1';
```

- [ ] **Step 2: 新增 in-flight 状态 ref**

在 `audio_collector.js` 的 `const processingTaskId = ref(null);`（第 17 行）之后新增：
```javascript
        // --- 写操作 in-flight 态（防重复点击 + 按钮 spinner/禁用）---
        const busyImport = ref(false);
        const busyAdd = ref(false);
        const savingTaskId = ref(null);
        const deletingTaskId = ref(null);
        const clearing = ref(false);
```

- [ ] **Step 3: 重写 uploadTxt（第 42-50 行）**

整段替换为：
```javascript
        const uploadTxt = async () => {
            const file = txtInput.value.files[0];
            if (!file) return dialog.alert("请先选择一个 TXT 文件！", { variant: 'warning' });
            if (busyImport.value) return;
            busyImport.value = true;
            const formData = new FormData();
            formData.append("file", file);
            try {
                const res = await apiFetch('/api/upload_txt', { method: 'POST', body: formData });
                if (!res.ok) throw new Error(await friendlyHttpError(res));
                const data = await res.json().catch(() => ({}));
                txtInput.value.value = "";            // 成功后才清空文件选择
                if (data.message) dialog.alert(data.message, { variant: 'success' });
            } catch (e) {
                // fetch 网络层 reject 抛 TypeError；我们自己 throw 的是 Error
                const msg = (e instanceof TypeError) ? friendlyNetworkError() : e.message;
                if (msg) dialog.alert(msg, { variant: 'danger' });
            } finally {
                busyImport.value = false;
                await fetchTasks();
            }
        };
```

- [ ] **Step 4: 重写 addTask（第 52-61 行）**

整段替换为：
```javascript
        const addTask = async () => {
            const text = newTaskText.value.trim();
            if (!text) return;
            if (busyAdd.value) return;
            busyAdd.value = true;
            try {
                const res = await apiFetch('/api/task', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                if (!res.ok) throw new Error(await friendlyHttpError(res));
                newTaskText.value = "";               // 成功后才清空输入，失败保留用户已打的字
            } catch (e) {
                const msg = (e instanceof TypeError) ? friendlyNetworkError() : e.message;
                if (msg) dialog.alert(msg, { variant: 'danger' });
            } finally {
                busyAdd.value = false;
                await fetchTasks();
            }
        };
```

- [ ] **Step 5: 重写 saveEdit 的网络部分（第 82-88 行）**

把 `saveEdit` 里 confirm 之后的这段：
```javascript
            await apiFetch(`/api/task/${taskId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: newText })
            });
            editingTaskId.value = null;
            fetchTasks();
        };
```
替换为（in-flight ref 在 confirm 解析后才置位，避免用户取消 confirm 时卡死）：
```javascript
            if (savingTaskId.value) return;
            savingTaskId.value = taskId;
            try {
                const res = await apiFetch(`/api/task/${taskId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: newText })
                });
                if (!res.ok) throw new Error(await friendlyHttpError(res));
                editingTaskId.value = null;           // 成功后才关闭编辑框
            } catch (e) {
                const msg = (e instanceof TypeError) ? friendlyNetworkError() : e.message;
                if (msg) dialog.alert(msg, { variant: 'danger' });
            } finally {
                savingTaskId.value = null;
                await fetchTasks();
            }
        };
```

- [ ] **Step 6: 重写 deleteTask（第 91-99 行）**

整段替换为：
```javascript
        const deleteTask = async (taskId) => {
            const ok = await dialog.confirm(
                "确定要删除这条录音任务吗？已录制的音频也会被永久删除！",
                { variant: 'danger', confirmText: '删除', cancelText: '取消' }
            );
            if (!ok) return;
            if (deletingTaskId.value) return;
            deletingTaskId.value = taskId;
            try {
                const res = await apiFetch(`/api/task/${taskId}`, { method: 'DELETE' });
                if (!res.ok) throw new Error(await friendlyHttpError(res));
            } catch (e) {
                const msg = (e instanceof TypeError) ? friendlyNetworkError() : e.message;
                if (msg) dialog.alert(msg, { variant: 'danger' });
            } finally {
                deletingTaskId.value = null;
                await fetchTasks();
            }
        };
```

- [ ] **Step 7: 重写 clearAllTasks（第 101-109 行）**

整段替换为：
```javascript
        const clearAllTasks = async () => {
            const ok = await dialog.confirm(
                `确定要清空代号 [${props.currentUser}] 的所有任务和音频吗？此操作无法撤销！`,
                { variant: 'danger', confirmText: '全部清空', cancelText: '取消' }
            );
            if (!ok) return;
            if (clearing.value) return;
            clearing.value = true;
            try {
                const res = await apiFetch('/api/tasks', { method: 'DELETE' });
                if (!res.ok) throw new Error(await friendlyHttpError(res));
            } catch (e) {
                const msg = (e instanceof TypeError) ? friendlyNetworkError() : e.message;
                if (msg) dialog.alert(msg, { variant: 'danger' });
            } finally {
                clearing.value = false;
                await fetchTasks();
            }
        };
```

- [ ] **Step 8: 把新 ref 加入 setup 的 return（第 198-204 行）**

把 return 对象第一行：
```javascript
            tasks, txtInput, uploadTxt, startRecording, stopRecording, recordingTaskId, processingTaskId,
```
改为：
```javascript
            tasks, txtInput, uploadTxt, startRecording, stopRecording, recordingTaskId, processingTaskId,
            busyImport, busyAdd, savingTaskId, deletingTaskId, clearing,
```

- [ ] **Step 9: 模板加 `:disabled` + spinner（`templates/index.html`）**

通用 spinner 片段（按钮内用，复用文件已有的 animate-spin SVG）：
```html
<svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
```

9a. **清空全部**（第 597 行）：在 `<button v-if="tasks.length > 0" @click="clearAllTasks"` 上加 `:disabled="clearing"` 和 `disabled:opacity-50 disabled:cursor-not-allowed`（追加到 class 串）。

9b. **导入**（第 612 行）：整个按钮替换为：
```html
                        <button @click="uploadTxt" :disabled="busyImport" class="bg-blue-500 hover:bg-blue-600 disabled:opacity-60 disabled:cursor-not-allowed text-white px-5 py-2.5 rounded-xl font-bold transition whitespace-nowrap ml-1 shadow-sm flex items-center gap-1.5">
                            <svg v-if="busyImport" class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                            <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path></svg>
                            {{ busyImport ? '导入中…' : '导入' }}
                        </button>
```

9c. **添加**（第 616 行）：整个按钮替换为：
```html
                        <button @click="addTask" :disabled="busyAdd" class="bg-green-500 hover:bg-green-600 disabled:opacity-60 disabled:cursor-not-allowed text-white px-5 py-2.5 rounded-xl font-bold transition whitespace-nowrap shadow-sm flex items-center gap-1.5">
                            <svg v-if="busyAdd" class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                            <svg v-else class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path></svg>
                            {{ busyAdd ? '添加中…' : '添加' }}
                        </button>
```

9d. **保存（内联编辑）**（第 629 行）：在 `<button @click="saveEdit(task.id)"` 上加 `:disabled="savingTaskId === task.id"` 和 `disabled:opacity-60` class，并把按钮文字 `保存` 改为 `{{ savingTaskId === task.id ? '保存中…' : '保存' }}`。

9e. **删除（行内）**（第 638 行）：在 `<button @click="deleteTask(task.id)"` 上加 `:disabled="deletingTaskId === task.id"`。

- [ ] **Step 10: bump 版本**

- `static/js/app.js:4`：`import AudioCollector from './audio_collector.js?v=1.2';` → `?v=1.3`
- 因上一行改了 app.js 内容，`templates/index.html:1305`：`app.js?v=2.6` → `?v=2.7`

- [ ] **Step 11: 浏览器验证**

启动应用并硬刷新。验证：
1. 正常：手动添加一句 → 列表出现、输入框清空、无弹窗。导入合法 TXT → 弹绿色「成功导入 N 条」。
2. 失败：临时把后端关掉（或浏览器 devtools 设 offline），点「添加」→ 弹红色「无法连接服务器…」，**输入框内容保留**；按钮在请求中禁用且转圈。
3. 删除/清空：请求期间按钮禁用，不可连点。

Expected: 全部符合；控制台无未捕获异常。

- [ ] **Step 12: Commit**

```bash
cd /home/xiejianan/whisper-finetune-plus
git add static/js/audio_collector.js static/js/app.js templates/index.html
git commit -m "feat(collector): surface write errors + in-flight disable/spinner (tier-1 fix 1)"
```

---

## Task 3: 其余错误文案路由经 errors.js（FIX ⑤ 收尾：inference + app 登录）

> finetune_panel 的错误/ SSE 文案在 Task 6/7 一并改（与停止逻辑同处一文件，避免重复编辑）。

**Files:**
- Modify: `static/js/inference_panel.js`（import + 2 处 typeStatus 错误）
- Modify: `static/js/app.js`（import + 登录网络错误）

- [ ] **Step 1: inference_panel.js 加 import**

`static/js/inference_panel.js:3`：
```javascript
import { apiFetch } from './api.js?v=1.2';
```
其下新增：
```javascript
import { friendlyHttpError, friendlyNetworkError } from './errors.js?v=1';
```

- [ ] **Step 2: 服务端识别的错误文案（第 272、280 行）**

第 272 行：
```javascript
                if (!res.ok) { const err = await res.json(); await typeStatus(`❌ 错误: ${err.detail}`, 'error'); return; }
```
改为：
```javascript
                if (!res.ok) { const m = await friendlyHttpError(res); if (m) await typeStatus(`❌ ${m}`, 'error'); return; }
```
第 280 行：
```javascript
                await typeStatus(`❌ 网络请求失败: ${err.message}`, 'error');
```
改为：
```javascript
                await typeStatus(`❌ ${friendlyNetworkError()}`, 'error');
```

- [ ] **Step 3: app.js 加 import 并改登录网络错误**

`static/js/app.js:8`：
```javascript
import { apiFetch, setToken, clearToken, getToken, getStoredUsername } from './api.js?v=1.2';
```
其下新增：
```javascript
import { friendlyNetworkError } from './errors.js?v=1';
```
第 118 行：
```javascript
                await dialog.alert("网络请求失败，请检查后端服务是否启动！", { variant: 'danger' });
```
改为：
```javascript
                await dialog.alert(friendlyNetworkError(), { variant: 'danger' });
```
（保留第 103 行读 `errorData.message` 不变 —— 登录/注册接口刻意用 message 做防枚举提示。）

- [ ] **Step 4: bump 版本**

- `static/js/app.js:6`：`inference_panel.js?v=1.12` → `?v=1.13`
- `templates/index.html:1305`：`app.js?v=2.7` → `?v=2.8`

- [ ] **Step 5: 浏览器验证**

硬刷新。devtools 设 offline 后在「语音识别」上传一个音频 → 终端区显示 `❌ 无法连接服务器，请检查网络后重试。`（无 "Failed to fetch"）。退出登录页 offline 点登录 → 弹「无法连接服务器…」。

- [ ] **Step 6: Commit**

```bash
cd /home/xiejianan/whisper-finetune-plus
git add static/js/inference_panel.js static/js/app.js templates/index.html
git commit -m "feat(inference,auth): route errors through errors.js, no raw English leak (tier-1 fix 5)"
```

---

## Task 4: 微调表单黑话渐进式披露（FIX ④）

**Files:**
- Modify: `static/js/finetune_panel.js`（showAdvanced ref + return）
- Modify: `templates/index.html`（表单 step-2 结构）

- [ ] **Step 1: finetune_panel.js 加 showAdvanced**

在 `static/js/finetune_panel.js` 的 `const chartError = ref("");`（第 16 行）之后新增：
```javascript
        const showAdvanced = ref(false);   // 高级参数默认收起：家长保持默认即可，工程师可展开
```
并在 setup return 对象里（第 905 行 `chartError,` 之后）加一行：
```javascript
            showAdvanced,
```

- [ ] **Step 2: 简版字段微调文案（`templates/index.html`）**

- 模型名称括注（第 779 行）`(显示在推理下拉列表)` 改为 `给这个专属模型起个名字`。
- 训练轮数 helper（第 868 行）`让 AI 把全部录音反复听多少遍。默认 20 遍。` 改为 `AI 要把录音反复听多少遍，越多学得越细但越久。默认 20。`

- [ ] **Step 3: 在「训练轮数」块之后插入披露开关 + 高级块开头**

在第 869 行（训练轮数块的 `</div>`）之后、第 871 行（学习步长块开头 `<div>`）之前，插入：
```html
                    <!-- 高级设置披露开关：默认收起；展开后是原全部进阶参数 -->
                    <div class="col-span-1 md:col-span-2">
                        <button type="button" @click="showAdvanced = !showAdvanced"
                                class="inline-flex items-center gap-2 text-sm font-bold text-blue-600 hover:text-blue-700 transition">
                            <span>⚙️ 高级设置（可选，保持默认即可）</span>
                            <svg class="w-4 h-4 transition-transform" :class="showAdvanced ? 'rotate-180' : ''" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                        </button>
                    </div>
                    <div v-show="showAdvanced" class="col-span-1 md:col-span-2">
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
```

- [ ] **Step 4: 在音频时长块之后闭合高级块**

在第 957 行（音频时长块的 `</div>`）之后、第 958 行（grid 的 `</div>`）之前，插入：
```html
                        </div>
                    </div>
```
（这两层分别闭合 Step 3 里新开的内层 `grid` 和外层 `v-show` 容器。学习步长/批大小/梯度累计/三个 checkbox/音频时长 5 个块现在都落在高级块内，无需移动它们本身。）

- [ ] **Step 5: 给高级块内 3 个 checkbox 补中文 helper**

在第 904-923 行的 checkbox 容器里，每个 `<label>` 的 `</label>` 之前各加一行说明（紧跟在对应 label 内容下方）。具体：在 AdaLora 那个 `<input ...>` 之后、`</label>` 之前不便插入；改为在三个 label 共同的容器（第 904 行 `<div class="flex flex-col justify-center gap-3 ...">`）末尾、`</div>`（第 923 行）之前，插入一行总说明：
```html
                        <p class="text-[11px] text-gray-400 mt-1">🧠 AdaLora 动态分配学习容量、📦 8-bit 省显存、⚡ FP16 半精度加速——通常保持默认即可。</p>
```

- [ ] **Step 6: bump 版本**

- `static/js/app.js:5`：`finetune_panel.js?v=1.7` → `?v=1.8`
- `templates/index.html:1305`：`app.js?v=2.8` → `?v=2.9`

- [ ] **Step 7: 浏览器验证**

硬刷新进「模型微调」。验证：
1. 默认只见 模型名称 / 基础模型 / 训练轮数 + 「⚙️ 高级设置」按钮；学习步长/批大小/梯度累计/三 checkbox/音频时长 **隐藏**。
2. 点「高级设置」→ 全部展开，▼ 旋转；栅格两列不错位；双滑块仍能拖、值保留。
3. 在控制台 `document.querySelector('[v-cloak]')` 无影响；切到别的 tab 再回来，展开状态/参数值保持（v-show 不卸载）。

- [ ] **Step 8: Commit**

```bash
cd /home/xiejianan/whisper-finetune-plus
git add static/js/finetune_panel.js static/js/app.js templates/index.html
git commit -m "feat(finetune): progressive-disclose advanced ML params behind 高级设置 (tier-1 fix 4)"
```

---

## Task 5: 焦点/键盘管理（FIX ③：弹窗 + 登录 + 沉浸式）

**Files:**
- Modify: `static/js/app.js`（弹窗焦点移入/还原 + Tab 陷阱；import watch）
- Modify: `templates/index.html`（弹窗 ref/role、登录 form/label、沉浸式行可键盘打开）
- Modify: `static/js/audio_collector.js`（沉浸式焦点移入/还原；import watch）

- [ ] **Step 1: app.js 引入 watch，新增弹窗焦点逻辑**

`static/js/app.js:1`：
```javascript
const { createApp, ref, onMounted, watchEffect, nextTick } = Vue;
```
改为：
```javascript
const { createApp, ref, onMounted, watchEffect, nextTick, watch } = Vue;
```
在 setup 内 `const passwordInput = ref("");`（第 15 行）之后新增：
```javascript
        // --- 弹窗焦点管理（无障碍：键盘/运动障碍用户）---
        const dialogRoot = ref(null);          // 绑定到弹窗卡片
        let lastFocusedBeforeDialog = null;
        watch(() => dialog.dialogState.visible, async (visible) => {
            if (visible) {
                lastFocusedBeforeDialog = document.activeElement;
                await nextTick();
                const el = dialogRoot.value;
                if (el) {
                    const target = el.querySelector('[data-dialog-confirm]')
                        || el.querySelector('button, input, select, textarea, [href]');
                    if (target) target.focus();
                }
            } else if (lastFocusedBeforeDialog && lastFocusedBeforeDialog.focus) {
                lastFocusedBeforeDialog.focus();
                lastFocusedBeforeDialog = null;
            }
        });
```
并把 `dialogRoot` 加入 setup return（第 194 行 `dialogState: dialog.dialogState,` 之前）：
```javascript
            dialogRoot,
```

- [ ] **Step 2: 扩展全局 keydown 处理为 Tab 焦点陷阱（app.js 第 169-178 行）**

整段替换为：
```javascript
            // 全局键盘支持：dialog 显示时 Enter 确认 / Esc 取消 / Tab 在弹窗内循环
            window.addEventListener('keydown', (e) => {
                if (!dialog.dialogState.visible) return;
                if (e.key === 'Enter') {
                    e.preventDefault();
                    dialog.handleConfirm();
                } else if (e.key === 'Escape') {
                    e.preventDefault();
                    dialog.handleCancel();
                } else if (e.key === 'Tab') {
                    const el = dialogRoot.value;
                    if (!el) return;
                    const f = Array.from(el.querySelectorAll('button, input, select, textarea, [href], [tabindex]:not([tabindex="-1"])'))
                        .filter(n => !n.disabled && n.offsetParent !== null);
                    if (!f.length) return;
                    const first = f[0], last = f[f.length - 1];
                    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
                    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
                }
            });
```

- [ ] **Step 3: 弹窗卡片加 ref + dialog 语义（`templates/index.html` 第 420 行）**

把第 420 行：
```html
                    <div class="bg-white rounded-3xl shadow-2xl max-w-md w-full overflow-hidden">
```
改为：
```html
                    <div ref="dialogRoot" role="dialog" aria-modal="true" :aria-label="dialogState.title"
                         class="bg-white rounded-3xl shadow-2xl max-w-md w-full overflow-hidden">
```
并在确认按钮（第 458 行 `<button @click="dialogConfirm"`）上加属性 `data-dialog-confirm`。

- [ ] **Step 4: 登录表单化 + label（`templates/index.html` 第 508-527 行）**

把第 508 行 `<div class="login-field mb-4">` 之前插入 `<form @submit.prevent="handleAuth('login')">`；在第 527 行（按钮容器 `</div>`）之后插入 `</form>`。
- 用户名 input（第 510-513 行）加属性 `aria-label="代号"`。
- 密码 input（第 517-522 行）加属性 `aria-label="密码"`，并**删除**其 `@keyup.enter="handleAuth('login')"`（表单 submit 已覆盖，避免双触发）。
- 登录按钮（第 525 行）把 `<button @click="handleAuth('login')"` 改为 `<button type="submit"`（移除 @click，提交由 form 触发）。
- 注册按钮（第 526 行）加 `type="button"`（确保它不触发表单 submit）。

- [ ] **Step 5: audio_collector.js 沉浸式焦点移入/还原**

`static/js/audio_collector.js:1`：
```javascript
const { ref, computed, onMounted, onUnmounted, nextTick } = Vue;
```
改为：
```javascript
const { ref, computed, onMounted, onUnmounted, nextTick, watch } = Vue;
```
在 `const closeFocusMode = () => { focusedTaskIndex.value = null; };`（第 170 行）之后新增：
```javascript
        // 沉浸式开合时的焦点管理（键盘/运动障碍用户）
        const focusModeRoot = ref(null);
        let lastFocusedBeforeFocus = null;
        watch(focusedTaskIndex, async (idx, prev) => {
            if (idx !== null && prev === null) {
                lastFocusedBeforeFocus = document.activeElement;
                await nextTick();
                const el = focusModeRoot.value;
                const btn = el && el.querySelector('button');
                if (btn) btn.focus();
            } else if (idx === null && prev !== null) {
                if (lastFocusedBeforeFocus && lastFocusedBeforeFocus.focus) lastFocusedBeforeFocus.focus();
                lastFocusedBeforeFocus = null;
            }
        });
```
并把 `focusModeRoot` 加入 return（第 202 行 `focusedTaskIndex, focusedTask, openFocusMode, ...` 那一行末尾追加）：
```javascript
            focusedTaskIndex, focusedTask, openFocusMode, closeFocusMode, prevFocusTask, nextFocusTask, focusModeRoot,
```

- [ ] **Step 6: 沉浸式容器加 ref + 行可键盘打开（`templates/index.html`）**

- 第 666 行 `<div v-if="focusedTaskIndex !== null && focusedTask" class="fixed inset-0 z-[100] ...">` 加 `ref="focusModeRoot"`。
- 任务行（第 622 行 `<div v-for="(task, index) in tasks" ... @click="openFocusMode(index)" class="...">`）追加属性使其可键盘聚焦并回车打开：`tabindex="0" @keyup.enter="openFocusMode(index)"`。

- [ ] **Step 7: bump 版本**

- `static/js/app.js:4`：`audio_collector.js?v=1.3` → `?v=1.4`
- `templates/index.html:1305`：`app.js?v=2.9` → `?v=2.10`

- [ ] **Step 8: 仅用键盘验证**

硬刷新。全程不用鼠标：
1. 触发任意弹窗（如点登录页"登录"留空代号）→ 焦点自动落在确认按钮；Tab 在按钮间循环不跑到背后；Esc 关闭后焦点回到原触发处。
2. 登录页：Tab 到代号→密码→按钮；在任一输入框按 Enter 可登录（仅一次请求）。
3. 采集页：Tab 到某任务行按 Enter → 打开沉浸式，焦点落在录音按钮；← →切题、Esc 关闭后焦点回到原任务行。

- [ ] **Step 9: Commit**

```bash
cd /home/xiejianan/whisper-finetune-plus
git add static/js/app.js static/js/audio_collector.js templates/index.html
git commit -m "feat(a11y): dialog focus trap/restore, login form+labels, keyboard-open focus mode (tier-1 fix 3)"
```

---

## Task 6: 停止训练 — 后端（FIX ②）

**Files:**
- Modify: `finetune_controller.py`（模块全局、子进程进程组、stop 端点、cancelled 处理、log_generator）

> 先确认顶部 import：需要 `os`、`signal`、`shutil`、`asyncio`。`os`/`asyncio` 已在用；缺 `signal`/`shutil` 则在 import 区补上。下面所有行号以本计划撰写时的 finetune_controller.py 为准，实施时按上下文锚点定位。

- [ ] **Step 1: 新增模块全局 TRAINING_PROC**

在 `finetune_controller.py` 顶部 import 之后（或 router 定义附近的模块级位置）新增：
```python
# 正在运行的训练子进程登记表（仅 uvicorn --workers 1 下有效，与 GPU_STATE 同约束，见 P1-3）。
# cancelled 在下一次 start_finetune 抢锁时复位；保留到那时，是为了让 log_generator 在
# GPU_STATE 复位为 IDLE 之后仍能判断「这是用户主动停止」从而发 stopped 而非 finished。
TRAINING_PROC = {"proc": None, "pgid": None, "cancelled": False}
```
若顶部缺少：`import signal` / `import shutil`，一并补上。

- [ ] **Step 2: 子进程自成进程组并登记句柄（run_finetune_process，第 218 行）**

把：
```python
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
```
改为：
```python
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,   # 自成进程组；停止时可连带 dataloader worker 一起 killpg
            )
            # 登记句柄/进程组，供 /api/stop_finetune 终止
            TRAINING_PROC["proc"] = process
            try:
                TRAINING_PROC["pgid"] = os.getpgid(process.pid)
            except Exception:
                TRAINING_PROC["pgid"] = None
```

- [ ] **Step 3: 取消时跳过导出并清理输出目录（run_finetune_process，第 227-279 行）**

把第 227 行 `if process.returncode == 0:` 改为分三路（在它前面加 cancelled 分支，原成功块降级为 `elif`）：
```python
            if TRAINING_PROC["cancelled"]:
                print(f"[{username}] 训练已被用户停止，清理半成品输出目录：{output_dir}")
                await asyncio.to_thread(shutil.rmtree, output_dir, ignore_errors=True)
            elif process.returncode == 0:
                print(f"[{username}] 微调成功！")
```
（即：只把原来的 `if process.returncode == 0:` 这一行替换为上面的 cancelled 分支 + `elif process.returncode == 0:`，其后的成功大块、`else: print(报错)` 全部保持不变。`output_dir` 是本函数已有的局部变量，见第 181/186/229 行引用。）

- [ ] **Step 4: start_finetune 抢锁时复位 TRAINING_PROC（第 354-357 行）**

把：
```python
        GPU_STATE["status"] = GPUStatus.TRAINING
        GPU_STATE["current_user"] = current_user
        # 记下正在训练的模型名，方便前端刷新后从 /api/gpu_status 恢复 SSE 连接
        GPU_STATE["current_model_name"] = req.model_name
```
改为（仍在 `async with GPU_LOCK:` 内）：
```python
        GPU_STATE["status"] = GPUStatus.TRAINING
        GPU_STATE["current_user"] = current_user
        # 记下正在训练的模型名，方便前端刷新后从 /api/gpu_status 恢复 SSE 连接
        GPU_STATE["current_model_name"] = req.model_name
        # 复位上一轮的停止标志与句柄（句柄稍后在 run_finetune_process 里登记）
        TRAINING_PROC["cancelled"] = False
        TRAINING_PROC["proc"] = None
        TRAINING_PROC["pgid"] = None
```

- [ ] **Step 5: 新增 /api/stop_finetune 端点**

在 `start_finetune` 函数之后新增（owner-only；SIGTERM 后 10s 宽限再 SIGKILL；不在此复位 GPU_STATE，交给 run_finetune_process 的 finally）：
```python
@router.post("/api/stop_finetune")
async def stop_finetune(current_user: str = Depends(get_current_user)):
    # 原子校验：必须正在训练，且只有发起者本人可停
    async with GPU_LOCK:
        if GPU_STATE["status"] != GPUStatus.TRAINING:
            raise HTTPException(status_code=409, detail="当前没有正在进行的训练任务。")
        if GPU_STATE["current_user"] != current_user:
            raise HTTPException(
                status_code=423,
                detail=f"当前训练由用户 [{GPU_STATE['current_user']}] 发起，只有发起者本人可以停止。"
            )
        TRAINING_PROC["cancelled"] = True
        proc = TRAINING_PROC.get("proc")
        pgid = TRAINING_PROC.get("pgid")

    # 锁外执行 kill，避免阻塞只读状态查询
    if proc is None or proc.returncode is not None:
        # 进程已退出（或刚启动尚未登记句柄），交由 run_finetune_process 的 finally 复位
        return {"message": "训练已停止。"}

    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        pass

    # SIGKILL 兜底：宽限 10s，若仍未退出（torch/CUDA 卡住）则强杀整组，防止锁永久卡 TRAINING
    async def _force_kill_after_grace(p, gid):
        try:
            await asyncio.sleep(10)
            if p.returncode is None:
                if gid is not None:
                    os.killpg(gid, signal.SIGKILL)
                else:
                    p.kill()
        except ProcessLookupError:
            pass
        except Exception as ex:
            print(f"[stop_finetune] 强杀兜底异常: {ex}")
    asyncio.create_task(_force_kill_after_grace(proc, pgid))

    return {"message": "已发送停止信号，训练即将结束。"}
```

- [ ] **Step 6: log_generator 在取消时发 stopped（第 446-479 行）**

6a. 文件还没生成就被杀的分支（第 448-450 行）：
```python
        if not _is_user_training(username):
            yield f"data: {{\"status\": \"error\", \"message\": \"训练未启动或提前异常终止\"}}\n\n"
            return
```
改为：
```python
        if not _is_user_training(username):
            if TRAINING_PROC.get("cancelled"):
                yield f"data: {{\"status\": \"stopped\", \"message\": \"训练已被用户停止\"}}\n\n"
            else:
                yield f"data: {{\"status\": \"error\", \"message\": \"训练未启动或提前异常终止\"}}\n\n"
            return
```
6b. 正常 EOF 后训练结束的分支（第 476-479 行）：
```python
                if not _is_user_training(username):
                    # 训练已经结束，发送完成信号跳出循环
                    yield f"data: {{\"status\": \"finished\", \"message\": \"训练已完成\"}}\n\n"
                    break
```
改为：
```python
                if not _is_user_training(username):
                    if TRAINING_PROC.get("cancelled"):
                        yield f"data: {{\"status\": \"stopped\", \"message\": \"训练已被用户停止\"}}\n\n"
                    else:
                        yield f"data: {{\"status\": \"finished\", \"message\": \"训练已完成\"}}\n\n"
                    break
```

- [ ] **Step 7: 端点存在性 + 鉴权冒烟（无需 GPU）**

启动应用。先验证未训练时的行为：
```bash
# 取一个有效 token（登录后从浏览器 localStorage.whisper_token 复制），设为 $TOK
curl -s -X POST http://127.0.0.1:8011/api/stop_finetune -H "Authorization: Bearer $TOK"
```
Expected: 返回 `{"detail":"当前没有正在进行的训练任务。"}`（HTTP 409）。未带 token → 401。

- [ ] **Step 8: Commit**

```bash
cd /home/xiejianan/whisper-finetune-plus
git add finetune_controller.py
git commit -m "feat(finetune): add /api/stop_finetune (owner-only, killpg SIGTERM->SIGKILL, SSE 'stopped', rmtree partial) (tier-1 fix 2 backend)"
```

---

## Task 7: 停止训练 — 前端（FIX ②）+ SSE 文案收尾（FIX ⑤）

**Files:**
- Modify: `static/js/finetune_panel.js`（import errors、isStopping、handleStopFinetune、SSE stopped 分支、onerror 提示、各 catch 文案、return）
- Modify: `templates/index.html`（红色停止按钮）

- [ ] **Step 1: finetune_panel.js 加 import**

`static/js/finetune_panel.js:3`：
```javascript
import { apiFetch, sseUrl } from './api.js?v=1.2';
```
其下新增：
```javascript
import { friendlyHttpError, friendlyNetworkError } from './errors.js?v=1';
```

- [ ] **Step 2: 新增 isStopping 与 SSE 错误一次性提示标志**

在 `const isTraining = ref(false);`（第 14 行）之后新增：
```javascript
        const isStopping = ref(false);
```
在 `let eventSource = null;`（第 134 行）之后新增：
```javascript
        let sseErrorNotified = false;   // 避免 EventSource 自动重连时反复弹「连接中断」
```

- [ ] **Step 3: startSSE 里复位 sseErrorNotified（第 562 行附近）**

在 `eventSource = new EventSource(sseUrl('/api/train_stream', { model_name: modelName }));`（第 562 行）之前一行加：
```javascript
            sseErrorNotified = false;
```

- [ ] **Step 4: onmessage 里成功收到数据时复位标志 + 新增 stopped 分支**

在 `eventSource.onmessage = (event) => {` 内、`try { data = JSON.parse(event.data); }` 解析成功之后（第 571 行 `}` 之后）加一行：
```javascript
                sseErrorNotified = false;
```
然后在 `if (data.status === 'finished') {` 分支（第 573 行）**之前**插入新分支：
```javascript
                if (data.status === 'stopped') {
                    isTraining.value = false;
                    eventSource.close();
                    eventSource = null;
                    dialog.alert('训练已停止，本次进度未保存为可用模型。', { variant: 'info', title: '已停止' });
                    loadUserModels();
                    return;
                }
```

- [ ] **Step 5: 替换 onerror（第 615-617 行）为带门控的可见提示**

把：
```javascript
            eventSource.onerror = (err) => {
                console.error("[SSE] error =", err);
            };
```
改为：
```javascript
            eventSource.onerror = (err) => {
                console.error("[SSE] error =", err);
                // 正常 finished/stopped 已先 close()；仅当我们仍认为在训练且尚未提示过时，提醒一次
                if (isTraining.value && !sseErrorNotified) {
                    sseErrorNotified = true;
                    dialog.alert(
                        friendlyNetworkError() + '\n训练可能仍在后台继续，可稍后切到别的标签页再返回以重新连接查看进度。',
                        { variant: 'warning', title: '连接中断' }
                    );
                }
            };
```

- [ ] **Step 6: 新增 handleStopFinetune**

在 `handleStartFinetune` 函数（第 715 行 `};`）之后新增：
```javascript
        // --- 动作：停止当前训练 ---
        const handleStopFinetune = async () => {
            if (!isTraining.value || isStopping.value) return;
            const ok = await dialog.confirm(
                '确定要停止当前训练吗？已训练的进度不会保存为可用模型。',
                { variant: 'danger', title: '停止训练', confirmText: '停止训练', cancelText: '继续训练' }
            );
            if (!ok) return;
            isStopping.value = true;
            try {
                const res = await apiFetch('/api/stop_finetune', { method: 'POST' });
                if (!res.ok) throw new Error(await friendlyHttpError(res));
                // 不在这里翻 isTraining —— 交给 SSE 的 stopped 帧统一处理（单一真相源）
            } catch (e) {
                const msg = (e instanceof TypeError) ? friendlyNetworkError() : e.message;
                if (msg) dialog.alert(msg, { variant: 'danger' });
            } finally {
                isStopping.value = false;
            }
        };
```

- [ ] **Step 7: 收尾 FIX ⑤ —— 把 finetune 里几处裸文案改走 errors.js**

- 第 708 行 `dialog.alert(\`启动失败：${errorData.detail}\`, ...)`（在 `const errorData = await res.json();` 之后）改为：
```javascript
                    const msg = await friendlyHttpError(res, '启动失败，请稍后重试。');
                    if (msg) dialog.alert(msg, { variant: 'danger' });
```
  （删除其上一行 `const errorData = await res.json();`，因 friendlyHttpError 自行读 body。）
- 第 712 行 `dialog.alert("网络请求失败，无法启动训练！", ...)` 改为 `dialog.alert(friendlyNetworkError(), { variant: 'danger' });`
- 第 644 行 `buildResult.value = { success: false, message: "网络请求失败，请检查后端服务。" };` 改为 `buildResult.value = { success: false, message: friendlyNetworkError() };`

- [ ] **Step 8: 把 isStopping、handleStopFinetune 加入 return**

在 return 对象 `handleStartFinetune,`（第 916 行）之后加：
```javascript
            isStopping,
            handleStopFinetune,
```

- [ ] **Step 9: 模板加红色停止按钮（`templates/index.html`）**

在启动按钮 `</button>`（第 967 行）之后、`<p v-if="!hasGpu" ...>`（第 968 行）之前插入：
```html
                <button v-if="isTraining" @click="handleStopFinetune" :disabled="isStopping"
                        class="w-full md:w-auto bg-gradient-to-r from-red-500 to-rose-600 hover:from-red-600 hover:to-rose-700 disabled:from-gray-300 disabled:to-gray-400 text-white px-12 py-5 rounded-full font-extrabold text-xl transition transform hover:scale-105 active:scale-95 flex items-center justify-center gap-3 shadow-xl mx-auto mb-6 disabled:transform-none disabled:shadow-none">
                    <svg v-if="!isStopping" class="w-7 h-7" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clip-rule="evenodd"></path></svg>
                    <svg v-else class="w-7 h-7 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                    {{ isStopping ? '正在停止…' : '停止训练' }}
                </button>
```

- [ ] **Step 10: bump 版本**

- `static/js/app.js:5`：`finetune_panel.js?v=1.8` → `?v=1.9`
- `templates/index.html:1305`：`app.js?v=2.10` → `?v=2.11`

- [ ] **Step 11: 真机验证（需 GPU 主机）+ 本地可做的部分**

本地（无 GPU）：硬刷新进「模型微调」，确认未训练时不显示红色停止按钮、不报错。
**GPU 主机**（真正验证）：
1. 生成数据集 → 启动一次真训练（小 epoch）。
2. 训练中出现红色「停止训练」→ 点它 → confirm → 观察：
   - 后端日志出现「训练已被用户停止，清理半成品输出目录」；
   - `nvidia-smi` 显存释放、`python finetune.py` 及其 dataloader worker 全部退出（无孤儿）；
   - 前端弹「训练已停止」，**不弹发布弹窗**，loss 看板停更；
   - `output/<user>/<model>/` 目录被删，模型名可再次使用；
   - `curl /api/gpu_status` 回到 `IDLE`，可立即再启动新训练。
3. 若 10s 后进程仍在，确认 SIGKILL 兜底生效。

- [ ] **Step 12: Commit**

```bash
cd /home/xiejianan/whisper-finetune-plus
git add static/js/finetune_panel.js static/js/app.js templates/index.html
git commit -m "feat(finetune): stop-training button + SSE 'stopped'/onerror handling, errors.js routing (tier-1 fix 2 frontend, fix 5 finetune)"
```

---

## 自检清单（撰写后核对，已通过）

- **Spec 覆盖**：① Task2、② Task6+7、③ Task5、④ Task4、⑤ Task1+3+7(SSE/finetune)。全部有对应任务。
- **类型/命名一致**：`friendlyHttpError(res)`/`friendlyNetworkError()` 全程同签名；`TRAINING_PROC` 字段（proc/pgid/cancelled）在 Step1/2/4/5/6 一致；前端 `isStopping`/`handleStopFinetune`/`showAdvanced`/`dialogRoot`/`focusModeRoot` 均在对应 return 中导出。
- **占位符**：无 TBD/TODO；每个改动给出完整代码或精确锚点+片段。
- **缓存版本**：每个改 app.js 的 Task 都 bump `app.js?v`：2.7(T2)→2.8(T3)→2.9(T4)→2.10(T5)→2.11(T7)；audio_collector 1.3(T2)→1.4(T5)；finetune 1.8(T4)→1.9(T7)；inference 1.13(T3)；errors.js 恒 ?v=1。按 Task 顺序执行版本单调递增。
- **已知边界**：`delete_task` 后端仍 200-on-not-found（设计文档列为 out of scope，前端只删可见 ID，不受影响）；FIX② 的 kill 链路需 GPU 主机实测（Task7 Step11）。
