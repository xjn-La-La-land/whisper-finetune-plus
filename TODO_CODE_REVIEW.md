# Code Review 修复 To-Do

> 本文档汇总了 2026-05-23 code review 发现的问题，按 **严重 → 较重 → 一般 → 细节** 分级，
> 每项包含：**问题描述**、**相关文件 / 行号**、**修复方案**、**预估工作量**、**完成状态**。
>
> 建议按"严重 → 较重"的顺序逐项推进；公网部署前必须完成 P0 全部项。

---

## 优先级速览

| 优先级 | 项数 | 必须在何时完成                |
| ------ | ---- | ----------------------------- |
| P0 严重 | 4    | 公网部署 / 任何真实用户接入前 |
| P1 较重 | 6    | 多用户灰度前                  |
| P2 一般 | 8    | 第一个正式版本前              |
| P3 细节 | 6    | 有空就清                      |

---

# 🔴 P0 严重问题（部署前必须修）

## [x] P0-1 完全没有认证机制，用户身份可冒充

**问题**
- `POST /api/login` 只校验 username 存在与否，没有密码 / token / session
- 前端把 `username` 写进 `localStorage`，所有 `/api/*` 通过 query / form 里的 `username` 字段做隔离
- 任意一个网页都能 POST `{"username":"小明"}` 冒充小明，调用 `DELETE /api/tasks?username=小明` 把别人录音清空
- 目标用户是脑瘫儿童，涉及未成年人音频和身份信息，合规风险高

**相关文件**
- `data_collector.py:108-145`（register / login）
- `static/js/app.js:38-70`（前端登录态）
- 所有路由的 `username` 参数

**修复方案**
1. `users` 表新增 `password_hash`、`created_at`、`last_login_at` 字段（保留旧数据兼容）
2. 注册时让用户/家长设一个 4-6 位 PIN（脑瘫儿童家长易操作），用 `bcrypt` 存哈希
3. 登录成功下发 JWT（`HS256` + 7 天过期 + refresh token），前端只存 token
4. 写一个 `Depends(get_current_user)`：从 `Authorization: Bearer ...` 解出 username
5. **所有 handler 把 `username` 参数从 query/form 里删除**，统一从 `Depends(get_current_user)` 取
6. 添加依赖：`python-jose[cryptography]`、`passlib[bcrypt]`

**预估工作量** 1 天

✅ 已修复 @ 2026-05-23
   方案：bcrypt 密码哈希 + JWT (HS256, 7 天) Authorization Bearer header；
   密码长度 4-16 字符任意 chars（用户选项 A）；旧用户行直接清空（用户选项 A）

   **新增 `utils/auth.py`**：
     * `hash_password` / `verify_password` —— bcrypt 含 salt 常量时间对比
     * `create_access_token(username)` —— 生成 7 天过期的 JWT (HS256)
     * `_decode_token(token)` —— 验证签名 + exp，失败返 None
     * `JWT_SECRET` 优先级：env `JWT_SECRET_KEY` > `data/.jwt_secret` 文件 > 自动生成持久化
     * `get_current_user`           FastAPI dep 从 `Authorization: Bearer <token>` 取
     * `get_current_user_from_query` 仅供 SSE / EventSource 用，从 `?token=` 取
       （EventSource 不支持自定义 header，只能走 query；权衡接受）

   **数据库 schema 迁移** (`data_collector.init_db`)：
     * `users` 表新增 password_hash / created_at / last_login_at
     * 旧版无密码的 users 行整体 DELETE；tasks 表 username 字段不动
     * 用户重新注册同名后可恢复 tasks 访问

   **register / login 改造**：
     * `UserAuth` 加 `password: str = Field(min=4, max=16)`
     * 注册：bcrypt 哈希 + 时间戳 + 直接返回 JWT，免去再 login 一次
     * 登录错误信息统一为"代号或密码错误"，不泄露用户名是否存在
     * 新增 `GET /api/me`：token 健康检查，返回 current user

   **所有 handler 改造**（21 个，全部从 query/form/body 移除 username，改为 Depends）：
     * data_collector.py: upload_txt / get_tasks / add_single_task /
       update_task_text / delete_task / clear_all_tasks / upload_audio
     * dataset_builder.py: check_dataset / build_dataset
     * inference_controller.py: user_models / latest_model_info /
       download_published_model / api_recognition
     * finetune_controller.py: start_finetune / check_model_name /
       base_models / base_models/download / gpu_status /
       **train_stream (query token!)** / train_history / publish_model / check_model
     * `FinetuneRequest` / `PublishModelRequest` 删 username 字段
     * `run_finetune_process` 改签名为 `(username, req)`，username 显式传入
     * `delete_task` 增强：WHERE username = current_user 防跨用户删除

   **前端新增 `static/js/api.js`**：
     * `apiFetch(url, opts)` —— 自动加 `Authorization: Bearer <token>` 头
     * 401 时清空 localStorage + 派发 `whisper:unauthorized` 事件
     * `sseUrl(path, params)` —— EventSource 用，token 拼到 query
     * `getToken` / `setToken` / `clearToken` / `getStoredUsername`

   **前端改造**：
     * `app.js`：登录/注册带 password；token 存 localStorage；启动时 `GET /api/me`
       验证 token；监听 `whisper:unauthorized` 自动登出
     * `index.html`：登录界面加密码输入框（autocomplete="current-password"）
     * audio_collector / inference_panel / finetune_panel：所有 fetch(?username=)
       替换为 apiFetch(无 username)；EventSource 改用 sseUrl()；POST body 移除 username

   **依赖**：`requirements.txt` 加入 `bcrypt`、`python-jose[cryptography]`

   **测试覆盖（14/14 通过）**：
     - 无 token → 401；篡改 token / 错误 scheme → 401
     - 注册返回 token；重名注册 → 400；密码错误 → 400
     - alice 看不到 bob 任务（数据隔离）；bob 改/删 alice 任务被拦
     - SSE 无 token → 401；query token → 200
     - 用户切换：`/api/me` 各返回各自 username

   **安全 trade-offs**：
     * SSE 走 query token —— 会出现在 nginx 访问日志，生产应屏蔽 `?token=`
     * 抢注风险：任何人能注册任意未占用代号；属公开注册系统常态，未来可加邀请码
     * **P0-2 / P0-3 / P0-4 仍待修**，共同构成"公网部署前必须修"全集
     * P0-3 没修：uploads 还在通过 StaticFiles 公开，知道路径仍能下载所有人录音

---

## [x] P0-2 路径穿越漏洞（uploads / dataset / output 目录）

**问题**
- `audio.filename` 来自客户端，`file_extension` 没做白名单
- `username` 全程没做字符校验，构造 `"../../etc"` 可逃出 uploads 目录
- `model_name` 同样未限制字符（影响 `output/{username}/{model_name}/`）

**相关文件**
- `data_collector.py:266-321`（upload_audio）
- `data_collector.py:218-243`（delete_task 里的 `os.remove(clean_path)` 用了 DB 里的相对路径）
- `finetune_controller.py:160-244`（构造 dataset_dir / output_dir / cmd）

**修复方案**
1. 新增 `utils/path_safety.py`：
   ```python
   USERNAME_RE = re.compile(r"^[a-zA-Z0-9_一-龥-]{1,20}$")
   MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9._-]{1,80}$")
   ALLOWED_AUDIO_EXT = {"webm", "wav", "mp3", "ogg", "m4a"}

   def safe_join(base: str, *parts) -> str:
       p = os.path.abspath(os.path.join(base, *parts))
       if os.path.commonpath([os.path.abspath(base), p]) != os.path.abspath(base):
           raise HTTPException(400, "非法路径")
       return p
   ```
2. 注册接口校验 `USERNAME_RE`
3. `start_finetune` 校验 `MODEL_NAME_RE`
4. 所有 `os.path.join(UPLOAD_DIR, username, ...)` 替换为 `safe_join(UPLOAD_DIR, username, ...)`
5. `upload_audio` 中 `file_extension` 走白名单
6. `delete_task` 里 `clean_path` 也走 `safe_join` 校验

**预估工作量** 0.5 天

✅ 已修复 @ 2026-05-24
   方案：纵深防御 = 输入正则白名单 + safe_join 越界检查 + 扩展名白名单。

   **新增 `utils/path_safety.py`**：
     * 常量 `UPLOAD_BASE` / `DATASET_BASE` / `OUTPUT_BASE` —— 锚定项目根的绝对路径
     * `USERNAME_RE = r"^[a-zA-Z0-9_一-龥-]{1,20}$"` —— 允许 ASCII + 常用 CJK
     * `MODEL_NAME_RE = r"^[a-zA-Z0-9_][a-zA-Z0-9._-]{0,79}$"` —— **比 review 草案更严**：
       首字符必须是字母/数字/下划线，以拒绝 `..` / `.` / `.foo` / `-foo`
       （review 的 `^[a-zA-Z0-9._-]{1,80}$` 会放过 `..` —— 测试时发现，已收紧）
     * `ALLOWED_AUDIO_EXT = {webm, wav, mp3, ogg, m4a}` —— 浏览器/桌面/移动端常见格式
     * `safe_join(base, *parts)` —— abspath + commonpath 校验；越界抛 HTTPException(400)
     * `safe_resolve_under(base, candidate)` —— 软失败版本，越界返 None
       （用于"DB 里的脏 audio_path 历史数据"等场景，避免单条脏数据让整个接口 400）

   **入口白名单**：
     * `register_user`：clean_name 通过 `is_valid_username` 才入库
     * `start_finetune`：model_name 通过 `is_valid_model_name` 才上锁
     * `check_model_name`：新增 reason `"invalid_chars"`，前端 onBlur 即时反馈
     * `upload_audio`：file_extension 不在 `ALLOWED_AUDIO_EXT` 中时回落到 `webm`

   **路径拼接保护**：把以下文件里所有"用 username/model_name 拼路径"的地方
   全部从 `os.path.join` 改为 `safe_join`：
     * `data_collector.py`：`sync_user_words_txt` / `clear_all_tasks` / `upload_audio`
     * `data_collector.py`：`delete_task` / `update_task_text` 里 DB 读出的 audio_path
       走 `safe_resolve_under(UPLOAD_BASE, …)`，挡历史脏数据里的 `../`
     * `finetune_controller.py`：`start_finetune` / `run_finetune_process` /
       `log_generator` / `get_train_history` / `publish_model` / `check_model`
       —— output 路径用 **两层 safe_join**（先 user 根，再 model_name），
       因为 `safe_join(OUTPUT_BASE, user, "..")` 的 abspath 还在 OUTPUT_BASE 内，
       只有两层 safe_join 才能挡 model_name=`..` 这种"上跳一层"的攻击
     * `dataset_builder.py`：`check_dataset` / `build_dataset` + DB 读 audio_path
       走 `safe_resolve_under`
     * `inference_controller.py`：`resolve_user_model_path` / `get_latest_model_info` /
       `download_published_model`

   **前端 `finetune_panel.js`**：
     * `checkModelNameAvailable` 增加 `invalid_chars` 分支，提示用户具体规则；
       与现有 `taken` / `too_long` / `empty` 一致地驱动输入框颜色 + 按钮禁用

   **端到端测试覆盖 (67/67 通过)**，关键场景：
     - T1) 注册时 `../etc` / `a/b` / `a.b` / `..` 全部 400；`alice` / `小明` 正常
     - T2) **核心攻击场景**：上传 `evil.py` 文件 → 扩展名被白名单回落到 `webm`，
           ffmpeg 转 py 文本失败 500，但 uploads/alice/ 下不会有 .py 文件落地
     - T3) **核心攻击场景**：直接在 DB 注入 `/uploads/../data/sentinel.txt` 的脏
           audio_path，调 `DELETE /api/task/{id}` → 接口 200 (DB 行被清)，
           但 sentinel.txt **没被删除**（`safe_resolve_under` 把越界路径返回 None）
     - T4) `PUT /api/task/{id}` 改文本时同样防御
     - T5) `safe_join` 拒绝 `..` / `/etc/passwd`；`safe_resolve_under` 越界返 None
     - T6) `MODEL_NAME_RE` 拒绝 `.` / `..` / `.foo` / `-foo` / 中文 / 含空格 / 81 字符
     - T7) `USERNAME_RE` 接受 CJK + ASCII；拒绝点号 / 空格 / 反斜杠 / 21 字符
     - T8) `ALLOWED_AUDIO_EXT` 大小写不敏感；拒绝 `py` / `sh` / `wav.py` / `..wav`
     - T9) `check_model_name` 实时校验路径打通：`../etc` → invalid_chars
     - T10) 数据隔离回归：alice / 小明 互相看不到对方任务

   **安全 trade-offs**：
     * `safe_join(OUTPUT_BASE, user, model_name)` 单层调用挡不住 model_name=`..`，
       已通过 MODEL_NAME_RE 在入口拦住 + 两层 safe_join 双保险
     * `safe_resolve_under` 对脏 DB 数据软失败：宁可"少删一个孤儿 wav"也不要
       因为一条脏数据让 DELETE 接口 500
     * 历史 DB 行的 model_name 未必过 MODEL_NAME_RE 校验，
       `resolve_user_model_path` 用 `safe_resolve_under` 跳过非法行而非整个 list 失败

---

## [x] P0-3 上传目录被 `app.mount` 公开，可遍历下载

**问题**
- `main.py:18` `app.mount("/uploads", StaticFiles(directory="uploads"))`
- 配合 P0-1 的"无认证 username 隔离"，任何人能拼出 `/uploads/小明/task_1.wav` 下载所有人录音
- 即使修了 P0-1，FastAPI 的 StaticFiles 也不经过认证依赖

**相关文件**
- `main.py:18`

**修复方案**
1. 删掉 `app.mount("/uploads", ...)` 这行
2. 改为一个受保护的路由：
   ```python
   @router.get("/api/audio/{task_id}")
   async def get_audio(task_id: int, user = Depends(get_current_user)):
       # 校验 task 属于 user
       # 返回 FileResponse
   ```
3. 前端 `audio_collector.js` / `inference_panel.js` 里所有 `task.audio_path` 改为请求新路由

**预估工作量** 0.5 天

✅ 已修复 @ 2026-05-24
   方案：StaticFiles 挂载彻底删除；新增 query-token 鉴权的受保护路由；前端用
   `sseUrl` 拼带 token 的 URL，模板照常用 `<custom-audio :src=...>`。

   **后端 `main.py`**：
     * 删 `app.mount("/uploads", StaticFiles(...))`，留下注释解释为什么不能用 StaticFiles
       （StaticFiles 不经过依赖系统，任何人猜到 `/uploads/{u}/task_X.wav` 都能下，
       P0-1 加的认证白做）
     * `/static` 仍走 StaticFiles，不影响

   **后端 `data_collector.py`**：
     * 新增 `GET /api/audio/{task_id}`，依赖 `get_current_user_from_query`
       （`<audio>` 元素不能附 Header，只能走 query token，与 SSE 一致的权衡）
     * **统一返回 404**：task 不存在 / 不属于当前用户 / 尚未录音 / 文件丢失，
       一律 404；避免状态码差异让攻击者通过遍历 task_id 推测系统状态
     * 用 `safe_resolve_under(UPLOAD_BASE, ...)` 兜历史脏数据，即便 DB 里有
       `/uploads/../something` 也只能在 UPLOAD_BASE 之下找文件
     * `get_tasks` 不再向前端返回 `audio_path` 字段——`/uploads/{u}/task_X.wav`
       是内部存储路径，外部用 `is_completed` + `updated_at` 就足够
     * `upload_audio` 响应也不再回显 db_audio_path

   **前端 `static/js/audio_collector.js` + `templates/index.html`**：
     * 新 helper `audioSrc(task)`：返回 `sseUrl('/api/audio/{id}', {v: updated_at})`
       —— 复用现成的 sseUrl 工具，token 透明拼到 query
     * 两处模板 `<custom-audio :src="task.audio_path">` 改为 `:src="audioSrc(task)"`,
       同时去掉 `&& task.audio_path` 条件（前端已不持有该字段）
     * inference_panel.js 不动（不消费 audio_path）

   **DB schema 保持不变**：`tasks.audio_path` 列仍存 `/uploads/{u}/task_X.wav`,
   服务端清理 / 数据集构建仍按此字段定位文件——只是这条信息不再外泄给客户端。

   **端到端测试 (18/18 通过)**：
     - T1) `/uploads/alice/task_1.wav` → 404（StaticFiles 已移除）；
           `/uploads/../etc/passwd` 也 404
     - T2) `/api/audio/{id}` 无 token / 错 token / 仅走 Header → 全部 401
           （这接口的认证模式只接受 query token）
     - T3) **核心场景**：alice 自己的 task 用 alice token → 200 +
           Content-Type=audio/wav + 文件字节完全匹配
     - T4) **核心场景**：用 bob token 取 alice 的 task → 404
           （不区分"不属于"和"不存在"，避免枚举）；bob 拿自己的 task 仍 200
     - T5) 不存在的 task_id / 尚未录音的 task → 都是 404（统一错误）
     - T6) `GET /api/tasks` 响应里不再含 `audio_path` 字段；
           `is_completed` + `updated_at` 仍提供给前端拼 URL
     - T7) DB 内部 schema 不变（audio_path 列仍存内部路径）
     - T8) 文件被外部删除 → 404 而非 500（safe_resolve_under 软失败 + 文件存在性检查）

   **安全 trade-offs**：
     * Token 走 query 出现在 nginx 访问日志里——同 SSE，已知问题。
       生产部署应在 nginx 配 `log_format` 屏蔽 `?token=`，或换成 cookie-based 短期 token。
     * `<audio>` 元素发起的请求可能带 Range，FastAPI/Starlette `FileResponse` 原生支持
       206 Partial Content，所以播放进度条 seek 应该照常工作（未单测验证）。
     * `Cache-Control` 未显式设置——FileResponse 默认不会让浏览器持久缓存，配合
       URL 上的 `?v=updated_at` cache buster，重录后能正常刷新。

   **剩余 P0**：P0-4（CORS）—— 与 P0-3 关系不大，可单独修。

---

## [x] P0-4 CORS / CSRF 防护缺失

**问题**
- FastAPI 默认无 CORS 中间件；cookie / token 也无 `SameSite=Strict`
- 用户在浏览器上同时打开其他网页，那个网页可以静默 POST `/api/start_finetune` 触发训练

**相关文件**
- `main.py`

**修复方案**
1. JWT 改用 `Authorization` 头（不放 cookie）天然防 CSRF
2. 添加 CORS 中间件，允许列表只放生产域名：
   ```python
   app.add_middleware(CORSMiddleware,
       allow_origins=[FRONTEND_ORIGIN],
       allow_credentials=False,
       allow_methods=["*"],
       allow_headers=["*"])
   ```
3. cpolar 公网链接做白名单，开发环境用 `*.localhost`

**预估工作量** 0.5 小时（与 P0-1 一起做）

✅ 已修复 @ 2026-05-24
   方案：JWT 走 Authorization header 已天然防 CSRF（P0-1 落地），CORS 中间件
   显式声明 deny-all 默认 + env 白名单做卫生层。

   **威胁模型澄清**：
     * 我们的鉴权完全走 `Authorization: Bearer <JWT>`，token 存 localStorage。
     * 跨域恶意页面（evil.com）拿不到 localStorage（同源策略），所以即便发
       cross-origin fetch 到我们的接口，**也没有 token 可附**，handler 直接 401。
     * 也就是说 CSRF 在架构层已被根除，不依赖 CORS。CORS 在此项目里更多是
       "显式声明只接受指定 origin"的卫生层，避免后续不小心引入 cookie 鉴权或
       新的同源假设时出问题。

   **后端 `main.py`**：
     * 加 `CORSMiddleware`，`allow_origins` 从 env `ALLOWED_ORIGINS`
       逗号分隔解析；空值 → 空列表 → deny-all 默认
     * `allow_credentials=False` —— 不用 cookie，关掉可以兼容 `["*"]`
       且不至于把 cookie 跨域漂移
     * `allow_methods=["*"]` / `allow_headers=["*"]` —— 不收紧，简化部署
     * 启动时打印当前白名单，方便发现 ALLOWED_ORIGINS 拼写错误

   **部署文档**（待 P3-6 同步更新）：
     - cpolar 公网链接（如 `https://abc.cpolar.io`）放进 `ALLOWED_ORIGINS`
     - 同源场景（用户直接访问 `http://localhost:8000`）无需配置——同源请求
       不带 Origin 头，CORSMiddleware 不会拦
     - 多 origin 用逗号分隔：`ALLOWED_ORIGINS="https://a.cpolar.io,http://localhost:8000"`

   **端到端测试 (12/12 通过)**：
     - G1) 未配置 ALLOWED_ORIGINS：同源 GET 正常 200；跨域 GET 服务端返 200
           但无 ACAO 头（浏览器层会拦 JS 读 response，安全）；preflight 也不返 ACAO
     - G2) 配置 `http://app.localhost:8000,https://abc.cpolar.io`：
           匹配的 origin preflight 回显 ACAO + 允许 Authorization header；
           简单 GET 也带 ACAO；非白名单 origin 无 ACAO
     - G3) **关键回归**：CORS 中间件不能挡掉同源调用——register / 拉 tasks
           全部走通（同源请求不带 Origin 头，CORSMiddleware 旁路）

   **安全 trade-offs / 已知限制**：
     * 没改 `allow_origin_regex`——如果以后要"所有 *.cpolar.io 都允许"，再加。
       当前部署只有一两个固定 origin，列表足够
     * `allow_methods=["*"]` 会回 `GET, POST, PUT, DELETE, OPTIONS, ...`
       而不是逐项列出。如果合规上要求最小化，可改成 `["GET", "POST", "PUT", "DELETE"]`
     * Token 走 query 的接口（SSE / `/api/audio`）和 CORS 无关——浏览器不会对
       `<audio src=...>` / `EventSource` 触发 CORS（它们是 simple cross-origin
       fetches，无 preflight），所以这部分本来就要靠 token 本身防泄露

   **P0 阶段全部完成**：P0-1（认证）/ P0-2（路径穿越）/ P0-3（uploads 公开）/
   P0-4（CORS）全绿，公网部署的硬性门槛已清。下一步 P1。

---

# 🟠 P1 较重问题

## [x] P1-1 GPU 状态锁存在竞态（多用户并发）

**问题**
- `GPU_STATE` 是普通 dict，"检查 + 上锁"两步之间会让出 event loop
- 两个并发请求都可能通过检查，同时进入训练 / 推理状态

**相关文件**
- `shared_state.py`
- `finetune_controller.py:263-271`
- `inference_controller.py:215-290`

**修复方案**
1. `shared_state.py` 加 `import asyncio; GPU_LOCK = asyncio.Lock()`
2. 所有 GPU 状态读写都包在 `async with GPU_LOCK:` 里
3. 推理快速路径（模型已加载）也要进锁——避免推理中途被另一个推理覆盖 `current_user`

**预估工作量** 0.5 天

✅ 已修复 @ 2026-05-23
   - `shared_state.py` 新增 `GPU_LOCK = asyncio.Lock()`，文档注释中说明锁的语义：
     "仅在检查 + 转移瞬间持有，长时占用通过 status != IDLE 表达"
   - `finetune_controller.start_finetune`：检查 + 转移用 `async with GPU_LOCK` 包成原子
   - `finetune_controller.run_finetune_process` 的 finally：状态重置也包进 lock
   - `inference_controller.api_recognition`：重构为「锁外查模型路径 → 锁内原子转移
     → 锁外做加载/推理 → finally 锁内重置」四段式
   - 行为变化（请知悉）：原代码允许多个推理请求"同时"开跑（实际在 GPU 上仍串行，
     但状态有 race），新代码后到的推理请求会直接返回 423。如果想改成"排队等候"，
     未来可以把 `if status != IDLE: raise` 改成 `while ...: await asyncio.sleep`，
     或者用 `asyncio.Condition`。当前选择拒绝是因为 UX 上"立刻告诉用户忙"比无声等待好。
   - 验证：6 个并发场景全绿
     T1) 两个并发 start_finetune → 1 成功 1 个 423
     T2) 训练中收到推理请求 → 423
     T3) 推理中收到训练请求 → 423
     T4) 两个并发推理请求 → 1 成功 1 个 423（关键 race，原代码两个都通过）
     T5) 训练正常结束 → IDLE
     T6) 推理内部抛异常 → IDLE

---

## [x] P1-2 `run_finetune_process` 的 GPU 锁释放不在真正的 finally

**问题**
- `run_finetune_process` 里 `open(web_log_path, "w")` 在 `try` 之外，如果它抛异常（磁盘满、目录被删），`GPU_STATE` 永远不会被释放

**相关文件**
- `finetune_controller.py:160-244`

**修复方案**
- 把整段逻辑包到 `try/finally`，`finally` 里强制 `GPU_STATE["status"] = IDLE`

**预估工作量** 10 分钟

✅ 已修复 @ 2026-05-23
   - 变更 1：`run_finetune_process` 整个函数体包进顶层 `try/finally`，
     `open(web_log_path)` 和命令拼接全部移入 try 块，
     新增 `os.makedirs(dataset_dir, exist_ok=True)` 防御性兜底
   - 变更 2：`start_finetune` 在 GPU 上锁之前加 train.json/test.json 存在性校验，
     缺失时返回 400 且不上锁，把错误路径前置
   - 验证方式：3 个异常路径单测（open 抛异常 / subprocess 退出非零 / subprocess 启动失败）
     全部 GPU 锁能正确释放；start_finetune 前置校验返回 400 且 GPU_STATE 不变；
     happy path 正常上锁并触发 background_task

---

## [ ] P1-3 生产部署必须去掉 `--reload`

**问题**
- 部署文档教学用 `uvicorn main:app --reload`，文件变动时主进程被重启
- 训练子进程还在烧显存，但 `GPU_STATE` 已重置 → 状态错乱

**相关文件**
- `Whisper语音识别项目网页端部署文档.md:80`
- 部署脚本（如有）

**修复方案**
1. 文档分两节："开发模式 / 生产模式"
2. 生产模式用 `uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1`（不能加 workers > 1，否则 `GPU_STATE` 多份）
3. 可考虑用 systemd 单元写一份示例

**预估工作量** 30 分钟

---

## [x] ~~P1-4 `finetune.py` 的 `type=bool` argparse 失效~~ — **已核实，非 bug**

**复核结论（2026-05-23）**
- review 时漏看了 `utils/utils.py` 的 `add_arguments` 包装函数
- `add_arguments` 第 34 行 `type = strtobool if type == bool else type` 已经把 `bool` 替换成了 `strtobool`
- `strtobool`（utils.py:16-23）正确解析 `"True"/"False"/"1"/"0"` 字符串
- `finetune_controller.py:187-189` 拼出来的 `"--use_8bit=False"` 到 finetune.py 这边会被正确解析为 Python `False`

**结论**：原代码正确处理布尔参数，无需修改。

---

## [x] P1-5 SQLite 同步调用阻塞 event loop（长期方案）

**问题**
- 所有 handler 是 `async def`，内部 `sqlite3.connect/execute/fetchall` 是同步阻塞
- 录音过程前端高频轮询 `/api/tasks`，并发上来时整个 worker 卡顿

**相关文件**
- 几乎所有 `*.py` 的 DB 调用

**修复方案（短期）**
- 抽一个工具函数 `db_query(sql, params)` 内部 `await asyncio.to_thread(...)`，所有 DB 调用统一走它

**修复方案（长期）**
- 换 `aiosqlite` 或 SQLAlchemy async
- 同时把 `with sqlite3.connect(...) as conn:` 统一改成上下文管理器，避免异常路径泄漏（参见 P2-1）

**预估工作量** 0.5 天（短期）/ 1 天（长期）

✅ 已修复（长期方案）@ 2026-05-23
   - 选用 aiosqlite 而非 SQLAlchemy async：项目只有 3 张表纯 SQL，ORM 是过度工程；
     且 `aiosqlite.IntegrityError is sqlite3.IntegrityError`，迁移机械化
   - `requirements.txt` 加入 `aiosqlite`
   - `utils/db.py` 重写为双 API：
     * `get_db()` —— 异步版（aiosqlite），供 FastAPI handler 等 async 上下文用
     * `get_db_sync()` —— 同步版（sqlite3），供 init_db / CLI 脚本等无 event loop 的场景用
   - 迁移所有调用点（共 19 处）：
     * `data_collector.py`：1 处 sync (init_db) + 10 处 async；`sync_user_words_txt`
       从同步函数升级为 `async def`，调用方相应 await
     * `finetune_controller.py`：`_upsert_user_model` / `_model_name_exists` / publish_model
       全部 async；调用方相应 await
     * `inference_controller.py`：`resolve_user_model_path` 升级 async；3 个 handler
       全部用 async with
     * `dataset_builder.py`：1 处 async
     * `sync_uploads_to_db.py`：保持同步，改用 `get_db_sync`
   - aiosqlite 语法关键差异：
     * `with` → `async with`
     * `conn.execute(...)` → `await conn.execute(...)`，返回 cursor 也是 async
     * `c.fetchone()` / `.fetchall()` → `await c.fetchone()` / `await c.fetchall()`
     * `conn.commit()` → `await conn.commit()`
     * 用 `cursor.rowcount` 取行数（与 sqlite3 一致）
   - 验证（5/5 通过）：
     T1) 业务模块全部 import 成功（含 init_db 在 import 时同步执行）
     T2) 异步 DB 单次读写正常
     **T3) 关键性能验证**：20 个并发 DB worker × 各 20 个 INSERT + 1 SELECT，
          总耗时 1348ms 期间心跳协程（5ms 一跳）打了 **261 次**，证明 event
          loop 没被 DB 阻塞。同样负载用 sync sqlite3 心跳大概只能跑 0-2 次
     T4) IntegrityError 异常路径下连接仍 close，后续写入正常
     T5) 业务 handler 端到端（register/login/get_tasks）走通

---

## [x] P1-6 SSE 日志路径设计存在串流风险

**问题**
- `web_log_path` 现在按 `dataset/{username}/training_log.jsonl` 命名
- 同一用户两次连续训练会清空 → 新写，SSE 端的"truncate 回退"逻辑（`f.seek(0)`）可能读到旧训练残留
- 日志和具体模型版本无绑定，不方便追溯

**相关文件**
- `finetune_controller.py:165-170`、`315-372`
- `finetune.py:WebMonitorCallback`

**修复方案**
1. 把日志路径改为 `output/{username}/{model_name}/training_log.jsonl`
2. SSE 接口新增 `model_name` 参数
3. 前端 `startSSE` 也带上 `model_name`
4. 删掉 `dataset/{username}/training_log.jsonl` 的旧文件清理逻辑

**预估工作量** 1 小时

✅ 已修复 @ 2026-05-23
   - **后端 `finetune_controller.py`**：
     * `run_finetune_process`: `web_log_path` 改到 `output/{username}/{model_name}/training_log.jsonl`
       并 `os.makedirs(output_dir, exist_ok=True)` 兜底
     * `log_generator(username, model_name)` 新增 model_name 参数
     * SSE 端点 `train_stream` 和 history 端点 `get_train_history` 都要求 `model_name`（必填）
   - **共享状态 `shared_state.py`**：
     * `GPU_STATE` 新增 `current_model_name` 字段
     * `start_finetune` 在 GPU 上锁时同步设置；`run_finetune_process` 的 finally 一并重置
     * 让前端刷新页面后能从 `/api/gpu_status` 知道"正在训练的是哪个模型"
   - **前端 `finetune_panel.js`**：
     * `startSSE({ resetChart, modelName })`: SSE URL 拼上 model_name
     * `loadChartHistory(modelName)`: 必须带 model_name，否则清空图表
     * `checkTrainingStatus`: 从 GPU 状态拿 `current_model_name`，自动填回输入框、
       加载已有 loss 曲线、续连 SSE
     * `onMounted` / `onActivated` / `watch(currentUser)`: 不再无条件调
       `loadChartHistory`，由 `checkTrainingStatus` 在有 model_name 上下文时主动调用
     * `handleStartFinetune`: `startSSE` 调用带上 `modelName: finetuneParams.model_name`
   - 验证（5/5 通过）：
     T1) train_history 接口正确读取 `output/{u}/{m}/training_log.jsonl`
     T2) SSE / history 路由签名要求 model_name 必填（FastAPI 422）
     T3) /api/gpu_status 暴露 current_model_name
     **T4) 核心 race 场景**：用户连续训练 model_v1 → model_v2，v1 有遗留日志
          (step=10,20, loss=0.9,0.8)，SSE 接入 v2 后只读到 v2 自己的
          (step=5,10, loss=0.6,0.5)，**绝无串流**
     T5) run_finetune_process 失败时 current_model_name 被释放为 None

---

# 🟡 P2 一般问题

## [x] P2-1 异常路径下 SQLite 连接泄漏

**问题**
- `delete_task`、`resolve_user_model_path` 等地方在 `os.remove` / 文件 IO 抛异常时不会关 conn

**修复方案**
- 统一改 `with sqlite3.connect(DB_PATH) as conn:`
- 或抽 FastAPI 依赖 `def get_db(): conn = ...; try: yield conn; finally: conn.close()`

**预估工作量** 0.5 小时

✅ 已修复 @ 2026-05-23
   - 新建 `utils/db.py` 提供 `get_db()` 上下文管理器
     * 注意 sqlite3.Connection 自带的 `with conn` 只 commit/rollback **不 close**
       （Python 文档明确的坑），所以必须用我们自己的 helper
     * 接受可选 `row_factory=sqlite3.Row` 和 `db_path=...`（仅测试用）
   - 把请求路径上所有 18 处 `conn = sqlite3.connect(...) ... conn.close()` 模式
     全部改为 `with get_db() as conn:`：
     * `data_collector.py`：11 处（init_db / sync_user_words_txt / register /
       login / upload_txt / get_tasks / add_single_task / update_task_text /
       delete_task / clear_all_tasks / upload_audio）
     * `finetune_controller.py`：3 处（_upsert_user_model / _model_name_exists /
       publish_model 的状态更新段）
     * `inference_controller.py`：3 处（resolve_user_model_path /
       get_latest_model_info / download_published_model）
     * `dataset_builder.py`：1 处（build_dataset）
   - `sync_uploads_to_db.py` 也顺手用 `get_db(db_path=...)` 替换
   - `finetune_controller.py` 不再直接 `import sqlite3`（无残留引用）
   - 验证（5/5 通过）：
     T1) 业务模块全部 import 成功，含 `init_db()` 启动时调用
     T2) 基本读写 opened==closed
     T3) `IntegrityError` 异常路径 opened==closed
     T4) `with` 块内早 return / 早 raise 都 opened==closed
     T5) 连续 10 次异常请求 opened==closed（验证无累积泄漏）
   - **遗留小尾巴**：`data_collector` / `inference_controller` / `finetune_controller`
     / `dataset_builder` 各自模块级别还有一个未使用的 `DB_PATH` 常量。功能上
     是死代码但不影响行为，未来清理时一并删掉。

---

## [x] P2-2 修改任务文本时未清除旧音频

**问题**
- `PUT /api/task/{task_id}` 只改文本，旧录音和新文本对不上，`is_completed=1` 还在
- 下次生成数据集时会污染训练数据（旧音频 + 新文本）

**相关文件**
- `data_collector.py:203-215`

**修复方案**
- 修改文本时：`is_completed = 0`, `audio_path = NULL`, 物理删除旧 wav
- 前端弹一个 confirm："修改文本会丢失已录音频，是否继续？"

**预估工作量** 30 分钟

✅ 已修复 @ 2026-05-23
   - **后端 `data_collector.update_task_text`**：
     * 先 SELECT 旧 text 和 audio_path
     * 找不到 task 或不属于当前用户 → 404（之前会静默 0 行 UPDATE）
     * 文本未变化 → 提前 return，保留录音（用户开了编辑又原样保存的场景）
     * 文本变化 → SET audio_path=NULL, is_completed=0, updated_at=now
     * 提交 DB 之后再删物理文件：即使 os.remove 抛权限错也只是留下孤儿 wav，
       不会污染 dataset（is_completed=0 是源头标记）
     * 返回值新增 `audio_cleared` 字段供前端反馈
   - **前端 `audio_collector.saveEdit`**：
     * 比对新旧文本，仅当文本真的改了且任务已有录音时才弹 confirm
     * 用户点取消 → 不发请求
   - 验证（6/6 通过）：
     T1) 有录音 + 改文本 → 删除文件 + DB 重置 + audio_cleared=True
     T2) 文本未变 → 保留录音 + audio_cleared=False
     T3) 无录音 + 改文本 → 仅更新文本
     T4) 不存在的 task_id → 404
     T5) 修改别人的 task → 404，原 task 不动
     T6) 物理文件删除失败 → DB 状态仍更新，留孤儿 wav 但不影响 dataset

---

## [x] P2-3 音频质量校验缺失

**问题**
- 脑瘫儿童录音容易出现"录了 0.3 秒就停"或"全程没说话"
- `upload_audio` 没有任何长度 / 能量校验

**相关文件**
- `data_collector.py:265-321`

**修复方案**
- ffmpeg 转换后 `sf.info(final_wav_path)` 检查 `duration >= 0.5s`
- 可选：计算 RMS 能量，太低时给前端返回"似乎没有录到声音"
- 不达标时不写 `is_completed = 1`，前端给一个 toast 提示重录

**预估工作量** 1 小时

✅ 已修复 @ 2026-05-24
   方案：ffmpeg 转换完直接 sf.read 算时长 + RMS；不达标删坏文件 + 重置 DB + 400。

   **阈值选取（重要）**：
     * `MIN_AUDIO_DURATION_SEC = 0.5` —— 挡"录到 0.3s 就停"
     * `MIN_AUDIO_RMS = 0.002` —— **刻意定得很宽**，只挡纯零 / 极端弱噪。
       实测：正常室内底噪 -50dB FS ≈ 0.003，悄声细语 -30dB FS ≈ 0.030，
       脑瘫儿童含糊低声也远超 0.002。绝不能误伤这部分用户的录音。
     * 阈值选取过程已在代码注释里留下，方便后续调参时回顾

   **后端 `data_collector.py`**：
     * 新增 `validate_audio_quality(wav_path) -> Optional[str]`：
       sf.read → 多声道兜底降单声道 → 算 duration → 算 RMS → 返中文错误或 None
     * RMS 用 float64 累加避免大文件 float32 精度损失
     * `upload_audio` 在 ffmpeg 成功后、DB UPDATE 前调校验；失败时:
       (a) 删坏文件；(b) **重置 DB 行**（audio_path=NULL, is_completed=0）
       关键场景：用户已有合格 v1 录音，再录一个失败的 v2 —— ffmpeg 用 `-y`
       已经覆盖了 v1，此时如果不回滚 DB，会出现"DB 说 completed=1 但文件是坏的"
       脏状态，UI 上 task 还显示完成但播放会 404
     * 不需要前端任何改动：现有 `dialog.alert(e.message)` 直接吐 detail 给用户

   **端到端 + 单元测试 (20/20 通过)**：
     - U1-U6) `validate_audio_quality` 单元覆盖：合格 / 太短 / 全静音 /
              不存在的文件 / 0.5s 边界 / RMS 边界
     - E2E1) 合格 1.5s 录音 → 200，is_completed=1
     - E2E2) **核心场景**：先上传合格 v1（completed=1），再上传 0.3s v2 →
             400 提示太短，is_completed 回滚到 0，且 `GET /api/audio` 立刻 404
             （证明 ffmpeg 覆盖产生的坏文件已被清掉）→ 再上传合格 v3 →
             重新 200 + completed=1
     - E2E3) 全静音 wav 上传 → 400，提示"似乎没有录到声音"，DB 同样回滚
     - E2E4) 新任务首次就上传坏录音 → 400，is_completed 保持 0（无回滚需要）
     - 回归：P0-2 (67) / P0-3 (18) / P0-4 (12) 全绿

   **trade-offs**：
     * **替换式语义**：用户点"再录一次" + 失败 → 之前的好录音也没了。可以改成
       sidecar (写 .new + 校验通过才 atomic rename) 来保留 v1，但 UX 更让人困惑
       （"我刚录了一次，怎么显示的还是旧的？"）。当前选简单语义。
     * RMS 阈值是经验值，没做大规模真实脑瘫儿童录音的统计。如果上线后发现
       误伤率高，把 0.002 降到 0.001 即可（仍能挡纯零）。
     * 没做"前端预校验时长"——浏览器 MediaRecorder API 上可以拿到 duration,
       但前端预校验仍不可信，后端必须兜底。前端可选添加（属 UX 优化，不属安全）。

---

## [x] P2-4 依赖未钉版本

**问题**
- `requirements.txt` 里 `transformers / peft / fastapi / torch / tensorflow` 全无版本号
- 代码里有 `# transformer 4.42.1 不支持 processing_class` 这种针对特定版本的 workaround
- 新机器装最新 transformers 会立刻 break

**相关文件**
- `requirements.txt`

**修复方案**
1. 在能跑通的机器上 `pip freeze > requirements.lock.txt`，钉所有 ML 依赖
2. `requirements.txt` 钉到 minor（`transformers~=4.42.0`）
3. 评估升级 transformers，把 `tokenizer=` 改回 `processing_class=`

**预估工作量** 1 小时

✅ 已修复 @ 2026-05-24
   方案：硬件无关的依赖全部 `==X.Y.Z` 严格钉死；硬件相关（torch / tf / flash-attn）
   不钉，放注释区写清楚不同 GPU/CUDA 的对应安装命令。

   **决策依据**：
     * 在用户当前能跑通的 `whisper` conda env (Python 3.11 / RTX 4090 / CUDA 13.0)
       上 pip freeze 拿到准确版本，作为 known-good 基线
     * 用 `==` 而不是 `~=`：reproducibility 优先，宁可让用户手动升级也不要某次
       `pip install` 把 pydantic 偷偷升到 v2 让 fastapi 0.95.1 直接 crash
     * `evaluate` 包从清单中删除——grep 全工程没人 import，是死依赖

   **关键版本约束理由**：
     * `fastapi==0.95.1` + `pydantic==1.10.26` + `starlette==0.26.1` 必须三件套
       钉死。fastapi 0.95 只兼容 pydantic v1；任一升级到 v2 都要改 BaseModel 写法
     * `transformers==4.51.3`：当前能跑。代码里有"4.42.1 不支持 processing_class
       所以用 tokenizer=" 的历史注释，新版 `tokenizer=` 仍是 deprecated alias 仍可用
       —— 等真要升级 transformers 再回头改成 `processing_class=`
     * `numpy==1.26.4` 而非 env.yaml 里的 1.23.5：现在的 librosa/soundfile 已要求
       numpy>=1.24，env.yaml 数据过期

   **硬件相关依赖处理**：放进 `requirements.txt` 末尾大段注释里，分三档：
     - 必装 torch：分 RTX 4090+CUDA 13.0 和 RTX 3090/A100+CUDA 12.8 两套示例命令
     - 可选 tensorflow+keras (2.12.0)：仅 TFLite 导出需要；现在 `finetune_controller`
       已经 try/except 兜住 import，不装也能跑微调
     - 可选 flash-attn：推理时模块加载自动检测 try import，没装就 fallback SDPA

   **验证**：
     - `pip install --dry-run -r requirements.txt` 在当前 env 上 0 冲突，
       所有版本已 satisfied（说明 pin 列表与实际跑通环境完全一致）
     - 回归 P0-2 (67) / P0-3 (18) / P0-4 (12) / P2-3 (20)，全部依然绿

   **遗留小尾巴**：env.yaml 还有一份旧的依赖列表（torch 2.7.1 / transformers 4.42.1
   / Python 3.9），与新 requirements.txt 不一致。env.yaml 是历史 conda 环境定义，
   现在已被 requirements.txt 取代。下一次清理 P3-6 部署文档时一起处理。

---

## [x] P2-5 数据集划分不可复现

**问题**
- `dataset_builder.py:90` `random.shuffle(data_list)` 没固定 seed，每次 train/test 划分都不同
- 不利于做"调参 → 对比"实验

**修复方案**
- `DatasetBuildRequest` 增加 `seed: int = 42`
- `random.Random(seed).shuffle(data_list)`

**预估工作量** 5 分钟

✅ 已修复 @ 2026-05-24
   - `DatasetBuildRequest` 新增 `seed: int = Field(default=42)` 字段
   - `random.shuffle(data_list)` 改为 `random.Random(request.seed).shuffle(data_list)`
   - 用独立 `Random` 实例而非 `random.seed(...)`，避免污染全局 random 状态
     （否则其他用 `random.random()` 的代码会被这次调用副作用化）
   - 前端暂未暴露 seed 参数，默认 42 即可；未来要做"调参对比"可以在 UI
     加个高级选项
   - 验证：同一 seed → 同样的 [25, 23, 19, 11, 4, 45, ...] 顺序；不同 seed
     给出不同顺序；并且 build_dataset 调用前后全局 random 状态保持不变

---

## [ ] P2-6 文档与实现不一致（evaluation.py 缺失）

**问题**
- 部署文档目录树列了 `evaluation.py`
- 项目根目录实际没有此文件，"测试集字错率"功能未实现或被合并

**修复方案**
- 二选一：把字错率计算逻辑补回来；或者更新文档

**预估工作量** 视实现决定（0.5h ~ 2h）

---

## [x] P2-7 重名校验只在提交时返回错误

**问题**
- 用户填完所有参数点了"启动微调"，才告诉他"模型名已存在"

**修复方案**
- 新增 `GET /api/check_model_name?username=&name=`
- 前端 `finetune_panel.js` 在 model_name 输入框 `onBlur` 时调用，红字提示

**预估工作量** 30 分钟

✅ 已修复 @ 2026-05-23
   - **后端**：`finetune_controller.py` 新增 `GET /api/check_model_name?username=&model_name=`
     * 返回 `{"available": bool, "reason": "empty"|"too_long"|"taken"|None}`
     * 复用已有的 `_model_name_exists` async helper
     * 校验长度 ≤ 80（与 `FinetuneRequest.model_name` 的 max_length 一致）
     * `model_name.strip()` 后再查，与 `start_finetune` 行为对齐
   - **前端 JS** `finetune_panel.js`：
     * 新增 `modelNameStatus` ref（state: idle/checking/available/taken/invalid）
     * `checkModelNameAvailable()` 在失焦时调，根据 reason 给中文提示
     * `clearModelNameStatus()` 在 @input 时调，避免 stale 提示
     * `handleStartFinetune` 增加前端守门：状态为 taken/invalid 时直接拦截，
       但后端 `start_finetune` 仍是最终把关
     * `watch(currentUser)` 切换用户时清状态
   - **前端 HTML** `index.html`：
     * 输入框 `@blur` / `@input` 绑定上述函数
     * 输入框边框 + 背景颜色按状态切换（灰/绿/红）
     * 提示文字按状态切换颜色（灰/绿/红）
     * "启动微调训练" 按钮在 taken/invalid 时 `:disabled`，与现有的 isTraining/!hasDataset 串联
   - 验证（7/7 通过）：
     T1) 空字符串 / 纯空格 → reason='empty'
     T2) 81 字符 → reason='too_long'，恰好 80 字符 → available=True（边界）
     T3) 已存在 → reason='taken'
     T4) 全新名称 → available=True
     T5) 同名但不同用户 → available=True（unique 约束是 (user, name) 组合键）
     T6) `"  vocal_v1  "` 前后空格 → strip 后识别为 taken

---

## [x] P2-8 前端用浏览器原生 `alert/confirm`

**问题**
- 目标用户群（脑瘫儿童 + 家长）对原生弹窗不友好（字小、按钮挤）
- 视觉上和精心设计的 Tailwind UI 不协调

**修复方案**
- 写一个 `<UiDialog>` 全局组件，提供 `alert(msg)` / `confirm(msg)` 同名 API
- 替换 `audio_collector.js` / `finetune_panel.js` / `inference_panel.js` 里所有原生调用

**预估工作量** 0.5 天

✅ 已修复 @ 2026-05-23
   - **新增 `static/js/dialog.js`**：Promise-based API，单例 reactive state
     * `dialog.alert(msg, opts?)` 返回 `Promise<void>`，单按钮
     * `dialog.confirm(msg, opts?)` 返回 `Promise<boolean>`，两按钮
     * `opts.variant`: `'info' | 'success' | 'warning' | 'danger'` 决定配色和图标
     * 上一个对话框未关时新对话框开启会 resolve(false) 旧的，避免 Promise 堆积
   - **`templates/index.html` 顶层加 dialog 模板**：放在 `#app` 顶层 + 登录遮罩外，
     z-index 300（高于一切），登录界面也能用
   - **`static/js/app.js`**：import dialog 模块，把 state + 处理函数 return 给模板；
     全局 `keydown` 监听 Enter（确认）/ Esc（取消）
   - **替换 26 处原生调用**：
     * `app.js` 4 处（注册 / 登录失败 / 网络错误 / 退出确认）
     * `audio_collector.js` 7 处（含 3 处 confirm：删任务 / 清空 / 改文本丢音频）
     * `inference_panel.js` 3 处（选模型 / 麦克风权限 / 复制失败）
     * `finetune_panel.js` 12 处（含训练异常、参数校验、发布成功等）
     * 全部按场景使用适当 variant：表单校验 warning，网络/系统错误 danger，
       破坏性操作 confirm + danger，发布成功 success
   - **UX 设计要点（针对脑瘫儿童 + 家长）**：
     * 按钮 ≥ 60px 高度（`py-4`），`gap-4` 间距避免肌肉震颤误触
     * 大字号（`text-base` 正文 + `text-xl` 标题），高对比配色
     * 类型变体顶部色条 + 大图标（💡 ⚠️ 🎉 ⛔）作视觉锚点
     * 点背景**不**自动关闭（无意识动作不会关掉对话框）
     * 键盘可访问（Enter 确认 / Esc 取消）
   - 验证：
     * 6 个 JS 文件括号平衡 + 0 处裸 `alert()` / `confirm()` 残留
     * uvicorn 启动后 `static/js/dialog.js` 200 OK
     * `index.html` 包含 dialog 模板节点（grep dialogState.visible）

---

# 🟢 P3 细节 / 优化

## [x] P3-1 `finetune_controller.py` 函数内重复定义 `PROJECT_ROOT`
- 行 319、395 函数体里又 `PROJECT_ROOT = os.path.dirname(...)`，模块级已经有了
- **删掉局部赋值即可**，5 分钟

✅ 已修复 @ 2026-05-23
   - 删除 `log_generator` 和 `get_train_history` 中的局部 `PROJECT_ROOT` 赋值
   - 两处统一使用模块顶部已有的 `PROJECT_ROOT`
   - 验证：`inspect.getsource` 检查函数体内不再含局部赋值，模块级 `PROJECT_ROOT` 解析为 `/home/xiejianan/whisper-finetune-plus`

## [x] P3-2 `FileResponse` 的 import 位置
- `inference_controller.py:117` 在文件中间 import，挪到顶部
- 2 分钟

✅ 已修复 @ 2026-05-23
   - `from fastapi.responses import FileResponse` 挪到文件顶部，与其他 fastapi 相关 import 紧邻
   - 删除中间的散乱 import 行
   - 验证：`from inference_controller import FileResponse` 正常解析

## [x] P3-3 Windows num_workers 判断
- `finetune.py:63-64` 部署环境是 Linux/WSL，这段死代码可移除（或保留兼容）
- 2 分钟

✅ 已修复 @ 2026-05-23
   - 删除 `if platform.system() == "Windows": args.num_workers = 0` 整段
   - 同时移除孤立的 `import platform`
   - 验证：`python finetune.py --help` 正常打印所有参数说明

## [x] P3-4 ModelScope token 在异常 stack trace 里泄漏
- `finetune_controller.py:447-477` 把 `str(e)` 直接返回给前端
- 修：在 raise 前 `str(e).replace(ms_token, "***")`
- 5 分钟

✅ 已修复 @ 2026-05-23
   - 在 `publish_model` 的 `except` 块中先 `err_text = str(e).replace(ms_token, "***")`
   - 用脱敏后的 `err_text` 同时给 `print` 和 `HTTPException(detail=...)`
   - 验证：构造 `HTTPError 401: token=super_secret_modelscope_token_xyz` 字符串，
     脱敏后变为 `HTTPError 401: token=***`，原 token 完全消失

## [ ] P3-5 TFLite 导出独立子进程
- 现在 `tflite_export` 和训练在同一个 Python 进程，TF 和 torch 同时占显存
- 改为 `asyncio.create_subprocess_exec("python", "tflite_export.py", ...)`
- 1 小时

## [ ] P3-6 README / 部署文档同步更新
- 修完 P0-1（认证）后，前端登录流程截图需要更新
- 注明：生产模式不要用 `--reload`
- 1 小时

---

# 实施顺序建议

**第 1 个工作日（部署前必须）**
1. P0-1 认证系统（JWT + bcrypt）
2. P0-2 路径校验
3. P0-3 uploads 公开下载改为受保护路由
4. P0-4 CORS 配置
5. P1-2 GPU 锁 finally
6. ~~P1-4 argparse type=bool~~ — 已核实非 bug

**第 2 个工作日（多用户灰度前）**
7. P1-1 asyncio.Lock 替换 GPU_STATE
8. P1-3 部署文档分开发/生产
9. P1-5 DB 异步化（短期方案：to_thread 包一层）
10. P1-6 SSE 日志路径按模型名隔离
11. P2-1 SQLite 连接泄漏
12. P2-2 修改文本时清音频
13. P2-3 音频质量校验

**第 3 个工作日（首版发布前）**
14. P2-4 依赖钉版本
15. P2-5 数据集 seed
16. P2-6 evaluation.py
17. P2-7 重名实时校验
18. P2-8 自定义 Dialog

**有空就清** P3 全部

---

# 验收清单

每修完一项，请在该行的 `[ ]` 改为 `[x]`，并在条目末尾追加一行：

```
✅ 已修复 by <名字> @ <日期>
   - PR / commit: <链接>
   - 验证方式: <命令 / 操作步骤>
```
