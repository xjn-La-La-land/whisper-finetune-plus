# 公网部署运行手册（后端 + Cloudflare 隧道）

固定入口 **https://app.carespeechai.cn** → Cloudflare 隧道 `carespeech` → 本机 `localhost:8000`。
更完整的背景见根目录 `TODO_PUBLIC_DEPLOY.md`。

## 前提（本文档假设已就绪）

- 已 `git clone` 并 `cd` 到 `whisper-finetune-plus` 根目录
- conda 环境已 `activate`（如 `whisper`）
- 基座模型已下载到 `whisper-base-models/`
- 数据已就位：`uploads/`、`data/tasks.db`

下面分两件事：**① 启动后端进程**、**② 连接 Cloudflare 隧道**。两个都是长驻进程，建议各开一个 `tmux` 窗口，方便随时回看日志。

> 本文 ①② 是 **Featurize 临时机**的 tmux 跑法（完整模式，含微调/识别）。若是**长期租用的纯 CPU 采集机**（Azure / 阿里云 等，只跑采集），请直接看文末「**长期常驻机部署（systemd）**」一节——用 systemd 托管，崩溃自动重启、重启服务器自动拉起。

---

## ① 启动后端

后端绑在 `127.0.0.1:8000`（只给本机的隧道连，不直接对外）。

### 1.0 设置 `JWT_SECRET_KEY`

`JWT_SECRET_KEY` 是登录态 token 的签名密钥，**必须显式设置且长期固定**。
不设的话，后端会在 `data/.jwt_secret` 自动生成一个随机密钥；而 Featurize 实例重启 /
换实例后该文件可能丢失，密钥一变，**所有已登录用户手里的 token 立刻失效、集体被踢回
登录页**（弹"登录已过期，请重新登录"）。固定一个 secret 后就不会再发生。

```bash
# 只生成一次，长期固定。把结果存进密码管理器，别入库、别每次重新生成。
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

把生成的值写进 `~/.bashrc`（每次开新 shell / 重启自动带上）：

```bash
echo 'export JWT_SECRET_KEY="把上面生成的值粘这里"' >> ~/.bashrc
source ~/.bashrc
[ -n "$JWT_SECRET_KEY" ] && echo "JWT_SECRET_KEY OK" || echo "❌ 仍为空"
```

> `MODELSCOPE_TOKEN` 等其它环境变量同理，在 `uvicorn` 之前 `export` 即可。

### 1.1 启动进程

```bash
tmux new -s backend
uvicorn main:app --host 127.0.0.1 --port 8000
```

看到 `Uvicorn running on http://127.0.0.1:8000` 即启动成功。
quick test（另开一个 tmux tab）：

```bash
curl -s http://127.0.0.1:8000/api/config     # 应返回 {"collect_only":false}
```

> 不想用 tmux 也可以：`nohup uvicorn main:app --host 127.0.0.1 --port 8000 > uvicorn.log 2>&1 &`

---

## ② 连接 Cloudflare 隧道

隧道 `carespeech` 已创建好（DNS `app.carespeechai.cn` 也已绑定），这台机只需跑一个连接器连上去。

### 2.1 准备 cloudflared 二进制（实例上没有才做）

装到 `~/.local/bin`（一般已在 PATH，之后直接用 `cloudflared` 调用）：

```bash
mkdir -p ~/.local/bin
curl -fsSL -o ~/.local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x ~/.local/bin/cloudflared
cloudflared --version    # 若提示找不到命令： export PATH="$HOME/.local/bin:$PATH"
```

### 2.2 隧道凭据 + 配置

策略：**凭据存在持久化的 `~/work/`，软链接到默认目录 `~/.cloudflared/`** 让 cloudflared 自动发现。
这样即使开个没用镜像的全新实例，凭据也还在 `~/work/`，只需重建软链接、无需重传。

**首次部署**（只做一次）：

```bash
mkdir -p ~/work/.cloudflared ~/.cloudflared
# 凭据：从本地笔记本上传到 work/（密钥，勿入库）
#   本地 ~/.cloudflared/df6be521-...json  →  这台机 ~/work/.cloudflared/df6be521-...json
# 软链接到默认目录（凭据 + 配置）
ln -sf ~/work/.cloudflared/df6be521-2389-407c-a4dc-d0748d8d7e50.json ~/.cloudflared/
ln -sf "$(pwd)/deploy/cloudflared-config.yml" ~/.cloudflared/config.yml
```

**以后每次只需检查存在**（缺了就重跑上面两条 `ln`，凭据无需重传）：

```bash
[ -f ~/.cloudflared/df6be521-2389-407c-a4dc-d0748d8d7e50.json ] \
  && echo "凭据 OK" || echo "软链接缺失：重建即可（凭据仍在 ~/work/.cloudflared/）"
ls -l ~/.cloudflared/    # 两条软链接应都指向实际文件
```

`~/.cloudflared/config.yml`（= 仓库 `deploy/cloudflared-config.yml`）内容：

```yaml
tunnel: df6be521-2389-407c-a4dc-d0748d8d7e50
ingress:
  - hostname: app.carespeechai.cn
    service: http://localhost:8000
  - service: http_status:404
```

### 2.3 启动隧道

```bash
tmux new -s tunnel
cloudflared tunnel --protocol http2 --no-autoupdate run
```

- 看到 `Registered tunnel connection` 即连接成功。
- `Ctrl-b` 再 `d` detach。
- `--protocol http2`：规避部分网络对 QUIC(UDP) 的干扰；想用默认 QUIC 去掉即可。

---

## ③ 浏览器端 WASM 本地推理（COOP/COEP 透传，**上线必查**）

本平台**默认**在用户浏览器里用 whisper.cpp 的 WASM 做识别（音频不离开本机、不占服务端 GPU、多用户可并发）。浏览器不支持时才回落到服务端 `/api/recognition`（受 GPU 锁约束）。要让多线程 WASM 跑起来，页面必须处于「跨源隔离」(`crossOriginIsolated === true`) 状态，这依赖两个响应头：

- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Embedder-Policy: require-corp`

这两个头由后端 `main.py` 的全局中间件统一加上。**风险点：反向代理 / 隧道有可能把它们剥掉。** 本平台走 Cloudflare 具名隧道，需实测确认它**原样透传**这两个头——否则 `crossOriginIsolated` 会是 `false`，前端能力探测判失败，**所有用户被迫回落服务端 GPU、WASM 方案在线上等于没上。**

**验证（浏览器开 https://app.carespeechai.cn，F12 控制台敲一行）：**

```js
crossOriginIsolated            // 必须为 true
```

- `true` → 透传正常。再到识别页确认「推理方式」默认停在「⚡ 本地浏览器」、选个模型下载后能识别出字即可。
- `false` → 头被剥了。排查：① 后端直连确认头在 —— `curl -I http://127.0.0.1:8000/ | grep -i cross-origin` 应看到 COOP/COEP 两行；② 若后端有、线上没有，就是隧道/边缘剥掉了。退路见 `TODO_WHISPER_CPP_WASM.md` 风险登记：客户端 `coi-serviceworker` 注入，或改用单线程 WASM 兜底（性能 -30%）。

> **模型来源**：WASM 产物（`static/wasm/`）已随仓库提交，服务机**无需装 emsdk / 重编**。但「没微调过的用户也能本地识别」需要基座的 ggml 量化版——在能跑 `whisper-quantize` 的机器上执行 `python ggml_export.py --base-models`（根 README 第 4 / 4b 步）；微调模型的 ggml 在训练成功后自动导出。
> 纯采集机（`COLLECT_ONLY=1`）不做识别，本节不适用（COOP/COEP 头仍会带上，无副作用）。

---

## 验证

浏览器打开 **https://app.carespeechai.cn**：能登录、采集页能录音、识别出字、微调 Loss 曲线实时刷新 → 上线成功。
**本地 WASM 推理是否可用见上节「③」**（控制台 `crossOriginIsolated` 应为 `true`）。

排查命令：

```bash
tmux attach -t backend            # 看后端日志（或 tail -f uvicorn.log）
tmux attach -t tunnel             # 看隧道日志
cloudflared tunnel info carespeech   # 看隧道当前连接数
```

---

## 关机 / 重启实例后

核心就是重跑 **①** 和 **② 的 2.3**（启动后端 + 启动隧道）。是否要补 2.1/2.2 取决于怎么开机：

- **用镜像开机**：二进制（`~/.local/bin`）和软链接（`~/.cloudflared`）都在镜像里 → 直接跑 ① 和 2.3。
- **没用镜像的全新实例**：凭据仍在 `~/work/`（持久不丢），但 home 里的二进制和软链接没了 → 补做 **2.1**（下载）+ **2.2 的两条 `ln`**（重建软链接，无需重传凭据），再跑 ① 和 2.3。

> ⚠️ 启动 ① 前务必确认 `JWT_SECRET_KEY` 还在（`echo "$JWT_SECRET_KEY"` 非空）。全新实例若 home 被冲掉，`~/.bashrc` 里的那行也会丢，需要按 **1.0** 重新写入（用之前存好的同一个值，**不要重新生成**，否则又会把大家登出）。

---

## 长期常驻机部署（systemd · Azure / 阿里云 等）

上面 ①② 是 Featurize **临时机**的 tmux 跑法。若是**长期租用的纯 CPU 机**（Azure for Students / 阿里云轻量 等），只跑「音频采集」（`COLLECT_ONLY=1`，不加载 torch / 识别 / 微调），用 systemd 托管更稳：**崩溃自动重启、重启服务器自动拉起**，彻底告别 tmux。

unit 文件都在 `deploy/systemd/`：

| 文件 | 作用 |
| --- | --- |
| `carespeech-collect.service` | uvicorn 采集后端（`COLLECT_ONLY=1`），崩溃自动重启 |
| `carespeech-tunnel.service` | cloudflared 隧道（`--protocol http2`），断线自动重连，后端起来后再连 |
| `collect.env.example` | `JWT_SECRET_KEY` 模板（机密单独存放，勿入库） |

**前提**：仓库已 clone；conda 环境（如 `collect`）已用 `requirements-collect.txt` 装好；已 `sudo apt install ffmpeg`；cloudflared 二进制 + 凭据已按 **② 的 2.1 / 2.2** 就位。

### S1 改 unit 里的路径

先查出 uvicorn 绝对路径：

```bash
conda activate collect && which uvicorn
# 例：/home/azureuser/miniconda3/envs/collect/bin/uvicorn
```

两个 `.service` 里凡是带 `>>>` 注释的行都要按本机改：`User=` / `Group=`、`WorkingDirectory=`（仓库根）、`ExecStart=` 里的 uvicorn 绝对路径，以及 tunnel 单元里的 cloudflared 路径 + `--config` 路径。默认值是 `ubuntu` + `~/miniconda3`，用户名不同就批量换：

```bash
cd ~/whisper-finetune-plus/deploy/systemd
sed -i 's#/home/ubuntu#/home/azureuser#g' carespeech-*.service     # 按实际用户名替换
```

### S2 安装到系统目录

```bash
cd ~/whisper-finetune-plus

# 1) 机密：JWT_SECRET_KEY（放 /etc，root 拥有，chmod 600）
sudo mkdir -p /etc/carespeech
sudo cp deploy/systemd/collect.env.example /etc/carespeech/collect.env
sudo nano /etc/carespeech/collect.env        # 填入长期固定的 JWT_SECRET_KEY
sudo chmod 600 /etc/carespeech/collect.env

# 2) 两个 unit
sudo cp deploy/systemd/carespeech-collect.service /etc/systemd/system/
sudo cp deploy/systemd/carespeech-tunnel.service  /etc/systemd/system/

# 3) 让 systemd 重新读配置
sudo systemctl daemon-reload
```

> 还没固定密钥就先 `python -c "import secrets; print(secrets.token_urlsafe(32))"` 生成一次，存进密码管理器长期复用。**从 Featurize 迁数据时，想让已登录用户不被踢下线，就在这里填和 Featurize 一模一样的值。**

### S3 开机自启 + 立即启动

```bash
sudo systemctl enable --now carespeech-collect.service
sudo systemctl enable --now carespeech-tunnel.service
```

`enable` = 开机自动拉起，`--now` = 顺便现在就启动。

### S4 验证

```bash
systemctl status carespeech-collect carespeech-tunnel --no-pager   # 均为 active (running)
curl -s http://127.0.0.1:8000/api/config                           # {"collect_only":true}
journalctl -u carespeech-tunnel -n 30 --no-pager                   # 出现 Registered tunnel connection
```

浏览器开 **https://app.carespeechai.cn**：能登录、能录音上传即成功。

### 日常运维速查

```bash
journalctl -u carespeech-collect -f        # 实时日志（-f 跟随，Ctrl-C 退出）
journalctl -u carespeech-tunnel  -f

sudo systemctl restart carespeech-collect  # git pull 改了代码后，重启后端即可（隧道不用动）
sudo systemctl stop  carespeech-collect carespeech-tunnel   # 停
sudo systemctl start carespeech-collect carespeech-tunnel   # 起
sudo systemctl disable carespeech-collect carespeech-tunnel # 取消开机自启

# 改了 .service 文件本身后，必须先 daemon-reload 再 restart 才生效
sudo systemctl daemon-reload && sudo systemctl restart carespeech-collect
```

**重启服务器后**：什么都不用做——两个服务已 `enable`，开机按「后端 → 隧道」顺序自动拉起。

### 几个要点

- `COLLECT_ONLY=1` 已写死在 collect 单元里，这台机即便误装了 torch 也不会去加载，稳。
- 用 `User=xxx` 跑时 systemd 会把 `$HOME` 设为该用户家目录，cloudflared 能在 `~/.cloudflared/` 自动找到 `<UUID>.json`。万一日志报找不到凭据，在 `~/.cloudflared/config.yml` 里补一行 `credentials-file: /home/<user>/.cloudflared/df6be521-2389-407c-a4dc-d0748d8d7e50.json`。
- 采集用 SQLite，保持**单 uvicorn 进程**（不加 `--workers`）最稳，也最省小机器内存。
