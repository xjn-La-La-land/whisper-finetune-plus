# 公网部署运行手册（后端 + Cloudflare 隧道）

固定入口 **https://app.carespeechai.cn** → Cloudflare 隧道 `carespeech` → 本机 `localhost:8000`。
更完整的背景见根目录 `TODO_PUBLIC_DEPLOY.md`。

## 前提（本文档假设已就绪）

- 已 `git clone` 并 `cd` 到 `whisper-finetune-plus` 根目录
- conda 环境已 `activate`（如 `whisper`）
- 基座模型已下载到 `whisper-base-models/`
- 数据已就位：`uploads/`、`data/tasks.db`

下面分两件事：**① 启动后端进程**、**② 连接 Cloudflare 隧道**。两个都是长驻进程，建议各开一个 `tmux` 窗口，方便随时回看日志。

---

## ① 启动后端

后端绑在 `127.0.0.1:8000`（只给本机的隧道连，不直接对外）。

```bash
tmux new -s backend
uvicorn main:app --host 127.0.0.1 --port 8000
```

看到 `Uvicorn running on http://127.0.0.1:8000` 即启动成功。
quick test（另开一个 tmux tab）：

```bash
curl -s http://127.0.0.1:8000/api/config     # 应返回 {"collect_only":false}
```

> 需要 `JWT_SECRET_KEY` / `MODELSCOPE_TOKEN` 等环境变量的话，在 `uvicorn` 之前 `export` 即可。

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

## 验证

浏览器打开 **https://app.carespeechai.cn**：能登录、采集页能录音、识别出字、微调 Loss 曲线实时刷新 → 上线成功。

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
