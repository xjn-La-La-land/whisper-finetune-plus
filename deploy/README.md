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

> ⚠️ 启动 ① 前务必确认 `JWT_SECRET_KEY` 还在（`echo "$JWT_SECRET_KEY"` 非空）。全新实例若 home 被冲掉，`~/.bashrc` 里的那行也会丢，需要按 **1.0** 重新写入（用之前存好的同一个值，**不要重新生成**，否则又会把大家登出）。
