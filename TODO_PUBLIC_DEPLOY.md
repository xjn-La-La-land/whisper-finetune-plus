# 网页端公网部署 To-Do（采集 + 推理）

> 本文档汇总把本平台（音频采集 + 在线推理）从 **cpolar 临时隧道** 迁移到 **稳定固定地址 +
> 不限速** 公网入口的实施路径，替换掉 cpolar「链接每次变 + 免费版 ~1Mbps 限速」两大痛点。
>
> 按 **公共前置 → Path 1(Cloudflare Tunnel，首选) / Path 2(frp+VPS+Caddy，备选) → 验证收尾** 推进，
> 每项包含：**任务目标**、**相关文件 / 命令**、**实施步骤**、**预估工作量**、**完成状态**。
>
> 两条路 **二选一即可**。**首选 Path 1（Cloudflare Tunnel）**：零额外服务器、当天可上线。
> Path 2 需要再订阅一台国内 VPS，仅当 Path 1 对国内用户实测太慢时再用。

---

## 关键结论（为什么 Path 1 首选）

1. **部署机是 Featurize GPU 租赁实例，不是本机笔记本**。本机（Windows + WSL2 + RTX 5060）
   **只用于本地调试**，跑通后才上 Featurize。

2. **Featurize 没有公网 IP，也不能自由放行入站端口**。官方对外暴露端口只有
   `featurize port export`，实测限制（[官方文档](https://docs.featurize.cn/docs/manual/port-exporting)）：
   - 给的是 `http://workspace.featurize.cn:<平台分配端口>` —— **HTTP，不是 HTTPS**
   - 端口由平台随机分配，**实例重建会变**（和 cpolar 一样不固定）
   - 服务必须绑 `0.0.0.0`，每台最多导出 10 个端口

   → 既不固定、又无 HTTPS（采集页 `getUserMedia` 录音必须 HTTPS），**所以必须在实例里跑出站隧道**。

3. **Path 1（Cloudflare 具名隧道）零额外成本、当天上线**：`cloudflared` 在 Featurize 出站连
   Cloudflare，给固定域名 `https://app.carespeechai.cn` + 自动 HTTPS，**实例重建/换机地址不变**，
   不用再买、配、维护任何服务器。代价：国内访问走海外边缘、延迟偏高，但功能完整、带宽不限速，
   对当前少量采集用户够用。

4. **Path 2（frp + 国内 VPS + Caddy）是备选**：好处是全程国内、延迟更低；代价是要**再订阅一台
   国内 VPS**（原 `43.143.17.185` 已退租），还要自己搭 frps + Caddy。**仅当 Path 1 对国内用户
   实测太慢时才上。**

---

## 背景与硬约束（动手前必读）

1. **GPU 必须在 Featurize 实例**：`inference_controller.py` / `finetune.py` 依赖 GPU，
   推理/微调只能在 Featurize 实例里跑，公网入口必须能连到它。

2. **浏览器录音必须 HTTPS**：采集用 `getUserMedia` 访问麦克风，现代浏览器**只在 HTTPS（或
   localhost）下放行**。任何方案都必须终结 TLS —— 裸 `http://host:端口`（含 Featurize 原生端口导出）
   会让采集功能直接失效。

3. **隧道是出站连接，契合 Featurize**：cloudflared / frpc 都从实例内部主动向外连，**不需要公网 IP
   或入站端口**——这正是无公网 IP 的 Featurize 唯一能走的方式。

4. **Featurize 是临时容器，无 systemd**：实例归还即销毁，只有同步盘 `/home/featurize/work`
   持久（默认 30GB）。**没有 systemd**，后台进程用 `tmux` / `nohup` + 日志文件
   （[后台运行](https://docs.featurize.cn/docs/manual/daemon)）。需跨实例存活的（项目代码、隧道二进制
   与凭据）放同步盘 / 靠镜像；实例每次启动后手动重跑后端 + 隧道（见 `deploy/README.md`）。

### 环境与资产速查

| 角色 | 值 |
| --- | --- |
| **部署机（GPU）** | Featurize 租赁实例（容器）。用户 `featurize`，**无公网 IP / 无 systemd / 实例临时**；持久化靠同步盘 `/home/featurize/work` + 镜像 |
| **稳定公网入口（Path 1）** | Cloudflare 具名隧道 `carespeech` → 固定 `https://app.carespeechai.cn`（cloudflared 在 Featurize 出站连，实例重建不变）。运行手册见 `deploy/README.md` |
| **本地调试机（仅开发）** | Windows 笔记本 + WSL2 Ubuntu 24.04 + RTX 5060，先在 `127.0.0.1:8000` 跑通流程再上 Featurize |
| 域名 | `carespeechai.cn`（腾讯云/DNSPod 购买，与模型仓库 CareSpeech-ASR 品牌对齐）；平台入口用子域名 `app.carespeechai.cn`，apex 根域留作介绍页 |
| 后端启动 | `uvicorn main:app --host 127.0.0.1 --port 8000`（隧道在同一实例内连 localhost，无需 0.0.0.0） |
| Path 2 才需要的 VPS | 一台国内 VPS（原 `43.143.17.185` 已退租，用 Path 2 需新租）；`sync_from_cloud.sh` 里有旧机引用 |

---

## 阶段速览

| 阶段 | 项数 | 进入下一阶段的硬条件 |
| --- | --- | --- |
| Phase A 公共前置 | 5 | 域名实名通过 + 本地调试机 `127.0.0.1:8000` 跑通 + Featurize 实例就绪 + 前端依赖本地化 |
| **Phase B  Path 1（首选）：Cloudflare Tunnel** | 4 | `https://app.carespeechai.cn` 能访问、能录音、能推理（固定地址、自动 HTTPS） |
| Phase C Path 2（备选）：frp + VPS + Caddy | 7 | 仅当 Path 1 国内太慢时；`https://app.carespeechai.cn:8443` 能访问、能录音、能推理（全程国内） |
| Phase D 验证 / 收尾 | 3 | 实测带宽优于 cpolar，确定主用路径，cpolar 退役 |

---

# 🟢 Phase A 公共前置（两条路都要做）

## [ ] A-1 购买域名并完成实名

**任务目标**
- 拿到 `carespeechai.cn`

**相关命令 / 链接**
- 选购页：https://buy.cloud.tencent.com/domain
- DNSPod 控制台：https://dnspod.cloud.tencent.com/

**实施步骤**
1. 搜索 `carespeechai` → 选 `.cn` → 微信/支付宝下单。
2. **立刻提交实名认证**（个人即可）——`.cn` 由 CNNIC 强制实名，**未实名几天内会被停到 ServerHold、域名不解析**，务必尽早提交。
3. 跳过 SSL 证书 / 备案 / 建站 等 upsell（Path 1 由 Cloudflare 自动出 HTTPS）。

**预估工作量**：10 分钟操作 + 几小时~1 天审核
**完成状态**：✅ 已完成（已购 `carespeechai.cn` 并实名通过）

---

## [ ] A-2 后端跑通（先本地、后 Featurize）

**任务目标**
- 确认完整模式（采集+推理+微调）可启动；并把 Featurize 上的环境/代码持久化好

**相关文件**
- `main.py`（`COLLECT_ONLY` 开关、CORS）、`requirements.txt`、`env.yaml`
- Featurize 持久化：[环境持久化](https://docs.featurize.cn/docs/manual/persistance)、[同步盘](https://docs.featurize.cn/docs/manual/work)

**实施步骤**
1. **本地调试机**：`conda activate whisper` → `uvicorn main:app --host 127.0.0.1 --port 8000`，
   浏览器开 `http://127.0.0.1:8000` 自测登录/采集/推理（localhost 下麦克风可用）。
2. **Featurize 实例**：把项目放同步盘 `/home/featurize/work/whisper-finetune-plus`；
   conda 环境 `whisper` 装在 home 默认位置，由镜像保存（不在 work/；镜像里已有就直接 activate）：
   ```bash
   conda create -n whisper python=3.11 -y      # 仅首次 / 新实例无镜像时需要
   conda activate whisper
   pip install -r requirements.txt   # torch 按实例 CUDA 单独装
   ```
3. Featurize 上跑后端（无 systemd，用 tmux 常驻；详细步骤见 `deploy/README.md` ①）：
   ```bash
   cd /home/featurize/work/whisper-finetune-plus
   tmux new -s backend
   uvicorn main:app --host 127.0.0.1 --port 8000     # Ctrl-b d 退出，进程继续后台跑
   ```

**预估工作量**：本地 10 分钟 + Featurize 30 分钟（含装环境）
**完成状态**：本地 ✅ 已跑通（UI + 采集 + 微调 Loss 曲线均正常，并已导入「小满」35 条数据）；**Featurize 侧待做**（装环境 + 把代码/数据搬上去 + 后台跑后端）。

---

## [ ] A-3 确认前端 API 走相对路径（同源）

**任务目标**
- 确保换入口后前端不需要改 URL：页面与 API 同源，`ALLOWED_ORIGINS` 可不配

**相关文件**
- `static/js/app.js`、`static/js/audio_collector.js`、`static/js/inference_panel.js`
- `main.py:24-38`（CORS 说明：同源不受影响）

**实施步骤**
1. 抽查前端 `fetch(...)` 是否用 `/api/...` 相对路径（不是写死的 `http://localhost:8000`）。
2. 若发现写死的绝对地址，改成相对路径。
3. 同源场景 `ALLOWED_ORIGINS` 留空即可；安卓 app 是原生请求不受 CORS 限制。

**预估工作量**：15 分钟
**完成状态**：✅ 已核对（`fetch` 全为 `/api/...` 相对路径；`static/js/api.js:73` 的 `sseUrl` 用 `new URL(path, location.origin)` 返回相对路径，SSE 同源）。换域名前端零改动。

---

## [ ] A-4 Featurize 实例特性与持久化认知

**任务目标**
- 理解部署机的临时性，避免实例重建后入口失效

**相关文件 / 链接**
- [后台运行](https://docs.featurize.cn/docs/manual/daemon)、[镜像](https://docs.featurize.cn/docs/manual/image)

**实施步骤**
1. 凡是要跨实例存活的（项目代码、cloudflared 凭据、conda env）放同步盘 `/home/featurize/work`，
   或靠镜像保存（conda env 在 home，随镜像走）。
2. 后台服务用 `tmux`（推荐，可回看）或 `nohup ... &` + 日志文件，**不要**指望 systemd。
3. 实例每次启动后需重跑后端 + 隧道（各开一个 tmux 手动跑，步骤见 `deploy/README.md`），
   或把配好的环境**保存为镜像**，下次直接用该镜像开实例。
4. 确认实例可出站访问外网（下载模型用得到，cloudflared 也靠它连出去）。

**预估工作量**：10 分钟（认知项）
**完成状态**：未开始

---

## [ ] A-5 前端依赖本地化（国内 CDN 问题，影响国内可用性）

**任务目标**
- 让页面不依赖海外 CDN，否则国内用户可能白屏/加载慢

**相关文件**
- `templates/index.html:8-10`（Vue / Tailwind / ECharts 三个海外 CDN）
- `main.py:48`（已 `app.mount("/static", ...)`，本地化后同源加载）

**问题**
- `https://unpkg.com/vue@3/...`、`https://cdn.tailwindcss.com`、`https://cdn.jsdelivr.net/...echarts...`
  三者都在境外；Vue 拉不到则整个界面无法启动（白屏）。

**实施步骤（二选一，推荐 A）**
- **方案 A：库文件本地化（最稳，零外部依赖）**
  1. 下载 `vue.global.js`、`echarts.min.js` 到 `static/vendor/`，把 index.html 的两个 `src` 改成
     `/static/vendor/xxx`。
  2. Tailwind 是 Play CDN（浏览器内 JIT，本不适合生产）：用 Tailwind CLI 预编译出一份静态 CSS 放
     `static/vendor/tailwind.css` 再引入；或先临时换国内镜像（见方案 B）。
- **方案 B：换国内 CDN 镜像（改动最小）**
  - 把三个 URL 换成 `bootcdn.cn` / `staticfile.org` 等国内镜像。省事但仍是外部依赖。

**预估工作量**：方案 A 约 40 分钟（Tailwind 预编译稍费时），方案 B 约 10 分钟
**完成状态**：✅ 已完成（方案 A）。Vue 3.5.35(prod) + ECharts 5.5.0 下载到 `static/vendor/`；
Tailwind 用 v3 CLI 预编译为 `static/vendor/tailwind.css`（配置 `tailwind.config.js` + `tailwind.input.css`，
扫描 html+js，改 class 后按 config 注释里的命令重跑）；`templates/index.html` 三处 CDN 引用已改为本地同源。
全站无残留海外 CDN。**已验证（用户本地实测）**：页面样式正常、微调 Loss 曲线（ECharts）正常显示。

---

# 🔵 Phase B  Path 1（首选）：Cloudflare 具名隧道

> 固定入口 `https://app.carespeechai.cn` → Cloudflare 隧道 `carespeech` → Featurize 上 `localhost:8000`。
> cloudflared 在 Featurize 出站连 Cloudflare，**无需公网 IP、无额外服务器**；实例重建地址不变。
> **完整运行步骤见 [`deploy/README.md`](deploy/README.md)**，本节只列里程碑。

## [x] B-1 域名接入 Cloudflare

**任务目标**：把 `carespeechai.cn` 的 DNS 托管到 Cloudflare（免费计划），拿到具名隧道能力。

**实施步骤**
1. dash.cloudflare.com → Add a site → `carespeechai.cn`（Free）。
2. DNSPod 控制台把 nameserver 改成 Cloudflare 给的两个（`lloyd` / `nina .ns.cloudflare.com`）。
3. 等 Cloudflare 显示 **Active**。

**预估工作量**：10 分钟 + 生效等待
**完成状态**：✅ 已完成（域名 Active）

---

## [x] B-2 创建具名隧道 + 绑定域名（本地 cloudflared CLI）

**任务目标**：建好永久隧道身份 + DNS 路由，供 Featurize 实例随时连入。

**实施步骤**（已在本地笔记本执行）
```bash
cloudflared tunnel login                                   # 浏览器授权 carespeechai.cn
cloudflared tunnel create carespeech                        # 生成凭据 <UUID>.json
cloudflared tunnel route dns carespeech app.carespeechai.cn # 自动建 CNAME
```

**完成状态**：✅ 已完成。隧道 `carespeech`（ID `df6be521-2389-407c-a4dc-d0748d8d7e50`）已建，
`app.carespeechai.cn` CNAME 已绑定；凭据 `df6be521-...json` 在本机 `~/.cloudflared/`（密钥）。

---

## [ ] B-3 在 Featurize 实例跑通后端 + 隧道

**任务目标**：实例上启动后端，并跑 cloudflared 连入具名隧道，`https://app.carespeechai.cn` 上线。

**实施步骤**：**完整照 [`deploy/README.md`](deploy/README.md) 走**，要点：
1. 前置：仓库已 clone、`conda activate whisper`、基座模型已下、数据（`uploads/`+`data/tasks.db`）已传。
2. cloudflared 装到 `~/.local/bin`。
3. 隧道凭据放 `~/work/.cloudflared/` 并软链接到 `~/.cloudflared/`（凭据 + `config.yml`），首次做一次。
4. `tmux` 各跑一个：后端 `uvicorn ... 127.0.0.1:8000`；隧道 `cloudflared tunnel --protocol http2 --no-autoupdate run`。

**预估工作量**：首次 20 分钟，之后每次开机 2 分钟
**完成状态**：未开始（待 Featurize 实例就绪）

---

## [ ] B-4 端到端验收（Path 1）

**实施步骤**
1. 浏览器开 `https://app.carespeechai.cn`，登录（`小满 / 0000`）。
2. 采集页**录音**（HTTPS 下麦克风可用）、识别页出字、微调 **Loss SSE** 实时刷新。

**验收硬条件**：录音 + 推理 + 微调日志流全部正常。
**预估工作量**：15 分钟
**完成状态**：未开始

---

# 🟠 Phase C  Path 2（备选）：frp + 国内 VPS + Caddy（全程国内，地址固定）

> **仅当 Path 1 对国内用户实测太慢时才上**；需先**新租一台国内 VPS**（原 `43.143.17.185` 已退租，
> 下面命令里的该 IP 请替换成你的新 VPS IP）。
> 架构：`用户 → Caddy(VPS:8443, 自动HTTPS) → frps(VPS) → frpc(Featurize实例) → uvicorn(127.0.0.1:8000)`
> 稳定入口在 VPS，Featurize 实例只出站连 frps；实例重建公网地址不变。
> 用 **高端口 8443** 避开腾讯云对未备案域名 80/443 的拦截；Caddy 用 **DNS-01** 自动签证书（不依赖 80 端口）。

## [ ] C-1 VPS 安全组放行端口

**实施步骤**
1. 腾讯云控制台 → 该轻量服务器 → 防火墙/安全组。
2. 放行入站：`7000`（frp 控制端口）、`8443`（Caddy HTTPS）。
3. **不要**放行 `8080`（frp 回源端口，仅 VPS 本机环回用）。
4. 确认 Featurize 实例能出站访问 `<VPS_IP>:7000`（一般无出站限制）。

**预估工作量**：5 分钟
**完成状态**：未开始

---

## [ ] C-2 VPS 上部署 frps（服务端，有 systemd）

**相关文件**：VPS `~/frp/frps.toml`
```toml
bindPort = 7000
# 让 frp 暴露的端口只绑在本机环回，仅 Caddy 能访问（不对公网直开）
proxyBindAddr = "127.0.0.1"
auth.method = "token"
auth.token = "<改成一个长随机串>"
```

**实施步骤**
```bash
# VPS 上
cd ~ && wget https://github.com/fatedier/frp/releases/latest/download/frp_*_linux_amd64.tar.gz
tar -xf frp_*_linux_amd64.tar.gz && mv frp_*_linux_amd64 frp
# 写上面的 frps.toml，然后注册 systemd（VPS 有 systemd）
sudo tee /etc/systemd/system/frps.service >/dev/null <<'EOF'
[Unit]
Description=frps
After=network-online.target
[Service]
ExecStart=/home/ubuntu/frp/frps -c /home/ubuntu/frp/frps.toml
Restart=on-failure
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now frps
```

**预估工作量**：20 分钟
**完成状态**：未开始

---

## [ ] C-3 Featurize 实例上部署 frpc（客户端，无 systemd）

**相关文件**：Featurize `/home/featurize/work/frp/frpc.toml`
```toml
serverAddr = "<VPS_IP>"
serverPort = 7000
auth.method = "token"
auth.token = "<与 frps 相同的随机串>"

[[proxies]]
name = "whisper-web"
type = "tcp"
localIP = "127.0.0.1"
localPort = 8000      # 实例内的 uvicorn
remotePort = 8080     # VPS 上环回端口，给 Caddy 回源
```

**实施步骤**
```bash
# Featurize 实例上：二进制 + 配置放同步盘，重建实例不丢
mkdir -p /home/featurize/work/frp && cd /home/featurize/work/frp
wget https://github.com/fatedier/frp/releases/latest/download/frp_*_linux_amd64.tar.gz
tar -xf frp_*_linux_amd64.tar.gz --strip-components=1
# 写上面的 frpc.toml，然后用 tmux 常驻（无 systemd），实例每次启动后重跑
tmux new -s frpc
/home/featurize/work/frp/frpc -c /home/featurize/work/frp/frpc.toml   # Ctrl-b d 退出
```

**预估工作量**：15 分钟
**完成状态**：未开始

---

## [ ] C-4 DNS 解析指向 VPS

**实施步骤**
1. 域名 NS 已在 Cloudflare（Path 1 接入时切的）→ 在 **Cloudflare DNS** 加一条 **A 记录**
   `app.carespeechai.cn` → `<VPS_IP>`，**Proxy status 设为 DNS only（灰云）**。
2. 与 Path 1 的隧道 CNAME 冲突：两条路同一 hostname 只能留一条记录，**二选一**（用 Path 2 时删掉 Path 1 的 CNAME）。

**预估工作量**：5 分钟
**完成状态**：未开始

---

## [ ] C-5 VPS 上 Caddy 自动 HTTPS（DNS-01 + 腾讯云/DNSPod 插件）

**任务目标**
- Caddy 在 `:8443` 终结 HTTPS、反代到 frp 回源端口；证书用 DNS-01 自动签（不需要 80 端口，避开备案）

**实施步骤**
1. 在 DNSPod / 腾讯云创建 API 密钥（SecretId + SecretKey），供 Caddy 做 DNS-01。
   （注意：域名 DNS 现在在 Cloudflare，DNS-01 要让 Caddy 改的是**当前生效的 DNS 托管方**——
   若 NS 在 Cloudflare 则用 `caddy-dns/cloudflare` 插件 + Cloudflare API Token；若改回 DNSPod 则用腾讯云插件。）
2. 用 xcaddy 构建带对应 DNS 插件的 Caddy（官方二进制不含 DNS 插件）：
   ```bash
   # VPS 上（示例：DNS 在 Cloudflare）
   go install github.com/caddyserver/xcaddy/cmd/xcaddy@latest
   xcaddy build --with github.com/caddy-dns/cloudflare
   ```
3. 写 `Caddyfile`：
   ```
   app.carespeechai.cn:8443 {
       tls {
           dns cloudflare <CLOUDFLARE_API_TOKEN>
       }
       reverse_proxy 127.0.0.1:8080
       request_body { max_size 200MB }
   }
   ```
4. 注册 systemd 跑 Caddy，确认自动签发证书成功。

**预估工作量**：40 分钟（含 xcaddy 构建）
**完成状态**：未开始

---

## [ ] C-6 端到端验收（Path 2）

**实施步骤**
1. 浏览器开 `https://app.carespeechai.cn:8443`，登录。
2. 测试**录音**（验证 HTTPS 下麦克风可用）、**推理**、**微调 Loss SSE**。
3. 确认证书有效（无浏览器告警）、链路全程国内。

**验收硬条件**：录音 + 推理 + 微调日志流全部正常，且证书可信。
**预估工作量**：15 分钟
**完成状态**：未开始

---

## [ ] C-7（可选）采集态进一步下沉到 VPS

**任务目标**
- 若采集上传量大，可把 `COLLECT_ONLY=1` 轻量服务直接跑在 VPS（满 VPS 带宽、免穿 frp），
  推理仍走 Featurize GPU；数据用 `sync_from_cloud.sh` 拉回训练机

**相关文件**
- `main.py`（`COLLECT_ONLY` 分支）、`sync_from_cloud.sh`、`sync_uploads_to_db.py`

**实施步骤**
1. VPS 上 `COLLECT_ONLY=1 uvicorn main:app ...`（不装 torch，轻量）。
2. Caddy 把采集相关路由指到 VPS 本地采集服务、推理相关指到 frp 回源（或两套子域名）。
3. 定期 `bash sync_from_cloud.sh` 把 `uploads/` 同步回训练机并回填 `data/tasks.db`
   （注意脚本里 `LOCAL_DIR` 应指向 Featurize 同步盘下的项目路径）。

**预估工作量**：1~2 小时（架构增量，按需做）
**完成状态**：未开始（可选）

---

# 🟣 Phase D 验证 / 收尾

## [ ] D-1 实测带宽与延迟

**实施步骤**
1. Path 1 上线后，用真实采集者网络测大文件上传耗时 + 页面打开速度，与 cpolar 旧值对比。
2. 若国内访问抖/慢到影响使用 → 再考虑 Path 2（需新租国内 VPS，按 Phase C 搭）。

**预估工作量**：30 分钟
**完成状态**：未开始

---

## [ ] D-2 固定主用入口并更新文档

**实施步骤**
1. 确定主用路径（**默认 Path 1**），更新 `README.md`「远程网页访问」一节（替换 cpolar 说明）。
2. 安卓 app「开发者配置」里的服务器地址改成 `https://app.carespeechai.cn`（Path 1；若改用 Path 2 则带 `:8443`）。
3. 把每次开机的启动步骤（后端 + 隧道，见 `deploy/README.md`）固化为习惯，或保存为镜像减少重复准备。

**预估工作量**：20 分钟
**完成状态**：未开始

---

## [ ] D-3 退役 cpolar

**实施步骤**
1. 新入口稳定运行数日后，停用 cpolar 进程与开机项。
2. 从文档/脚本移除 cpolar 相关说明。

**预估工作量**：10 分钟
**完成状态**：未开始
