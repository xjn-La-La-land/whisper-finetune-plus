# 登录页「梵高星空涂鸦风」动态背景 —— 实现记录（已完成 ✅）

> 把登录页从「极光光斑」原型升级为**梵高《星空》气质的涂鸦风动态背景**：深蓝乱线夜空 + 暖黄漩涡月亮/星星
> + 蓝色打卷云朵 + 底部小人/长椅剪影。**已上线**，全部实现在 [`templates/index.html`](templates/index.html)
> 的 `<style>` + `div.login-stage`，外加 [`static/js/login_parallax.js`](static/js/login_parallax.js)。
>
> **路线（最终）**：原创插画打底（用户从开源社区取图、版权 OK，Gemini 扩横屏 + 分层）+ 纯 CSS 动效；
> 鼠标/陀螺仪视差用一小段 vanilla JS。零外部 CDN、零动画库。

---

## 最终架构

**4 张图层图（自托管，透明处为真 alpha）**

| 资产 | 内容 | 体积 | 缓存 |
|---|---|---|---|
| `static/img/login/sky.webp` | 纯星空底（无月/云/人，不透明） | 266KB | `?v=3` |
| `static/img/login/clouds.webp` | 蓝色涂鸦云 + 暖黄云脊（透明） | 437KB | `?v=4` |
| `static/img/login/foreground.webp` | 小人/长椅/地面剪影（透明，盖在云上） | 93KB | `?v=2` |
| `static/img/login/moon.webp` | 金色漩涡月亮（透明，紧裁） | 37KB | `?v=2` |

合计 **836KB**（预算 < 1MB ✅）。月心在画布 **49% / 17.8%**、占宽 **12.1%**。

**合成图层栈（从后到前）**

```
.login-stage (fixed inset-0, overflow-hidden; L0 兜底深蓝渐变)
└─ .cover-frame  (居中容器 = 按 cover 裁剪的画布, width:max(100vw,100vh*1.7917))
   ├─ .layer-sky          z0  纯星空底
   ├─ .layer-clouds       z?  云（透明）—— clouddrift 横移视差(scale 1.08 留余量防漏边)
   ├─ .layer-fg           z?  前景（透明）—— 静止，盖在云上
   ├─ .moon-glow          z?  月亮处暖色光晕 —— glowpulse 呼吸
   └─ .moon (img)         z?  月亮（透明）—— moonspin 90s 原地自转
.layer-stars   z2   ~100 颗 .tw 发光星(含青色) —— twinkle2 错相位闪烁(screen 混合)；上半天空、避开卡片
.layer-atmos   z6   四周暗角，收拢到中部卡片
.login-card    z10  深蓝磨砂玻璃卡(rgba .38 + blur 18px)、暖金 CTA、透明描边输入框、麦克风暖金徽章
```

**动效（纯 CSS keyframes，登录后随 `v-if` 移除自动停）**

- `moonspin` 90s 月亮原地自转（`transform-origin:center` + translate 居中）
- `glowpulse` 6s 月晕呼吸；`clouddrift` 28s 云横移视差；`twinkle2` 2.6–5s 星星错相位闪烁
- `cardIn` 卡片入场；`layerFadeIn` 图层淡入

**鼠标/陀螺仪极轻视差（P4）** —— [`static/js/login_parallax.js`](static/js/login_parallax.js)（vanilla）

- 指针/倾斜 → 给 `.login-stage` 写 `--mx/--my`（-1..1，带缓动）；各层用**独立 `translate` 属性**
  按深度偏移（与 transform 动画叠加）：sky 9px / 月亮 14px / 云 24px / 星 11px；**前景不动**（脚下地面）。
- 生命周期：监听挂 `window`，事件里查 `.login-stage` 是否在 DOM，登录后即空转并停 rAF；
  `prefers-reduced-motion` 下整段不启用。

---

## 关键技术点（踩过的坑 / 解法）

1. **Gemini 的"透明"= 画了一张灰棋盘格**（PNG alpha 全 255，不是真透明；非 PNG 限制）。
   → 改为让 Gemini 把背景画成**纯色**（云/前景用品红 `#FF00FF`、月亮在暗场景里），本地按色/亮度抠：
   云/前景 chroma-key 品红 + despill 去紫边；月亮按"亮金"权重 + 径向收口；sky 不透明只 inpaint 角落。
2. **遮挡(occlusion)**：要让画里的东西动，必须抠成独立层 + 把底图对应位置补上。月亮自转 → 单独透明月亮 +
   sky 去月补星空；前景在云之上 → 单独前景层 + sky 也去掉前景（否则云后重影）。
3. **分层对齐难点 → `.cover-frame`**：把"cover 裁剪后的画布"显式做成居中容器，四层都铺进去 →
   共享同一裁剪、天然对齐；月亮按画布百分比定位即可原地自转（不绕屏幕中心打转）。
4. **云横移漏边** → 给 `clouddrift` 关键帧加 `scale(1.08)`（每侧 +4% 余量），平移 ±1.3% 始终把边缘留在画外，且不拉伸变形。
5. **视差不与动画打架** → 用 CSS **独立 `translate` 属性**（与 `transform` 动画分属不同属性，自动合成）。
6. **缓存**：覆盖同名 webp 时用 `?v=N` 击穿浏览器缓存（和项目 `app.js?v=1.3` 同套路）。

---

## 硬约束（已全部遵守）

- **零外部 CDN**：图片自托管 `static/img/login/`，无动画库（视差是 vanilla JS）。
- **Tailwind 预编译**：新视觉全写进 `<style>` 自定义类，未引入未编译的新工具类（无需重跑构建）。
- **无障碍 & 防闪烁**：`prefers-reduced-motion` 关闭一切动效（含视差 JS）；星星周期 2.6–5s、远低于 3 次/秒
  （WCAG 2.3.1）；磨砂卡 + 文字阴影保证白字对比；输入框聚焦态清晰、点击区大（关怀向产品）。
- **性能**：动画只用 `transform`/`opacity`/`translate`（合成器友好）；总图 836KB；L0 渐变兜底防白闪。
- **版权**：底图为开源社区取得、用户确认无版权问题；已去除原作签名与 Gemini 水印。

---

## 阶段状态

- [x] **P0 资产** —— 原图 + Gemini 扩横屏 + 分层（sky/clouds/foreground/moon），本地抠图 → 4 张 webp。
- [x] **P1 静态合成 + 深色卡** —— cover-frame 四层 + L0 兜底；磨砂玻璃卡 + 暖金 CTA + 透明描边输入 + 暖金徽章。
- [x] **P2 动效** —— 月亮自转、月晕呼吸、云视差、~100 星闪烁；reduced-motion 降级。
- [x] **P4 鼠标/陀螺仪视差** —— `login_parallax.js`，生命周期 + reduced-motion 已处理。
- [x] **清理** —— 旧 `scene.webp` / `*.placeholder.svg` / `_gen_placeholders.py` 已删；`assets/` 中间产物已清。
- [ ] **P3 收尾（剩余，需在真机/你这边）**：
  - [x] 更新 `doc/Web.md` 登录页截图 → `assets/login-page.png`（README 无登录截图；其余面板 UI 未变，配图照旧）。
  - [ ] 跨端实测：Safari / iOS 的 `backdrop-filter`、WebP、`translate` 属性；手机性能与流畅度；reduced-motion 实机。

**可选增强（不必做）**：云拆前/后两层更强纵深；移动端窄屏改用竖版底图；登录成功瞬间月亮/星星呼应动效。

---

## 怎么改 / 怎么换图（备查）

**微调旋钮**（都在 [`templates/index.html`](templates/index.html) `<style>`）：

- 卡片透明/磨砂：`.login-card` 的 `rgba(15,23,42,.38)`（越小越透）、`blur(18px)`（越大越磨砂）。
- 视差幅度：`.layer-*` 的 `translate: calc(var(--mx) * -Npx) ...` 那几个 `N`。
- 云速/月转速/月晕/星密：`@keyframes clouddrift|moonspin|glowpulse|twinkle2` 时长，及 `.layer-stars` 里 `.tw` 数量。
- 月亮位置/大小：`.moon` 与 `.moon-glow` 的 `left/top/width`（现 49% / 17.8% / 12.1%）。

**换插画 / 重抠图层**：用同一张横屏源图，按下方 prompt 让 Gemini 出
**纯色背景**的各层，再本地抠（云/前景品红 chroma-key、月亮暗场景按亮度、sky 纯星空），覆盖同名 webp 并把 `?v=` +1。

- sky：`keep ONLY the starry sky; remove moon, all clouds, and the foreground; fill with more starry sky; remove any signature; same canvas size.`
- clouds：`keep ONLY the blue clouds + golden crests in place; everything else solid pure magenta #FF00FF, flat; same canvas size.`
- foreground：`keep ONLY the person/bench/signpost/ground at the bottom; everything else solid pure magenta #FF00FF, flat; same canvas size.`
- moon：`keep ONLY the glowing moon, bright; everything else solid pure black #000000, flat; same canvas size.`

> 注意：Gemini 不输出真 alpha（会画灰棋盘格），所以一律走"纯色背景 + 本地抠"，别指望它直接给透明 PNG。

---

## 全站推广：星空背景 + 深蓝磨砂玻璃皮肤（已完成 ✅）

> 把登录页背景做到**所有页面**，并让登录后的内页配色贴合星空风（深蓝冷 + 暖黄）。
> 全部改在 [`templates/index.html`](templates/index.html) 的 `<style>` 与少量结构里，**零新增 Tailwind 工具类**
> （`static/vendor/tailwind.css` 是按需裁剪的，新工具类不会生效），故无需重跑 Tailwind 构建。

**背景全局化**
- 原来背景图层埋在登录遮罩 `.login-stage`(`v-if="!currentUser"`) 里；现抽到 `#app` 顶部的常驻
  `<div class="sky-bg" :class="currentUser ? 'is-calm' : 'is-login'">`，`position:fixed; inset:0; z-index:-1`
  沉到所有内容之下，所有页面共享同一张星空。`<body>` 去掉 `bg-blue-50/text-gray-800` 改透明 + 浅字，
  `html` 给 `#070b22` 兜底防白闪。登录卡片单独留在 `v-if="!currentUser"` 的 `.login-stage` 浮层(`z-[200]`)。
- **两种活力**：`.is-login`(登录页)=原灵动版（云移、月转、星闪、视差）；`.is-calm`(已登录内页)=安静版
  （`brightness(.72)` 轻调暗 + `::after` 压暗蒙版；月亮/云冻结，**星星保留 `twinkle2` 闪烁**），
  工作时不晃眼又不失灵气、且省电（无视差/无月转/无云移）。
- 视差 [`login_parallax.js`](static/js/login_parallax.js) 的 `getStage()` 选择器从 `.login-stage` 改为
  `.sky-bg.is-login` → 登录后背景切到 `.is-calm`、选择器落空 → rAF 自动停，内页零视差开销。

**内页重皮（深蓝磨砂玻璃）**
- 在已登录容器加 `class="vg-app"`，用更高优先级的 `.vg-app .<工具类>` 覆盖既有工具类的**颜色**（不动结构/布局，整段可删回滚）：
  白卡→`rgba(17,24,39,.5)`+`backdrop-filter:blur`；淡蓝/淡灰面板→半透明深色；深灰阶文字→浅色阶；
  `text-blue-*` 强调字→暖金 `#fcd34d`；绿/红/天蓝/紫等语义色调亮；边框→极淡冷光。
- 表单：浅底→深底浅字；**无 `bg-*` 工具类的数字框**用 `:where(input…)`（0 特异度）兜底深色，避免浅底浅字看不见；
  个别 `bg-white` 输入/下拉去掉卡片级模糊与重投影。
- ECharts 训练曲线（[`finetune_panel.js`](static/js/finetune_panel.js)）legend/坐标轴文字调亮以适配深底面板。
- 缓存击穿：`app.js?v=1.4`、`login_parallax.js?v=2`、`finetune_panel.js?v=1.4`。

**保留为浅色**：全局自定义对话框（`#app` 顶层、`.vg-app` 之外）仍是白底弹窗——瞬时提醒、高对比，刻意不染色；
如需统一为深色，给那块也套深色皮即可。

**已用 Playwright + 静态服 + mock /api 截图核验**：登录页、采集、微调、识别四态均为深蓝玻璃风、文字可读、无 page error。

---

## light / dark 主题切换（已完成 ✅）

> dark = 上面这套星空深色；light = 原来的亮色配色、无背景。右下角悬浮按钮切换，带「从按钮圆形扩散」动画。

**机制（极简、复用上面的皮）**
- 真值源：`theme`(ref，默认 dark) → 写到 `<html data-theme>` + 记忆 `localStorage['whisper_theme']`（见 [`app.js`](static/js/app.js)）。
- **dark→light 只做两件事**：① 登录后容器的 `:class="{ 'vg-app': theme==='dark' }"` 去掉 `.vg-app` → 整段深色覆盖失效 → 回到原生 Tailwind 亮色；② `<html>` 挂 `.theme-light-app`（仅「亮色 且 已登录」）→ `.sky-bg{display:none}` 藏星空 + `body` 还原 `#eff6ff/#1f2937`。**登录页恒星空**（未登录不挂该类），FAB 只在登录后出现。
- **防闪烁**：`<head>` 内联一小段在样式前先按 localStorage 定好 `data-theme`/`.theme-light-app`（用 token 存在与否猜是否已登录），再由 app.js 的 `watchEffect` 接管同步。
- **ECharts 训练曲线**随主题换轴/图例配色（[`finetune_panel.js`](static/js/finetune_panel.js) 的 `chartAxisColors`/`applyChartTheme`，监听 `whisper:theme` 事件即时重绘），否则亮色白底上浅字看不清。

**扩散动画** —— View Transitions API（[`app.js`](static/js/app.js) `toggleTheme`）
- `document.startViewTransition()` 拍快照后，对 `::view-transition-new(root)` 用 Web Animations 把 `clip-path: circle()` **从按钮中心半径 0 → 盖满全屏**地涨开；`<style>` 里把两张快照的默认淡入淡出关掉（`animation:none`）只留这个圆。
- **降级**：不支持（老 iOS / 部分国内内核）或 `prefers-reduced-motion` 时直接切换，不报错。

**FAB**：右下角圆形磨砂玻璃按钮，dark 显暖金月 ☾ / light 显暖橙日 ☀；移动端 `bottom:92px` 抬高避开底部 tab 栏，`≥1024px` 落到 `bottom:26px` 角落。缓存：`app.js?v=1.5`、`finetune_panel.js?v=1.5`。
