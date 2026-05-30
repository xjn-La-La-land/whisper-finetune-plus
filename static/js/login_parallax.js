// 登录页·极轻视差（P4）
// 鼠标移动 / 设备倾斜 → 给灵动版背景 .sky-bg.is-login 设 CSS 变量 --mx/--my（范围约 -1..1），
// 各背景图层在 CSS 里按"深度"用 translate 偏移几像素（与 transform 动画叠加），形成纵深感。
// 约束：仅登录页(背景为 .sky-bg.is-login)生效；登录后背景切到 .is-calm → 选择器落空 → 自动停；
//      prefers-reduced-motion 下完全关闭；无任何依赖。
(function () {
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  let tx = 0, ty = 0, cx = 0, cy = 0, raf = null;
  const getStage = () => document.querySelector('.sky-bg.is-login');

  function loop() {
    raf = null;
    const stage = getStage();
    if (!stage) { cx = cy = 0; return; }          // 登录成功后遮罩被 v-if 移除 → 自动停
    cx += (tx - cx) * 0.08;                        // 缓动跟随，丝滑不生硬
    cy += (ty - cy) * 0.08;
    stage.style.setProperty('--mx', cx.toFixed(4));
    stage.style.setProperty('--my', cy.toFixed(4));
    if (Math.abs(tx - cx) > 0.0008 || Math.abs(ty - cy) > 0.0008) raf = requestAnimationFrame(loop);
  }
  const kick = () => { if (raf == null) raf = requestAnimationFrame(loop); };

  // 桌面：鼠标
  window.addEventListener('mousemove', function (e) {
    if (!getStage()) return;                       // 不在登录页则空转，零开销
    tx = (e.clientX / window.innerWidth - 0.5) * 2;
    ty = (e.clientY / window.innerHeight - 0.5) * 2;
    kick();
  }, { passive: true });

  // 移动端：陀螺仪（iOS 13+ 需手势授权，未授权则不触发，自动降级为无视差）
  window.addEventListener('deviceorientation', function (e) {
    if (!getStage() || e.gamma == null) return;
    tx = Math.max(-1, Math.min(1, e.gamma / 30));        // 左右倾斜
    ty = Math.max(-1, Math.min(1, (e.beta - 45) / 30));  // 前后倾斜（45° 视为中性持握）
    kick();
  }, { passive: true });
})();
