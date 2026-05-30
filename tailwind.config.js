/**
 * Tailwind v3 构建配置 —— 把原来的 Play CDN (https://cdn.tailwindcss.com) 换成本地预编译 CSS，
 * 解决国内访问海外 CDN 慢/失败导致页面无样式的问题（见 TODO_PUBLIC_DEPLOY.md 的 A-5）。
 *
 * 重新生成 static/vendor/tailwind.css（改动 HTML/JS 里的 class 后需重跑）：
 *   npx --yes tailwindcss@3 -c tailwind.config.js -i tailwind.input.css -o static/vendor/tailwind.css --minify
 *
 * content 必须覆盖所有出现 class 的源文件：Tailwind 按原始文本扫描 token，
 * 连 Vue 的 enter-active-class="transition duration-200" 这类属性值里的类也要扫到，
 * 否则生产构建会漏掉这些工具类，导致样式缺失。
 */
module.exports = {
  content: [
    './templates/**/*.html',
    './static/js/**/*.js',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};
