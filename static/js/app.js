const { createApp, ref, onMounted } = Vue;

import CustomAudio from './custom_audio.js?v=1.2';
import AudioCollector from './audio_collector.js?v=1.2';
import FinetunePanel from './finetune_panel.js?v=1.2';
import InferencePanel from './inference_panel.js?v=1.2';
import * as dialog from './dialog.js?v=1.2';
import { apiFetch, setToken, clearToken, getToken, getStoredUsername } from './api.js?v=1.2';

const app = createApp({
    setup() {
        // --- 全局状态 ---
        const currentUser = ref("");
        const loginInput = ref("");
        const passwordInput = ref("");
        const isCollectOnly = ref(false);

        // --- Tab 控制与动画状态 ---
        const currentTab = ref('AudioCollector'); // 默认显示第一个节点
        const transitionName = ref('slide-left'); // 默认进场方向：向左滑动
        const tabOrder = ['AudioCollector', 'FinetunePanel', 'InferencePanel'];

        const changeTab = (targetTab) => {
            if (currentTab.value === targetTab) return;
            const currentIndex = tabOrder.indexOf(currentTab.value);
            const targetIndex = tabOrder.indexOf(targetTab);
            transitionName.value = targetIndex > currentIndex ? 'slide-left' : 'slide-right';
            currentTab.value = targetTab;
        };

        // --- 登录/注册逻辑 ---
        // 提交时校验密码长度，符合后端 4-16 字符的要求
        const validatePassword = (pwd) => {
            if (!pwd) return "密码不能为空哦！";
            if (pwd.length < 4) return "密码至少需要 4 个字符。";
            if (pwd.length > 16) return "密码最多 16 个字符。";
            return null;
        };

        const handleAuth = async (type) => {
            const name = loginInput.value.trim();
            const pwd = passwordInput.value;
            if (!name) return dialog.alert("代号不能为空哦！", { variant: 'warning' });
            const pwdErr = validatePassword(pwd);
            if (pwdErr) return dialog.alert(pwdErr, { variant: 'warning' });

            const endpoint = type === 'login' ? '/api/login' : '/api/register';
            try {
                // 登录/注册接口本身不需要 token，所以用原生 fetch
                const res = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username: name, password: pwd })
                });

                if (!res.ok) {
                    const errorData = await res.json();
                    await dialog.alert(errorData.message || '认证失败', { variant: 'warning' });
                    return;
                }

                const data = await res.json();
                // 服务器返回 { message, token, username }
                if (!data.token) {
                    await dialog.alert('登录响应缺少 token，请联系管理员', { variant: 'danger' });
                    return;
                }
                setToken(data.token, data.username || name);
                currentUser.value = data.username || name;
                // 清掉密码输入框，避免悬停回显
                passwordInput.value = "";
            } catch (err) {
                await dialog.alert("网络请求失败，请检查后端服务是否启动！", { variant: 'danger' });
            }
        };

        const handleLogout = async () => {
            const ok = await dialog.confirm("确定要退出当前账号吗？");
            if (!ok) return;
            clearToken();
            currentUser.value = "";
            loginInput.value = "";
            passwordInput.value = "";
        };

        onMounted(async () => {
            try {
                const configRes = await fetch('/api/config');
                const configData = await configRes.json();
                isCollectOnly.value = configData.collect_only;
            } catch (e) {
                console.error("获取系统配置失败", e);
            }

            // 从 localStorage 恢复 token，先调 /api/me 确认它还有效
            // 这样可以避免后续每个面板自己处理过期 token
            const savedToken = getToken();
            if (savedToken) {
                try {
                    const r = await apiFetch('/api/me');
                    if (r.ok) {
                        const data = await r.json();
                        currentUser.value = data.username;
                    } else {
                        // 401 已被 apiFetch 自动清掉 token
                        currentUser.value = "";
                    }
                } catch (e) {
                    console.error("token 校验失败", e);
                    currentUser.value = "";
                }
            }

            // 监听 apiFetch 抛出的 401 事件：任何后续请求收到 401 都自动把用户踢回登录页
            window.addEventListener('whisper:unauthorized', () => {
                if (currentUser.value) {
                    currentUser.value = "";
                    dialog.alert("登录已过期，请重新登录", { variant: 'warning' });
                }
            });

            // 全局键盘支持：dialog 显示时 Enter 确认 / Esc 取消
            window.addEventListener('keydown', (e) => {
                if (!dialog.dialogState.visible) return;
                if (e.key === 'Enter') {
                    e.preventDefault();
                    dialog.handleConfirm();
                } else if (e.key === 'Escape') {
                    e.preventDefault();
                    dialog.handleCancel();
                }
            });
        });

        return {
            currentUser,
            loginInput,
            passwordInput,
            isCollectOnly,
            currentTab,
            transitionName,
            changeTab,
            handleAuth,
            handleLogout,
            dialogState: dialog.dialogState,
            dialogConfirm: dialog.handleConfirm,
            dialogCancel: dialog.handleCancel
        }
    }
});

// 全局注册组件
app.component('custom-audio', CustomAudio);
app.component('AudioCollector', AudioCollector);
app.component('FinetunePanel', FinetunePanel);
app.component('InferencePanel', InferencePanel)
app.mount('#app');
