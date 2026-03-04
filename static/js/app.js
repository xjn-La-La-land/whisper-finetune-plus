const { createApp, ref, onMounted } = Vue;

import CustomAudio from './custom_audio.js';
import AudioCollector from './audio_collector.js';
import FinetunePanel from './finetune_panel.js';
import InferencePanel from './inference_panel.js';

const app = createApp({
    setup() {
        // --- 全局状态 ---
        const currentUser = ref("");
        const loginInput = ref("");
        
        // --- Tab 控制与动画状态 ---
        const currentTab = ref('AudioCollector'); // 默认显示第一个节点
        const transitionName = ref('slide-left'); // 默认进场方向：向左滑动
        const tabOrder = ['AudioCollector', 'FinetunePanel', 'InferencePanel'];

        // --- 新增：智能判断滑动方向的切换函数 ---
        const changeTab = (targetTab) => {
            if (currentTab.value === targetTab) return; 
            
            // 获取当前页面和目标页面在数组中的序号 (0, 1, 2)
            const currentIndex = tabOrder.indexOf(currentTab.value);
            const targetIndex = tabOrder.indexOf(targetTab);
            
            // 智能判断方向：目标序号大于当前序号就是前进（向左滑），否则就是后退（向右滑）
            if (targetIndex > currentIndex) {
                transitionName.value = 'slide-left';  
            } else {
                transitionName.value = 'slide-right'; 
            }
            
            currentTab.value = targetTab;
        };

        // --- 登录/注册逻辑 (保持原样即可) ---
        const handleAuth = async (type) => {
            // ... (你原来的代码逻辑) ...
            const name = loginInput.value.trim();
            if (!name) return alert("代号不能为空哦！");

            const endpoint = type === 'login' ? '/api/login' : '/api/register';
            try {
                const res = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username: name })
                });

                if (!res.ok) {
                    const errorData = await res.json();
                    alert(errorData.message);
                    return;
                }

                currentUser.value = name;
                localStorage.setItem("whisper_username", name);
            } catch (err) {
                alert("网络请求失败，请检查后端服务是否启动！");
            }
        };
        
        const handleLogout = () => {
            if (!confirm("确定要退出当前账号吗？")) return;
            currentUser.value = "";
            loginInput.value = "";
            localStorage.removeItem("whisper_username");
        };

        onMounted(() => {
            const savedName = localStorage.getItem("whisper_username");
            if (savedName) {
                currentUser.value = savedName;
            }
        });

        // ================= 注意这里！将新定义的变量和函数 return 暴露给 HTML =================
        return { 
            currentUser, 
            loginInput, 
            currentTab, 
            transitionName,  // 导出动画类名
            changeTab,       // 导出切换事件
            handleAuth, 
            handleLogout 
        }
    }
});

// 全局注册组件
app.component('custom-audio', CustomAudio);
app.component('AudioCollector', AudioCollector);
app.component('FinetunePanel', FinetunePanel);
app.component('InferencePanel', InferencePanel)
app.mount('#app');