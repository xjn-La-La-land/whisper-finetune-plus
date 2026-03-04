const { ref, onMounted, nextTick } = Vue;

export default {
    template: '#tpl-inference-panel',
    props: ['currentUser'],
    setup(props) {
        // --- 参数状态 ---
        const params = ref({
            to_simple: true,
            remove_pun: false,
            num_beams: 1,
            selected_model: 'base' // 默认选择基础模型
        });

        const hasModel = ref(false); // 标记用户是否有微调模型

        // --- 录音与 UI 状态 ---
        const isRecording = ref(false);
        const isProcessing = ref(false);
        const audioInput = ref(null);
        let mediaRecorder = null;
        let audioChunks = [];

        // --- 终端与回放状态 ---
        const terminalLines = ref([]); // 存储终端显示的行
        const usedModelType = ref(""); // 显示用的是Base还是微调模型
        const tempAudioPath = ref(""); // 用于回放刚刚录制的音频
        
        // 打字机效果
        const typeWriterEffect = async (text, type = "normal") => {
            // 1. 先创建一个空的行对象并推入数组
            const lineObj = { text: "", type: type };
            terminalLines.value.push(lineObj);

            // 2. 逐字填充
            for (const char of text) {
                lineObj.text += char; // Vue 会自动追踪这个对象属性的变化

                // 3. 自动滚动逻辑
                // 使用 nextTick 确保在 Vue 更新 DOM 后执行滚动
                await nextTick();
                const terminal = document.getElementById('terminal-container');
                if (terminal) {
                    terminal.scrollTop = terminal.scrollHeight;
                }

                // 4. 控制打字速度
                // 识别出来的结果 (success 类型) 建议慢一点 (50ms)，系统提示快一点 (20ms)
                const delay = type === 'success' ? 50 : 20;
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        };

        // 检查模型是否存在
        const checkModelStatus = async () => {
            if (!props.currentUser) return;
            try {
                const res = await fetch(`/api/check_model?username=${encodeURIComponent(props.currentUser)}`);
                const data = await res.json();
                hasModel.value = data.has_model;
            } catch (e) {
                console.error("检查模型状态失败", e);
            }
        };

        const sendAudioToBackend = async (audioBlob) => {
            isProcessing.value = true;
            // 每次开始前清空旧内容，让新的识别结果处于视觉重心
            terminalLines.value = []; 
            
            if (tempAudioPath.value) URL.revokeObjectURL(tempAudioPath.value);
            tempAudioPath.value = URL.createObjectURL(audioBlob);

            await typeWriterEffect("> 🎵 音频已捕获，正在发送至 GPU 节点...", "system");

            const formData = new FormData();
            formData.append("audio", audioBlob, "infer.wav");
            formData.append("username", props.currentUser);
            formData.append("to_simple", params.value.to_simple ? 1 : 0);
            formData.append("remove_pun", params.value.remove_pun ? 1 : 0);
            formData.append("num_beams", params.value.num_beams);
            formData.append("model_type", params.value.selected_model);

            try {
                const res = await fetch('/api/recognition', { method: 'POST', body: formData });
                if (!res.ok) {
                    const err = await res.json();
                    await typeWriterEffect(`> ❌ 错误: ${err.detail}`, "error");
                    return;
                }
                const data = await res.json();
                
                usedModelType.value = data.used_model === "Finetuned" ? "专属微调模型" : "基础模型";
                await typeWriterEffect(`> ✅ 模型加载就绪 [${usedModelType.value}]，开始解码...`, "system");

                // 逐句输出，这里必须用 await 确保打完一句再打下一句
                for (const chunk of data.results) {
                    const timeStr = `[${chunk.start.toFixed(1)}s - ${chunk.end.toFixed(1)}s] `;
                    await typeWriterEffect(timeStr + chunk.text, "success");
                }

                await typeWriterEffect("> ✨ 识别完成！等待下一次指令...\n", "system");

            } catch (err) {
                await typeWriterEffect(`> ❌ 网络请求失败: ${err.message}`, "error");
            } finally {
                isProcessing.value = false;
            }
        };

        onMounted(() => {
            checkModelStatus(); 
        });

        // --- 交互动作 ---
        const handleFileUpload = async (event) => {
            const file = event.target.files[0];
            if (!file) return;
            terminalLines.value = []; // 清空之前的文字
            await sendAudioToBackend(file);
            audioInput.value.value = ""; 
        };

        const toggleRecording = async () => {
            if (isRecording.value) {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(t => t.stop());
                isRecording.value = false;
            } else {
                terminalLines.value = []; // 开始新录音前清空终端
                usedModelType.value = "";
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
                    mediaRecorder.onstop = () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        sendAudioToBackend(audioBlob);
                    };
                    mediaRecorder.start();
                    isRecording.value = true;
                } catch (e) {
                    alert("无法访问麦克风，请检查权限！");
                }
            }
        };

        return { 
            params, hasModel, isRecording, isProcessing, audioInput, 
            terminalLines, usedModelType, tempAudioPath,
            handleFileUpload, toggleRecording 
        };
    }
}