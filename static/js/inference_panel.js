const { ref, onMounted, nextTick, watch } = Vue;

export default {
    template: '#tpl-inference-panel',
    props: ['currentUser'],
    setup(props) {
        // --- 参数状态 ---
        const params = ref({
            to_simple: true,
            remove_pun: false,
            num_beams: 1,
            selected_model: ''
        });

        const modelOptions = ref([]); // 用户微调模型列表
        const baseModelOptions = ref([]); // 基础模型列表

        // --- 录音与 UI 状态 ---
        const isRecording = ref(false);
        const isProcessing = ref(false);
        const audioInput = ref(null);
        let mediaRecorder = null;
        let audioChunks = [];

        // --- 终端与回放状态 ---
        const transcriptItems = ref([]); // 识别文本历史
        const historyContainer = ref(null);
        const currentStatus = ref("等待输入音频...");
        const statusType = ref("system");
        const copiedItemId = ref(null);
        const usedModelType = ref(""); // 显示用的是Base还是微调模型
        const tempAudioPath = ref(""); // 用于回放刚刚录制的音频

        const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

        // 顶部状态行打字机（每次覆盖上一条）
        const typeStatus = async (text, type = "system") => {
            statusType.value = type;
            currentStatus.value = "";
            for (const char of text) {
                currentStatus.value += char;
                await sleep(type === "error" ? 14 : 10);
            }
        };

        // 识别文本打字机（写入历史列表）
        const typeTranscript = async (chunk) => {
            const item = {
                id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
                start: chunk.start,
                end: chunk.end,
                text: ""
            };
            transcriptItems.value.push(item);

            for (const char of chunk.text) {
                item.text += char;
                await nextTick();
                if (historyContainer.value) {
                    historyContainer.value.scrollTop = historyContainer.value.scrollHeight;
                }
                await sleep(20);
            }
        };

        const resetStatusOnly = () => {
            currentStatus.value = "等待输入音频...";
            statusType.value = "system";
            copiedItemId.value = null;
        };

        const ensureSelectedModel = () => {
            const finetunedNames = modelOptions.value.map(m => m.model_name);
            const baseNames = [...baseModelOptions.value];
            const allNames = new Set([...finetunedNames, ...baseNames]);
            if (!allNames.size) {
                params.value.selected_model = '';
                return;
            }
            if (!params.value.selected_model || !allNames.has(params.value.selected_model)) {
                if (baseNames.includes('whisper-large-v3')) {
                    params.value.selected_model = 'whisper-large-v3';
                } else {
                    params.value.selected_model = baseNames[0] || finetunedNames[0];
                }
            }
        };

        const isBaseModelName = (modelName) => baseModelOptions.value.includes(modelName);
        const selectedModelTag = () => {
            if (!params.value.selected_model) return "";
            return isBaseModelName(params.value.selected_model) ? "base" : "finetuned";
        };

        // 检查模型是否存在
        const checkModelStatus = async () => {
            if (!props.currentUser) return;
            try {
                const [userModelsRes, baseModelsRes] = await Promise.all([
                    fetch(`/api/user_models?username=${encodeURIComponent(props.currentUser)}`),
                    fetch('/api/base_models', { cache: 'no-store' })
                ]);
                const userModelsData = await userModelsRes.json();
                const baseModelsData = await baseModelsRes.json();
                modelOptions.value = Array.isArray(userModelsData.models) ? userModelsData.models : [];
                baseModelOptions.value = Array.isArray(baseModelsData.models) ? baseModelsData.models : [];
                ensureSelectedModel();
            } catch (e) {
                console.error("检查模型状态失败", e);
                modelOptions.value = [];
                baseModelOptions.value = [];
                params.value.selected_model = '';
            }
        };

        const sendAudioToBackend = async (audioBlob) => {
            if (!params.value.selected_model) {
                alert("请先选择一个可用模型");
                return;
            }
            isProcessing.value = true;
            resetStatusOnly();
            
            if (tempAudioPath.value) URL.revokeObjectURL(tempAudioPath.value);
            tempAudioPath.value = URL.createObjectURL(audioBlob);

            await typeStatus("🎵 音频已捕获，正在发送至 GPU 节点...", "system");

            const formData = new FormData();
            formData.append("audio", audioBlob, "infer.wav");
            formData.append("username", props.currentUser);
            formData.append("to_simple", params.value.to_simple ? 1 : 0);
            formData.append("remove_pun", params.value.remove_pun ? 1 : 0);
            formData.append("num_beams", params.value.num_beams);
            formData.append("model_name", params.value.selected_model);

            try {
                const res = await fetch('/api/recognition', { method: 'POST', body: formData });
                if (!res.ok) {
                    const err = await res.json();
                    await typeStatus(`❌ 错误: ${err.detail}`, "error");
                    return;
                }
                const data = await res.json();
                
                const usedType = data.used_model_type === "base" ? "base" : "finetuned";
                usedModelType.value = `${data.used_model} (${usedType})`;
                await typeStatus(`✅ 模型加载就绪 [${usedModelType.value}]，开始解码...`, "system");

                for (const chunk of data.results) {
                    await typeTranscript(chunk);
                }

                await typeStatus("✨ 识别完成！等待下一次指令...", "system");

            } catch (err) {
                await typeStatus(`❌ 网络请求失败: ${err.message}`, "error");
            } finally {
                isProcessing.value = false;
            }
        };

        onMounted(() => {
            checkModelStatus();
        });

        watch(
            () => props.currentUser,
            async (newUser) => {
                if (!newUser) {
                    modelOptions.value = [];
                    baseModelOptions.value = [];
                    params.value.selected_model = '';
                    return;
                }
                await checkModelStatus();
            }
        );

        // --- 交互动作 ---
        const handleFileUpload = async (event) => {
            const file = event.target.files[0];
            if (!file) return;
            await sendAudioToBackend(file);
            audioInput.value.value = ""; 
        };

        const toggleRecording = async () => {
            if (isRecording.value) {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(t => t.stop());
                isRecording.value = false;
            } else {
                resetStatusOnly();
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

        const copyTranscript = async (item) => {
            try {
                await navigator.clipboard.writeText(item.text);
                copiedItemId.value = item.id;
                setTimeout(() => {
                    if (copiedItemId.value === item.id) copiedItemId.value = null;
                }, 1200);
            } catch (e) {
                alert("复制失败，请检查浏览器权限");
            }
        };

        return { 
            params, modelOptions, baseModelOptions, isRecording, isProcessing, audioInput, 
            transcriptItems, historyContainer, currentStatus, statusType, copiedItemId, usedModelType, tempAudioPath,
            selectedModelTag, isBaseModelName,
            handleFileUpload, toggleRecording, copyTranscript
        };
    }
}
