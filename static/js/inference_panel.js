const { ref, computed, onMounted, onActivated, nextTick, watch } = Vue;
import * as dialog from './dialog.js?v=1.2';
import { apiFetch } from './api.js?v=1.2';
import * as wasm from './wasm_whisper.js?v=1.1';
import { loadConverter, toSimplified } from './zhconv_lite.js?v=1.0';

// 基座 q5_0 的近似大小（仅用于卡片展示，真实大小以下载时 content-length 为准）
const BASE_MB = { 'whisper-tiny': 30, 'whisper-base': 55, 'whisper-small': 175 };

export default {
    template: '#tpl-inference-panel',
    props: ['currentUser'],
    setup(props) {
        // 焊死的识别参数（不再暴露给用户）：贪心解码 + 转简体 + 保留标点。
        const FIXED = { to_simple: 1, remove_pun: 0, num_beams: 1 };

        // --- 推理模式：local(浏览器 WASM) / server(服务端 GPU) ---
        const mode = ref('server');               // 探测后若支持本地则切到 local
        const localSupported = ref(false);
        const localUnsupportedReason = ref('');

        const selectedModel = ref('');
        const userModels = ref([]);               // [{model_name, has_ggml, is_published, ...}]
        const baseModelOptions = ref([]);         // 已下载的基座名（下拉用）
        const baseGgmlSet = ref(new Set());       // 有 ggml 的基座名

        // --- 本地模型卡片状态机 ---
        // 'unavailable' | 'need-download' | 'downloading' | 'loading' | 'ready'
        const localState = ref('unavailable');
        const downloadPct = ref(0);
        const downloadRecvMB = ref(0);

        // --- 录音 / UI 状态 ---
        const isRecording = ref(false);
        const isProcessing = ref(false);
        const processingElapsed = ref('');
        const audioInput = ref(null);
        let mediaRecorder = null;
        let audioChunks = [];
        let procTimer = null;

        // --- 终端 / 回放 ---
        const transcriptItems = ref([]);
        const historyContainer = ref(null);
        const currentStatus = ref('等待输入音频...');
        const statusType = ref('system');
        const copiedItemId = ref(null);
        const usedModelType = ref('');
        const tempAudioPath = ref('');

        const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

        const typeStatus = async (text, type = 'system') => {
            statusType.value = type;
            currentStatus.value = '';
            for (const ch of text) { currentStatus.value += ch; await sleep(type === 'error' ? 14 : 10); }
        };

        const typeTranscript = async (chunk) => {
            const item = { id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`, start: chunk.start, end: chunk.end, text: '' };
            transcriptItems.value.push(item);
            for (const ch of chunk.text) {
                item.text += ch;
                await nextTick();
                if (historyContainer.value) historyContainer.value.scrollTop = historyContainer.value.scrollHeight;
                await sleep(20);
            }
        };

        const resetStatusOnly = () => {
            currentStatus.value = '等待输入音频...';
            statusType.value = 'system';
            copiedItemId.value = null;
        };

        // --- 模型列表 ---
        const isBaseModelName = (name) => baseModelOptions.value.includes(name);
        const selectedModelTag = () => {
            if (!selectedModel.value) return '';
            return isBaseModelName(selectedModel.value) ? 'base' : 'finetuned';
        };

        const ensureSelectedModel = () => {
            const all = new Set([...userModels.value.map((m) => m.model_name), ...baseModelOptions.value]);
            if (!all.size) { selectedModel.value = ''; return; }
            if (!selectedModel.value || !all.has(selectedModel.value)) {
                selectedModel.value = baseModelOptions.value.includes('whisper-base')
                    ? 'whisper-base'
                    : (baseModelOptions.value[0] || userModels.value[0]?.model_name || '');
            }
        };

        const checkModelStatus = async () => {
            if (!props.currentUser) return;
            try {
                const [um, bm] = await Promise.all([
                    apiFetch('/api/user_models'),
                    apiFetch('/api/base_models', { cache: 'no-store' }),
                ]);
                const umData = await um.json();
                const bmData = await bm.json();
                userModels.value = Array.isArray(umData.models) ? umData.models : [];
                baseModelOptions.value = Array.isArray(bmData.models) ? bmData.models : [];
                baseGgmlSet.value = new Set((bmData.all_models || []).filter((m) => m.has_ggml).map((m) => m.name));
                ensureSelectedModel();
                await refreshLocalState();
            } catch (e) {
                console.error('检查模型状态失败', e);
                userModels.value = []; baseModelOptions.value = []; selectedModel.value = '';
            }
        };

        // 给定模型名，返回它的「本地可用信息」或 null（null = 无本地版，只能服务端）
        const localInfoFor = (name) => {
            if (!name) return null;
            if (isBaseModelName(name)) {
                if (!baseGgmlSet.value.has(name)) return null;
                return {
                    kind: 'base',
                    url: `/api/download_base_ggml?model=${encodeURIComponent(name)}`,
                    cacheKey: `base:${name}`,
                    sizeMB: BASE_MB[name] || null,
                };
            }
            // 用户微调模型：只要有 ggml 就能本地用（不必发布）。用 updated_at 做缓存隔离（重训会变）。
            const m = userModels.value.find((x) => x.model_name === name);
            if (!m || !m.has_ggml) return null;
            const v = String(m.updated_at || '');
            return {
                kind: 'finetuned',
                url: `/api/download_user_ggml?model=${encodeURIComponent(name)}&v=${encodeURIComponent(v)}`,
                cacheKey: `user:${props.currentUser}:${name}:${v}`,
                sizeMB: null,
            };
        };

        const cardInfo = computed(() => localInfoFor(selectedModel.value));
        const hasLocal = (name) => !!localInfoFor(name);   // 下拉里标注「无本地版」

        // 切模型 / 切到本地模式时，刷新卡片状态（已缓存的自动加载好，未缓存的等用户点下载）
        const refreshLocalState = async () => {
            if (mode.value !== 'local') return;
            const info = localInfoFor(selectedModel.value);
            if (!info) { localState.value = 'unavailable'; return; }
            if (wasm.getLoadedKey() === info.cacheKey) { localState.value = 'ready'; return; }
            try {
                if (await wasm.isCached(info.cacheKey)) {
                    localState.value = 'loading';
                    await wasm.ensureModel({ cacheKey: info.cacheKey, url: info.url, fetcher: apiFetch });
                    localState.value = 'ready';
                } else {
                    localState.value = 'need-download';
                }
            } catch (e) {
                localState.value = 'need-download';
            }
        };

        const downloadModel = async () => {
            const info = localInfoFor(selectedModel.value);
            if (!info) return;
            localState.value = 'downloading';
            downloadPct.value = 0; downloadRecvMB.value = 0;
            try {
                await wasm.ensureModel({
                    cacheKey: info.cacheKey, url: info.url, fetcher: apiFetch,
                    onProgress: (recv, total) => {
                        downloadRecvMB.value = (recv / 1048576).toFixed(0);
                        downloadPct.value = total ? Math.round((100 * recv) / total) : 0;
                    },
                });
                localState.value = 'ready';
            } catch (e) {
                localState.value = 'need-download';
                dialog.alert('模型下载/加载失败：' + e.message, { variant: 'danger' });
            }
        };

        const redownloadModel = async () => {
            const info = localInfoFor(selectedModel.value);
            if (!info) return;
            await wasm.evictModel(info.cacheKey);
            await downloadModel();
        };

        // --- 模式切换 ---
        const setMode = async (m) => {
            if (m === 'local' && !localSupported.value) return;
            mode.value = m;
            usedModelType.value = '';
            if (m === 'local') await refreshLocalState();
        };

        // --- 推理：服务端 ---
        const sendAudioToBackend = async (audioBlob) => {
            if (!selectedModel.value) { dialog.alert('请先选择一个可用模型', { variant: 'warning' }); return; }
            isProcessing.value = true;
            resetStatusOnly();
            if (tempAudioPath.value) URL.revokeObjectURL(tempAudioPath.value);
            tempAudioPath.value = URL.createObjectURL(audioBlob);
            await typeStatus('🎵 音频已捕获，正在发送至 GPU 节点...', 'system');

            const fd = new FormData();
            fd.append('audio', audioBlob, 'infer.wav');
            fd.append('to_simple', FIXED.to_simple);
            fd.append('remove_pun', FIXED.remove_pun);
            fd.append('num_beams', FIXED.num_beams);
            fd.append('model_name', selectedModel.value);
            try {
                const res = await apiFetch('/api/recognition', { method: 'POST', body: fd });
                if (!res.ok) { const err = await res.json(); await typeStatus(`❌ 错误: ${err.detail}`, 'error'); return; }
                const data = await res.json();
                const t = data.used_model_type === 'base' ? 'base' : 'finetuned';
                usedModelType.value = `${data.used_model} (服务端·${t})`;
                await typeStatus(`✅ 模型加载就绪 [${usedModelType.value}]，开始解码...`, 'system');
                for (const chunk of data.results) await typeTranscript(chunk);
                await typeStatus('✨ 识别完成！等待下一次指令...', 'system');
            } catch (err) {
                await typeStatus(`❌ 网络请求失败: ${err.message}`, 'error');
            } finally {
                isProcessing.value = false;
            }
        };

        // --- 推理：本地 WASM ---
        const runLocalInference = async (audioBlob) => {
            const info = localInfoFor(selectedModel.value);
            if (!info) { dialog.alert('该模型没有本地版本，请切换到服务端或换一个模型', { variant: 'warning' }); return; }
            if (localState.value !== 'ready') { dialog.alert('请先点「下载模型」把模型准备好', { variant: 'warning' }); return; }
            isProcessing.value = true;
            resetStatusOnly();
            if (tempAudioPath.value) URL.revokeObjectURL(tempAudioPath.value);
            tempAudioPath.value = URL.createObjectURL(audioBlob);

            const t0 = Date.now();
            procTimer = setInterval(() => { processingElapsed.value = ((Date.now() - t0) / 1000).toFixed(0); }, 200);
            try {
                await typeStatus('🎵 音频已捕获，本地解码中...', 'system');
                const pcm = await wasm.decodeToPcm16k(audioBlob);
                await loadConverter().catch(() => {});      // 简繁表（失败则不转，原样显示）
                await typeStatus(`⚡ 本地识别中（${selectedModel.value}）...`, 'system');
                const { segments } = await wasm.transcribe(pcm, 'zh');
                usedModelType.value = `${selectedModel.value} (本地·${info.kind})`;
                if (!segments.length) {
                    await typeStatus('（未识别到语音内容）', 'system');
                } else {
                    for (const seg of segments) {
                        await typeTranscript({ start: seg.start, end: seg.end, text: toSimplified(seg.text) });
                    }
                    await typeStatus('✨ 本地识别完成！', 'system');
                }
            } catch (e) {
                await typeStatus(`❌ 本地识别失败: ${e.message}`, 'error');
            } finally {
                clearInterval(procTimer);
                isProcessing.value = false;
                processingElapsed.value = '';
            }
        };

        const runInference = (blob) => (mode.value === 'local' ? runLocalInference(blob) : sendAudioToBackend(blob));

        // 录音 / 上传是否可用：处理中不可；本地模式需模型 ready
        const canCapture = computed(() => {
            if (isProcessing.value) return false;
            if (mode.value === 'local') return localState.value === 'ready';
            return !!selectedModel.value;
        });

        // --- 生命周期 ---
        onMounted(async () => {
            const cap = await wasm.detectCapabilities();
            localSupported.value = cap.ok;
            localUnsupportedReason.value = cap.reason;
            mode.value = cap.ok ? 'local' : 'server';
            await checkModelStatus();
        });
        onActivated(() => { checkModelStatus(); });

        watch(() => props.currentUser, async (u) => {
            if (!u) { userModels.value = []; baseModelOptions.value = []; selectedModel.value = ''; return; }
            await checkModelStatus();
        });
        watch(selectedModel, () => { refreshLocalState(); });

        // --- 交互 ---
        const handleFileUpload = async (event) => {
            const file = event.target.files[0];
            if (!file) return;
            await runInference(file);
            if (audioInput.value) audioInput.value.value = '';
        };

        const toggleRecording = async () => {
            if (isRecording.value) {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach((t) => t.stop());
                isRecording.value = false;
            } else {
                resetStatusOnly();
                usedModelType.value = '';
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
                    mediaRecorder.onstop = () => { runInference(new Blob(audioChunks, { type: 'audio/wav' })); };
                    mediaRecorder.start();
                    isRecording.value = true;
                } catch (e) {
                    dialog.alert('无法访问麦克风，请检查权限！', { variant: 'danger' });
                }
            }
        };

        const copyTranscript = async (item) => {
            try {
                await navigator.clipboard.writeText(item.text);
                copiedItemId.value = item.id;
                setTimeout(() => { if (copiedItemId.value === item.id) copiedItemId.value = null; }, 1200);
            } catch (e) {
                dialog.alert('复制失败，请检查浏览器权限', { variant: 'warning' });
            }
        };

        return {
            mode, setMode, localSupported, localUnsupportedReason,
            selectedModel, userModels, baseModelOptions, isBaseModelName, selectedModelTag, hasLocal,
            localState, cardInfo, downloadPct, downloadRecvMB, downloadModel, redownloadModel,
            isRecording, isProcessing, processingElapsed, canCapture, audioInput,
            transcriptItems, historyContainer, currentStatus, statusType, copiedItemId,
            usedModelType, tempAudioPath,
            handleFileUpload, toggleRecording, copyTranscript,
        };
    },
};
