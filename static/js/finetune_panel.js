const { ref, onMounted, onUnmounted, onActivated, nextTick, watch } = Vue;
import * as dialog from './dialog.js?v=1.2';
import { apiFetch, sseUrl } from './api.js?v=1.2';
import { friendlyHttpError, friendlyNetworkError } from './errors.js?v=1';

export default {
    template: '#tpl-finetune-panel',
    props: ['currentUser', 'hasGpu'],
    setup(props) {
        // --- 状态与参数 ---
        const hasDataset = ref(false);
        const isBuilding = ref(false);
        const buildResult = ref(null);
        
        const isTraining = ref(false);
        const isStopping = ref(false);
        const hasChartData = ref(false); // 控制图表和占位符的切换
        const chartError = ref("");
        const showAdvanced = ref(false);   // 高级参数默认收起：家长保持默认即可，工程师可展开
        const baseModelOptions = ref([]);
        const allBaseModels = ref([]);
        const baseModelError = ref("");
        let baseModelPollTimer = null;

        // --- 发布弹窗状态 ---
        const showPublishModal = ref(false);
        const lastTrainedModelName = ref("");
        const isPublishing = ref(false);

        // --- 「管理微调好的模型」状态 ---
        const userModels = ref([]);          // 该用户微调好的模型列表（含 is_published / has_tflite / updated_at）
        const isLoadingModels = ref(false);
        const selectedModelName = ref("");   // 当前在 loss 看板回放曲线的模型名
        const deletingModel = ref("");       // 正在删除的模型名（控制按钮 spinner / 禁用）
        // 模型导出（ggml/tflite）状态：{ model_name: { ggml:{status,message}, tflite:{status} } }；轮询填充
        const exportStates = ref({});
        const uploadingModel = ref("");      // 正在上传 LoRA 的模型名
        let exportPollTimer = null;

        // --- 模型名称实时校验状态 ---
        // state: 'idle' | 'checking' | 'available' | 'taken' | 'invalid'
        // 用于在用户失焦输入框后即时反馈名称是否可用，避免填完所有参数才报重复
        const modelNameStatus = ref({ state: 'idle', message: '' });

        // 用户在输入时清空旧状态，避免显示 stale 的"已被使用"提示
        const clearModelNameStatus = () => {
            if (modelNameStatus.value.state !== 'idle') {
                modelNameStatus.value = { state: 'idle', message: '' };
            }
        };

        const checkModelNameAvailable = async () => {
            const name = (finetuneParams.value.model_name || "").trim();
            if (!name) {
                modelNameStatus.value = { state: 'idle', message: '' };
                return;
            }
            if (!props.currentUser) return;

            modelNameStatus.value = { state: 'checking', message: '正在检查名称…' };
            try {
                const url = `/api/check_model_name?model_name=${encodeURIComponent(name)}`;
                const res = await apiFetch(url, { cache: 'no-store' });
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();

                if (data.available) {
                    modelNameStatus.value = { state: 'available', message: '✓ 名称可用' };
                } else if (data.reason === 'taken') {
                    modelNameStatus.value = { state: 'taken', message: '⚠ 该名称已被使用，请换一个' };
                } else if (data.reason === 'too_long') {
                    modelNameStatus.value = { state: 'invalid', message: '⚠ 名称过长（最多 80 字符）' };
                } else if (data.reason === 'invalid_chars') {
                    modelNameStatus.value = {
                        state: 'invalid',
                        message: '⚠ 仅支持字母、数字、点、下划线、连字符，且首字符需为字母/数字/下划线'
                    };
                } else if (data.reason === 'empty') {
                    modelNameStatus.value = { state: 'idle', message: '' };
                } else {
                    modelNameStatus.value = { state: 'invalid', message: '⚠ 名称无效' };
                }
            } catch (e) {
                console.error("检查模型名称失败", e);
                // 网络出错时不阻塞用户，回到 idle，后端 start_finetune 仍会做最终把关
                modelNameStatus.value = { state: 'idle', message: '' };
            }
        };

        // 检查数据集状态（页面刷新后恢复步骤解锁状态）
        const checkDatasetStatus = async () => {
            if (!props.currentUser) return;
            try {
                const res = await apiFetch('/api/check_dataset', { cache: 'no-store' });
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                hasDataset.value = Boolean(data.has_dataset);
            } catch (e) {
                console.error("检查数据集状态失败", e);
            }
        };

        const refreshStepStatus = async () => {
            await checkDatasetStatus();
        };

        // 检查训练占用状态（解决页面刷新后 isTraining 丢失的问题）
        const checkTrainingStatus = async () => {
            if (!props.currentUser) return;
            try {
                const res = await apiFetch('/api/gpu_status', { cache: 'no-store' });
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();

                const userIsTraining = data.status === 'TRAINING' && data.current_user === props.currentUser;
                isTraining.value = userIsTraining;

                // 刷新后若该用户训练仍在进行，从后端拿到正在训练的 model_name，
                // 把它填回输入框并自动重连 SSE / 重载历史曲线
                if (userIsTraining && data.current_model_name) {
                    finetuneParams.value.model_name = data.current_model_name;
                    // 训练中 → 同步加载已经写入的 loss 曲线（不重置）+ 续连 SSE
                    await loadChartHistory(data.current_model_name);
                    if (!eventSource) {
                        await startSSE({ resetChart: false, modelName: data.current_model_name });
                    }
                }
            } catch (e) {
                console.error("检查训练状态失败", e);
            }
        };


        // 图表相关变量
        const chartRef = ref(null);
        let myChart = null;
        let eventSource = null;
        let sseErrorNotified = false;   // 避免 EventSource 自动重连时反复弹「连接中断」
        let trainLossData = [];
        let evalLossData = [];

        const datasetParams = ref({ test_ratio: 0.05 });
        const finetuneParams = ref({
            model_name: "",
            base_model: "",
            learning_rate: 0.0002,
            epochs: 20,
            batch_size: 8,
            gradient_accumulation_steps: 1,
            use_adalora: false,
            use_8bit: false,
            fp16: true,
            min_audio_len: 0.5,
            max_audio_len: 30.0
        });

        const enforceMinLen = () => {
            if (finetuneParams.value.min_audio_len > finetuneParams.value.max_audio_len - 0.5) {
                finetuneParams.value.min_audio_len = finetuneParams.value.max_audio_len - 0.5;
            }
        };
        const enforceMaxLen = () => {
            if (finetuneParams.value.max_audio_len < finetuneParams.value.min_audio_len + 0.5) {
                finetuneParams.value.max_audio_len = finetuneParams.value.min_audio_len + 0.5;
            }
        };

        const loadBaseModelOptions = async () => {
            try {
                baseModelError.value = "";
                const res = await apiFetch('/api/base_models', { cache: 'no-store' });
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                baseModelOptions.value = Array.isArray(data.models) ? data.models : [];
                allBaseModels.value = Array.isArray(data.all_models) ? data.all_models : [];
                if (!allBaseModels.value.length) {
                    baseModelError.value = "未读取到可配置的基础模型列表。";
                    finetuneParams.value.base_model = "";
                    return;
                }

                const selectedModelStillExists = allBaseModels.value.some(
                    model => model.name === finetuneParams.value.base_model
                );
                if (!selectedModelStillExists) {
                    finetuneParams.value.base_model = allBaseModels.value[0].name;
                }

                const downloadedModels = allBaseModels.value.filter(model => model.is_downloaded);
                const hasDownloadingModel = allBaseModels.value.some(model => model.download_status === 'downloading');

                if (!downloadedModels.length) {
                    baseModelError.value = hasDownloadingModel
                        ? "基础模型正在下载中，下载完成后即可开始训练。"
                        : "当前本地还没有基础模型，可先选择带 ☁ 标记的模型并点击下载。";
                } else if (!baseModelOptions.value.includes(finetuneParams.value.base_model)) {
                    baseModelError.value = "当前选择的基础模型尚未下载完成，训练前请先下载。";
                }

                if (hasDownloadingModel) {
                    startBaseModelPolling();
                } else {
                    stopBaseModelPolling();
                }
            } catch (err) {
                console.error("加载基础模型列表失败", err);
                baseModelError.value = "加载基础模型列表失败，请检查后端服务是否正常。";
                baseModelOptions.value = [];
                allBaseModels.value = [];
                finetuneParams.value.base_model = "";
                stopBaseModelPolling();
            }
        };

        const selectedBaseModelMeta = () => {
            return allBaseModels.value.find(model => model.name === finetuneParams.value.base_model) || null;
        };

        const isSelectedBaseModelDownloaded = () => {
            const selected = selectedBaseModelMeta();
            return selected ? selected.is_downloaded : false;
        };

        const isSelectedBaseModelDownloading = () => {
            const selected = selectedBaseModelMeta();
            return selected ? selected.download_status === 'downloading' : false;
        };

        const selectedBaseModelStatusText = () => {
            const selected = selectedBaseModelMeta();
            if (!selected) return "";
            if (selected.is_downloaded) return "本地已就绪，可直接训练。";
            if (selected.download_status === 'downloading') {
                return selected.download_message || "正在从云端下载模型...";
            }
            if (selected.download_status === 'failed') {
                return selected.download_message || "上一次下载失败，可重试。";
            }
            return "该模型暂未下载，本次训练前需要先拉到本地。";
        };

        const stopBaseModelPolling = () => {
            if (baseModelPollTimer) {
                clearInterval(baseModelPollTimer);
                baseModelPollTimer = null;
            }
        };

        const startBaseModelPolling = () => {
            if (baseModelPollTimer) return;
            baseModelPollTimer = setInterval(() => {
                loadBaseModelOptions();
            }, 3000);
        };

        const handleDownloadBaseModel = async () => {
            const selected = selectedBaseModelMeta();
            if (!selected || selected.is_downloaded || selected.download_status === 'downloading') {
                return;
            }

            try {
                const res = await apiFetch('/api/base_models/download', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_name: selected.name })
                });
                const data = await res.json();
                if (!res.ok) {
                    dialog.alert(data.detail || '启动下载失败，请稍后重试', { variant: 'danger' });
                    return;
                }
                await loadBaseModelOptions();
            } catch (err) {
                dialog.alert('网络请求失败，无法启动基础模型下载', { variant: 'danger' });
            }
        };

        // --- ECharts 图表初始化 ---
        // 图表配色随主题：深色面板用浅色文字/淡线，亮色面板用原深灰（避免白底上看不清）
        const chartAxisColors = () => (
            document.documentElement.dataset.theme !== 'light'
                ? { text: '#cbd5e1', tick: '#94a3b8', line: 'rgba(148,163,184,.45)', split: 'rgba(148,163,184,.18)' }
                : { text: '#6B7280', tick: '#9CA3AF', line: '#D1D5DB', split: '#E5E7EB' }
        );
        // 切主题时即时给已存在的图表换轴/图例颜色（series 蓝绿两色两主题通用，无需改）
        const applyChartTheme = () => {
            if (!myChart || myChart.isDisposed()) return;
            const c = chartAxisColors();
            myChart.setOption({
                legend: { textStyle: { color: c.text } },
                xAxis: { nameTextStyle: { color: c.text }, axisLabel: { color: c.tick }, axisLine: { lineStyle: { color: c.line } } },
                yAxis: { nameTextStyle: { color: c.text }, axisLabel: { color: c.tick }, splitLine: { lineStyle: { color: c.split } } }
            });
        };

        const initChart = () => {
            if (!chartRef.value) return false;
            if (typeof echarts === 'undefined') {
                chartError.value = "图表库加载失败，请检查网络后刷新页面";
                return false;
            }

            try {
                if (myChart && (!myChart.isDisposed()) && myChart.getDom() !== chartRef.value) {
                    myChart.dispose();
                    myChart = null;
                }

                if (!myChart) {
                    // 初始化 ECharts 实例
                    myChart = echarts.init(chartRef.value);
                }
                chartError.value = "";
                
                // 配置酷炫的蓝绿色渐变折线图（坐标轴/图例配色随主题：深色面板浅字 / 亮色面板深灰）
                const c = chartAxisColors();
                const option = {
                    legend: {
                        top: 10,
                        right: 16,
                        textStyle: { color: c.text, fontWeight: 'bold' },
                        data: ['训练集 Loss', '验证集 Eval Loss']
                    },
                    tooltip: { 
                        trigger: 'axis',
                        backgroundColor: 'rgba(255, 255, 255, 0.9)',
                        borderColor: '#E5E7EB',
                        textStyle: { color: '#374151', fontWeight: 'bold' }
                    },
                    grid: { left: '8%', right: '5%', bottom: '15%', top: '18%' },
                    xAxis: { 
                        type: 'value', 
                        name: '步数 (Step)', 
                        nameLocation: 'middle',
                        nameGap: 25,
                        nameTextStyle: { color: c.text, fontWeight: 'bold' },
                        axisLabel: { color: c.tick },
                        splitLine: { show: false },
                        axisLine: { lineStyle: { color: c.line } }
                    },
                    yAxis: {
                        type: 'value',
                        name: '误差 (Loss)',
                        nameTextStyle: { color: c.text, fontWeight: 'bold' },
                        axisLabel: { color: c.tick },
                        splitLine: { lineStyle: { type: 'dashed', color: c.split } }
                    },
                    series: [{
                        name: '训练集 Loss',
                        data: [],
                        type: 'line',
                        smooth: true, // 开启曲线平滑
                        symbol: 'circle',
                        symbolSize: 6,
                        itemStyle: { color: '#3B82F6' }, // 蓝色圆点
                        lineStyle: { width: 3, color: '#3B82F6' }, // 蓝色的线
                        areaStyle: { // 渐变填充区域
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                { offset: 0, color: 'rgba(59, 130, 246, 0.4)' },
                                { offset: 1, color: 'rgba(59, 130, 246, 0.0)' }
                            ])
                        }
                    },
                    {
                        name: '验证集 Eval Loss',
                        data: [],
                        type: 'line',
                        smooth: true,
                        symbol: 'diamond',
                        symbolSize: 7,
                        itemStyle: { color: '#10B981' },
                        lineStyle: { width: 3, color: '#10B981', type: 'dashed' },
                        areaStyle: {
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                { offset: 0, color: 'rgba(16, 185, 129, 0.25)' },
                                { offset: 1, color: 'rgba(16, 185, 129, 0.0)' }
                            ])
                        }
                    }]
                };
                myChart.setOption(option);
                return true;
            } catch (err) {
                chartError.value = "图表初始化失败，请刷新页面重试";
                console.error("初始化图表失败", err);
                return false;
            }
        };

        const ensureChartReady = async () => {
            await nextTick();
            if (initChart()) return true;

            await new Promise(resolve => setTimeout(resolve, 120));
            return initChart();
        };


        // 加载某个具体模型的训练历史曲线。
        // 日志路径按 {username}/{model_name} 隔离，不带 modelName 就无从加载。
        const loadChartHistory = async (modelName) => {
            if (!props.currentUser || !modelName) {
                // 没有 model 上下文时清空图表，避免显示陈旧数据
                trainLossData = [];
                evalLossData = [];
                if (myChart) {
                    myChart.setOption({ series: [{ data: [] }, { data: [] }] });
                }
                hasChartData.value = false;
                return;
            }

            const ready = await ensureChartReady();
            if (!ready || !myChart) {
                chartError.value = "图表初始化失败，无法恢复历史曲线";
                return;
            }

            try {
                const url = `/api/train_history?model_name=${encodeURIComponent(modelName)}`;
                const res = await apiFetch(url, { cache: 'no-store' });
                if (!res.ok) throw new Error(`HTTP ${res.status}`);

                const data = await res.json();

                trainLossData = Array.isArray(data.train_loss) ? data.train_loss : [];
                evalLossData = Array.isArray(data.eval_loss) ? data.eval_loss : [];

                myChart.setOption({
                    series: [
                        { data: trainLossData },
                        { data: evalLossData }
                    ]
                });
                myChart.resize();

                hasChartData.value = trainLossData.length > 0 || evalLossData.length > 0;
                chartError.value = "";
            } catch (e) {
                console.error("加载历史训练曲线失败", e);
            }
        };



        // --- 「管理微调好的模型」逻辑 ---

        // 把 unix 秒时间戳格式化成本地可读字符串，列表里展示训练完成时间
        const formatTime = (ts) => {
            if (!ts) return "";
            try {
                return new Date(ts * 1000).toLocaleString();
            } catch (e) {
                return "";
            }
        };

        // 拉取当前用户微调好的模型列表（后端已按 updated_at DESC 排好序）
        const loadUserModels = async () => {
            if (!props.currentUser) {
                userModels.value = [];
                return;
            }
            isLoadingModels.value = true;
            try {
                const res = await apiFetch('/api/user_models', { cache: 'no-store' });
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                userModels.value = Array.isArray(data.models) ? data.models : [];
            } catch (e) {
                console.error("加载微调模型列表失败", e);
                userModels.value = [];
            } finally {
                isLoadingModels.value = false;
            }
        };

        // 点「刷新」：先扫描磁盘把「在终端用 finetune.py 训练好、但还没登记进库」的模型补登记，
        // 再重新拉列表。这样命令行训练的模型也能出现在这里（网页端训练会自动登记，无需扫描）。
        const refreshUserModels = async () => {
            if (!props.currentUser) {
                userModels.value = [];
                return;
            }
            if (isLoadingModels.value) return;   // 扫描进行中不重复触发
            let added = [];
            isLoadingModels.value = true;
            try {
                const res = await apiFetch('/api/rescan_models', { method: 'POST' });
                if (res.ok) {
                    const data = await res.json();
                    added = Array.isArray(data.added) ? data.added : [];
                } else {
                    console.warn(`扫描新模型失败 HTTP ${res.status}`);
                }
            } catch (e) {
                // 扫描失败不阻断：仍照常刷新一次列表
                console.warn("扫描新模型请求失败（不阻断刷新）", e);
            } finally {
                isLoadingModels.value = false;
            }
            await loadUserModels();   // 重新拉列表（含磁盘校验）
            if (added.length) {
                await dialog.alert(
                    `扫描到 ${added.length} 个新模型并已加入列表：\n${added.join('、')}`,
                    { variant: 'success', title: '🔍 已发现新模型' }
                );
            }
        };

        // 点击某个模型 → 在 loss 看板回放它的历史训练曲线
        const viewModelCurve = async (model) => {
            if (!model || !model.model_name) return;
            // 训练时 SSE 正实时往看板推当前模型曲线，切换查看会互相干扰，故直接拦下
            if (isTraining.value) {
                await dialog.alert("训练进行中，看板正在实时绘制当前模型曲线，训练结束后可回看其它模型。", { variant: 'info' });
                return;
            }
            selectedModelName.value = model.model_name;
            await loadChartHistory(model.model_name);
        };

        // 删除某个微调模型（后端会删 output/{user}/{model}/ 整目录 + DB 记录）
        const handleDeleteModel = async (model) => {
            if (!model || !model.model_name) return;
            if (isTraining.value) {
                await dialog.alert("训练进行中，暂时无法删除模型，请等训练结束。", { variant: 'warning' });
                return;
            }

            let message = `确定要删除模型「${model.model_name}」吗？\n此操作会永久删除该模型的全部文件（LoRA 权重、TFLite、训练日志），无法恢复。`;
            if (model.is_published) {
                message += `\n\n⚠ 该模型当前已发布，删除后安卓客户端将无法再下载它，会回退到基础模型。`;
            }
            const ok = await dialog.confirm(message, {
                variant: 'danger', title: '删除模型', confirmText: '删除'
            });
            if (!ok) return;

            deletingModel.value = model.model_name;
            try {
                const res = await apiFetch('/api/delete_model', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_name: model.model_name })
                });
                const data = await res.json();
                if (res.ok) {
                    // 若删掉的正是当前正在看的模型，清空看板
                    if (selectedModelName.value === model.model_name) {
                        selectedModelName.value = "";
                        await loadChartHistory(null);
                    }
                    await loadUserModels();
                    await dialog.alert(`模型「${model.model_name}」已删除。`, { variant: 'success', title: '删除成功' });
                } else if (res.status === 423) {
                    await dialog.alert(data.detail || "GPU 正忙，请稍后再删除", { variant: 'warning' });
                } else {
                    await dialog.alert(`删除失败：${data.detail || '未知错误'}`, { variant: 'danger' });
                }
            } catch (e) {
                await dialog.alert("网络请求失败，无法删除模型", { variant: 'danger' });
            } finally {
                deletingModel.value = "";
            }
        };

        // --- 监听 SSE 流数据 ---
        // modelName 必填：日志路径按模型隔离，没有它就找不到对应日志文件
        const startSSE = async ({ resetChart = true, modelName } = {}) => {
            if (!modelName) {
                console.error("[SSE] startSSE 需要 modelName 参数");
                return;
            }
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }

            const ready = await ensureChartReady();
            if (!ready || !myChart) {
                chartError.value = "图表初始化失败，无法开始实时绘图";
                console.error("[SSE] chart is not ready, abort startSSE");
                return;
            }

            myChart.resize();

            if (resetChart) {
                trainLossData = [];
                evalLossData = [];
                myChart.setOption({ series: [{ data: [] }, { data: [] }] });
            }

            hasChartData.value = trainLossData.length > 0 || evalLossData.length > 0;

            // sseUrl() 自动把 JWT token 拼到 query（EventSource 不支持自定义 header）
            sseErrorNotified = false;
            eventSource = new EventSource(sseUrl('/api/train_stream', { model_name: modelName }));

            eventSource.onmessage = (event) => {
                let data = null;
                try {
                    data = JSON.parse(event.data);
                } catch (err) {
                    console.warn("[SSE] JSON parse failed:", event.data);
                    return;
                }

                sseErrorNotified = false;

                if (data.status === 'stopped') {
                    isTraining.value = false;
                    eventSource.close();
                    eventSource = null;
                    dialog.alert('训练已停止，本次进度未保存为可用模型。', { variant: 'info', title: '已停止' });
                    loadUserModels();
                    return;
                }

                if (data.status === 'finished') {
                    isTraining.value = false;
                    eventSource.close();
                    eventSource = null;

                    // 记录刚训练好的模型名称并弹出发布框
                    lastTrainedModelName.value = finetuneParams.value.model_name;
                    showPublishModal.value = true;
                    // 看板此刻正展示这个刚训练好的模型的曲线，让「正在查看」标签与之对应；
                    // 并刷新管理列表，新模型训练完即出现在「管理微调好的模型」区
                    selectedModelName.value = finetuneParams.value.model_name;
                    loadUserModels();
                    return;
                }

                if (data.status === 'error') {
                    isTraining.value = false;
                    eventSource.close();
                    eventSource = null;
                    dialog.alert("训练中断或发生错误：\n" + data.message, { variant: 'danger', title: '训练异常' });
                    return;
                }

                if (data.loss !== undefined && data.step !== undefined) {
                    trainLossData.push([data.step, data.loss]);
                    hasChartData.value = true;
                    if (myChart) {
                        myChart.setOption({ series: [{ data: trainLossData }, { data: evalLossData }] });
                        myChart.resize();
                    }
                }

                if (data.eval_loss !== undefined && data.step !== undefined) {
                    evalLossData.push([data.step, data.eval_loss]);
                    hasChartData.value = true;
                    if (myChart) {
                        myChart.setOption({ series: [{ data: trainLossData }, { data: evalLossData }] });
                        myChart.resize();
                    }
                }
            };

            eventSource.onerror = (err) => {
                console.error("[SSE] error =", err);
                // 正常 finished/stopped 已先 close()；仅当我们仍认为在训练且尚未提示过时，提醒一次
                if (isTraining.value && !sseErrorNotified) {
                    sseErrorNotified = true;
                    dialog.alert(
                        friendlyNetworkError() + '\n训练可能仍在后台继续，可稍后切到别的标签页再返回以重新连接查看进度。',
                        { variant: 'warning', title: '连接中断' }
                    );
                }
            };
        };

        // --- 动作：生成数据集 ---
        const handleBuildDataset = async () => {
            if (!props.currentUser) return;
            isBuilding.value = true;
            buildResult.value = null;

            try {
                const res = await apiFetch('/api/build_dataset', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        // username 由后端从 JWT 取，不再传
                        test_ratio: datasetParams.value.test_ratio
                    })
                });
                const data = await res.json();
                
                if (res.ok) {
                    buildResult.value = { success: true, train_count: data.train_count, test_count: data.test_count };
                    hasDataset.value = true;
                } else {
                    buildResult.value = { success: false, message: data.detail || "生成失败，请检查是否有已录音的任务。" };
                }
            } catch (err) {
                buildResult.value = { success: false, message: friendlyNetworkError() };
            } finally {
                isBuilding.value = false;
            }
        };

        // --- 动作：启动微调 ---
        const handleStartFinetune = async () => {
            if (!props.currentUser || !hasDataset.value) return;
            // CPU 环境拦在最前面：微调需要 GPU（语音识别仍可在 CPU 上用）。
            // 后端 start_finetune 也会再兜一层底（503）。
            if (!props.hasGpu) {
                await dialog.alert("当前为 CPU 环境，微调需要 GPU 平台。语音识别可以在 CPU 上正常使用。", { variant: 'warning' });
                return;
            }
            if (!finetuneParams.value.model_name.trim()) {
                await dialog.alert("请先输入模型名称", { variant: 'warning' });
                return;
            }
            // 如果失焦校验已经发现重名 / 名称无效，直接拦下来；
            // 后端 start_finetune 也会再 check 一遍兜底
            if (modelNameStatus.value.state === 'taken' || modelNameStatus.value.state === 'invalid') {
                await dialog.alert(modelNameStatus.value.message || "模型名称不可用", { variant: 'warning' });
                return;
            }
            if (!finetuneParams.value.base_model) {
                await dialog.alert("请先选择基础模型", { variant: 'warning' });
                return;
            }
            if (!isSelectedBaseModelDownloaded()) {
                await dialog.alert("当前基础模型还未下载到本地，请先点击下载", { variant: 'warning' });
                return;
            }
            
            isTraining.value = true;

            try {
                const res = await apiFetch('/api/start_finetune', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        // username 由后端从 JWT 取
                        model_name: finetuneParams.value.model_name.trim(),
                        base_model: finetuneParams.value.base_model,
                        learning_rate: finetuneParams.value.learning_rate,
                        epochs: finetuneParams.value.epochs,
                        accumulation_steps: finetuneParams.value.gradient_accumulation_steps,
                        batch_size: finetuneParams.value.batch_size,
                        use_adalora: finetuneParams.value.use_adalora,
                        use_8bit: finetuneParams.value.use_8bit,
                        fp16: finetuneParams.value.fp16,
                        min_audio_len: finetuneParams.value.min_audio_len,
                        max_audio_len: finetuneParams.value.max_audio_len
                    })
                });

                if (res.ok) {
                    // API 返回成功代表后台进程已成功拉起，开始连 SSE 监听图表数据！
                    await startSSE({
                        resetChart: true,
                        modelName: finetuneParams.value.model_name.trim()
                    });
                } else {
                    const msg = await friendlyHttpError(res, '启动失败，请稍后重试。');
                    if (msg) dialog.alert(msg, { variant: 'danger' });
                    isTraining.value = false;
                }
            } catch (err) {
                dialog.alert(friendlyNetworkError(), { variant: 'danger' });
                isTraining.value = false;
            }
        };

        // --- 动作：停止当前训练 ---
        const handleStopFinetune = async () => {
            if (!isTraining.value || isStopping.value) return;
            const ok = await dialog.confirm(
                '确定要停止当前训练吗？已训练的进度不会保存为可用模型。',
                { variant: 'danger', title: '停止训练', confirmText: '停止训练', cancelText: '继续训练' }
            );
            if (!ok) return;
            isStopping.value = true;
            try {
                const res = await apiFetch('/api/stop_finetune', { method: 'POST' });
                if (!res.ok) throw new Error(await friendlyHttpError(res));
                // 不在这里翻 isTraining —— 交给 SSE 的 stopped 帧统一处理（单一真相源）
            } catch (e) {
                const msg = (e instanceof TypeError) ? friendlyNetworkError() : e.message;
                if (msg) dialog.alert(msg, { variant: 'danger' });
            } finally {
                isStopping.value = false;
            }
        };

        // 窗口大小改变时，图表自适应缩放
        const handleResize = () => { if (myChart) myChart.resize(); };

        // 生命周期管理
        // 注意：不再在 onMounted/onActivated 里无条件调 loadChartHistory()，
        // 因为日志现在按模型隔离，没有 model_name 上下文就不知道加载哪个。
        // 由 checkTrainingStatus 在发现"用户正在训练"时主动 loadChartHistory(currentModel)。
        onMounted(async () => {
            await ensureChartReady();
            if (myChart) myChart.resize();

            window.addEventListener('resize', handleResize);
            window.addEventListener('whisper:theme', applyChartTheme);

            await refreshStepStatus();
            await loadBaseModelOptions();
            await loadUserModels();
            await fetchExportStatus();
            if (anyExporting()) startExportPolling();
            await checkTrainingStatus();
        });

        onActivated(async () => {
            await ensureChartReady();
            if (myChart) myChart.resize();

            await refreshStepStatus();
            await loadBaseModelOptions();
            await loadUserModels();
            await fetchExportStatus();
            if (anyExporting()) startExportPolling();
            await checkTrainingStatus();
        });

        watch(
            () => props.currentUser,
            async (newUser) => {
                if (!newUser) {
                    hasDataset.value = false;
                    isTraining.value = false;
                    hasChartData.value = false;
                    trainLossData = [];
                    evalLossData = [];
                    stopBaseModelPolling();

                    if (myChart) {
                        myChart.setOption({ series: [{ data: [] }, { data: [] }] });
                    }

                    if (eventSource) {
                        eventSource.close();
                        eventSource = null;
                    }
                    finetuneParams.value.model_name = "";
                    finetuneParams.value.base_model = "";
                    modelNameStatus.value = { state: 'idle', message: '' };
                    userModels.value = [];
                    selectedModelName.value = "";
                    deletingModel.value = "";
                    return;
                }

                await refreshStepStatus();
                await loadBaseModelOptions();
                await loadUserModels();
                // loadChartHistory 由 checkTrainingStatus 在需要时按 model_name 主动调用
                await checkTrainingStatus();
            }
        );

        onUnmounted(() => {
            if (eventSource) eventSource.close();
            if (myChart) myChart.dispose();
            stopBaseModelPolling();
            stopExportPolling();
            window.removeEventListener('resize', handleResize);
            window.removeEventListener('whisper:theme', applyChartTheme);
        });

        // --- 动作：正式发布模型 ---
        const handlePublishModel = async () => {
            if (!lastTrainedModelName.value) return;
            isPublishing.value = true;
            try {
                const response = await apiFetch('/api/publish_model', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        // username 由后端从 JWT 取
                        model_name: lastTrainedModelName.value
                    })
                });
                const data = await response.json();
                if (response.ok) {
                    await dialog.alert('发布成功！客户端现在可以检测到更新了。', {
                        variant: 'success', title: '🚀 发布成功'
                    });
                    showPublishModal.value = false;
                } else {
                    await dialog.alert('发布失败：' + data.detail, { variant: 'danger' });
                }
            } catch (e) {
                await dialog.alert('网络请求失败，无法发布模型', { variant: 'danger' });
            } finally {
                isPublishing.value = false;
            }
        };

        const closePublishModal = () => {
            showPublishModal.value = false;
        };

        // --- 模型导出管理（ggml / tflite / 发布）方案 A：三个独立动作 ---
        const exportStatusOf = (modelName, kind) =>
            (exportStates.value[modelName] && exportStates.value[modelName][kind] && exportStates.value[modelName][kind].status) || 'idle';
        const isExporting = (modelName, kind) => exportStatusOf(modelName, kind) === 'running';
        const exportError = (modelName) => {
            const m = exportStates.value[modelName] || {};
            if (m.ggml && m.ggml.status === 'error') return 'ggml: ' + (m.ggml.message || '失败');
            if (m.tflite && m.tflite.status === 'error') return 'TFLite: ' + (m.tflite.message || '失败');
            return '';
        };
        const anyExporting = () => Object.values(exportStates.value || {}).some(
            (m) => (m.ggml && m.ggml.status === 'running') || (m.tflite && m.tflite.status === 'running'));

        const fetchExportStatus = async () => {
            try {
                const res = await apiFetch('/api/export_status', { cache: 'no-store' });
                if (res.ok) exportStates.value = await res.json();
            } catch (e) { /* 忽略，下次轮询再试 */ }
        };
        const stopExportPolling = () => { if (exportPollTimer) { clearInterval(exportPollTimer); exportPollTimer = null; } };
        const startExportPolling = () => {
            if (exportPollTimer) return;
            exportPollTimer = setInterval(async () => {
                await fetchExportStatus();
                if (!anyExporting()) { stopExportPolling(); await loadUserModels(); }   // 导出完 → 刷新徽章
            }, 2000);
        };

        const exportModel = async (model, kind) => {   // kind: 'ggml' | 'tflite'
            if (isTraining.value) { await dialog.alert('训练进行中，暂时无法导出', { variant: 'warning' }); return; }
            try {
                const res = await apiFetch(`/api/export_${kind}`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_name: model.model_name }),
                });
                if (!res.ok) { const d = await res.json(); await dialog.alert('导出失败：' + d.detail, { variant: 'warning' }); return; }
                // 乐观置 running + 启动轮询
                exportStates.value = {
                    ...exportStates.value,
                    [model.model_name]: { ...(exportStates.value[model.model_name] || {}), [kind]: { status: 'running' } },
                };
                startExportPolling();
            } catch (e) {
                await dialog.alert('网络请求失败，无法开始导出', { variant: 'danger' });
            }
        };

        // 上传该模型的 LoRA 权重到 ModelScope（共享仓库 <用户名>/<模型名>/ 子目录，见后端 /api/upload_lora）
        const uploadLoraToModelscope = async (model) => {
            if (!model || !model.model_name) return;
            if (isTraining.value) { await dialog.alert('训练进行中，暂时无法上传，请等训练结束。', { variant: 'warning' }); return; }
            const ok = await dialog.confirm(`将「${model.model_name}」的 LoRA 权重上传到 ModelScope？（用于备份 / 换机同步）`);
            if (!ok) return;
            uploadingModel.value = model.model_name;
            try {
                const res = await apiFetch('/api/upload_lora', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_name: model.model_name }),
                });
                const data = await res.json();
                if (res.ok) {
                    await dialog.alert(`LoRA 已上传到 ModelScope：\n${data.repo_id}/${data.path_in_repo}/`, { variant: 'success', title: '☁️ 上传成功' });
                } else {
                    await dialog.alert('上传失败：' + (data.detail || '未知错误'), { variant: 'danger' });
                }
            } catch (e) {
                await dialog.alert('网络请求失败，无法上传 LoRA', { variant: 'danger' });
            } finally {
                uploadingModel.value = "";
            }
        };

        // --- 后端命令预览：显示后端将执行的 finetune.py 命令，可复制到终端手动运行 ---
        const finetuneCommand = ref("");
        const commandCopied = ref(false);
        let commandTimer = null;

        const fetchFinetuneCommand = async () => {
            try {
                const res = await apiFetch('/api/finetune_command', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model_name: (finetuneParams.value.model_name || '').trim(),
                        base_model: finetuneParams.value.base_model,
                        learning_rate: finetuneParams.value.learning_rate,
                        epochs: finetuneParams.value.epochs,
                        accumulation_steps: finetuneParams.value.gradient_accumulation_steps,
                        batch_size: finetuneParams.value.batch_size,
                        use_adalora: finetuneParams.value.use_adalora,
                        use_8bit: finetuneParams.value.use_8bit,
                        fp16: finetuneParams.value.fp16,
                        min_audio_len: finetuneParams.value.min_audio_len,
                        max_audio_len: finetuneParams.value.max_audio_len
                    })
                });
                if (res.ok) {
                    const data = await res.json();
                    finetuneCommand.value = data.command || "";
                }
            } catch (e) {
                // 预览失败不打扰用户，保留上一次结果
            }
        };

        // 防抖：参数频繁变化（拖滑块）时只在停顿后请求一次
        const scheduleCommandRefresh = () => {
            if (commandTimer) clearTimeout(commandTimer);
            commandTimer = setTimeout(fetchFinetuneCommand, 400);
        };
        watch(finetuneParams, scheduleCommandRefresh, { deep: true, immediate: true });

        const copyFinetuneCommand = async () => {
            if (!finetuneCommand.value) return;
            try {
                await navigator.clipboard.writeText(finetuneCommand.value);
                commandCopied.value = true;
                setTimeout(() => { commandCopied.value = false; }, 1500);
            } catch (e) {
                await dialog.alert('复制失败，请手动选中命令文本复制', { variant: 'warning' });
            }
        };

        return {
            chartRef,
            datasetParams,
            finetuneParams,
            hasDataset,
            isBuilding,
            buildResult,
            isTraining,
            hasChartData,
            chartError,
            showAdvanced,
            baseModelOptions,
            allBaseModels,
            baseModelError,
            selectedBaseModelMeta,
            isSelectedBaseModelDownloaded,
            isSelectedBaseModelDownloading,
            selectedBaseModelStatusText,
            handleBuildDataset,
            handleDownloadBaseModel,
            handleStartFinetune,
            isStopping,
            handleStopFinetune,
            enforceMinLen,
            enforceMaxLen,
            showPublishModal,
            lastTrainedModelName,
            isPublishing,
            handlePublishModel,
            closePublishModal,
            modelNameStatus,
            checkModelNameAvailable,
            clearModelNameStatus,
            userModels,
            isLoadingModels,
            selectedModelName,
            deletingModel,
            exportStates,
            exportStatusOf,
            isExporting,
            exportError,
            uploadingModel,
            exportModel,
            uploadLoraToModelscope,
            formatTime,
            loadUserModels,
            refreshUserModels,
            viewModelCurve,
            handleDeleteModel,
            finetuneCommand,
            commandCopied,
            copyFinetuneCommand
        };

    }
}
