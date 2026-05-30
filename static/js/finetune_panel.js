const { ref, onMounted, onUnmounted, onActivated, nextTick, watch } = Vue;
import * as dialog from './dialog.js?v=1.2';
import { apiFetch, sseUrl } from './api.js?v=1.2';

export default {
    template: '#tpl-finetune-panel',
    props: ['currentUser'],
    setup(props) {
        // --- 状态与参数 ---
        const hasDataset = ref(false);
        const isBuilding = ref(false);
        const buildResult = ref(null);
        
        const isTraining = ref(false);
        const hasChartData = ref(false); // 控制图表和占位符的切换
        const chartError = ref("");
        const baseModelOptions = ref([]);
        const allBaseModels = ref([]);
        const baseModelError = ref("");
        let baseModelPollTimer = null;

        // --- 发布弹窗状态 ---
        const showPublishModal = ref(false);
        const lastTrainedModelName = ref("");
        const isPublishing = ref(false);

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
                
                // 配置酷炫的蓝绿色渐变折线图
                const option = {
                    legend: {
                        top: 10,
                        right: 16,
                        textStyle: { color: '#6B7280', fontWeight: 'bold' },
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
                        nameTextStyle: { color: '#9CA3AF', fontWeight: 'bold' },
                        splitLine: { show: false },
                        axisLine: { lineStyle: { color: '#D1D5DB' } }
                    },
                    yAxis: { 
                        type: 'value', 
                        name: '误差 (Loss)', 
                        nameTextStyle: { color: '#9CA3AF', fontWeight: 'bold' },
                        splitLine: { lineStyle: { type: 'dashed', color: '#E5E7EB' } }
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
            eventSource = new EventSource(sseUrl('/api/train_stream', { model_name: modelName }));

            eventSource.onmessage = (event) => {
                let data = null;
                try {
                    data = JSON.parse(event.data);
                } catch (err) {
                    console.warn("[SSE] JSON parse failed:", event.data);
                    return;
                }

                if (data.status === 'finished') {
                    isTraining.value = false;
                    eventSource.close();
                    eventSource = null;
                    
                    // 记录刚训练好的模型名称并弹出发布框
                    lastTrainedModelName.value = finetuneParams.value.model_name;
                    showPublishModal.value = true;
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
                buildResult.value = { success: false, message: "网络请求失败，请检查后端服务。" };
            } finally {
                isBuilding.value = false;
            }
        };

        // --- 动作：启动微调 ---
        const handleStartFinetune = async () => {
            if (!props.currentUser || !hasDataset.value) return;
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
                    const errorData = await res.json();
                    dialog.alert(`启动失败：${errorData.detail}`, { variant: 'danger' });
                    isTraining.value = false;
                }
            } catch (err) {
                dialog.alert("网络请求失败，无法启动训练！", { variant: 'danger' });
                isTraining.value = false;
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

            await refreshStepStatus();
            await loadBaseModelOptions();
            await checkTrainingStatus();
        });

        onActivated(async () => {
            await ensureChartReady();
            if (myChart) myChart.resize();

            await refreshStepStatus();
            await loadBaseModelOptions();
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
                    return;
                }

                await refreshStepStatus();
                await loadBaseModelOptions();
                // loadChartHistory 由 checkTrainingStatus 在需要时按 model_name 主动调用
                await checkTrainingStatus();
            }
        );

        onUnmounted(() => {
            if (eventSource) eventSource.close();
            if (myChart) myChart.dispose();
            stopBaseModelPolling();
            window.removeEventListener('resize', handleResize);
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
            enforceMinLen,
            enforceMaxLen,
            showPublishModal,
            lastTrainedModelName,
            isPublishing,
            handlePublishModel,
            closePublishModal,
            modelNameStatus,
            checkModelNameAvailable,
            clearModelNameStatus
        };

    }
}
