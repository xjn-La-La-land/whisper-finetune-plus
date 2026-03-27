const { ref, onMounted, onUnmounted, onActivated, nextTick, watch } = Vue;

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

        // 评估模块状态
        const hasModel = ref(false);      // 是否已生成模型
        const isEvaluating = ref(false);  // 是否正在评估
        const evalResult = ref(null);     // 评估分数(CER)

        // 检查模型状态
        const checkModelStatus = async () => {
            if (!props.currentUser) return;
            try {
                const res = await fetch(`/api/check_model?username=${encodeURIComponent(props.currentUser)}`, {
                    cache: 'no-store'
                });
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                hasModel.value = data.has_model;
            } catch (e) {
                console.error("检查模型状态失败", e);
            }
        };


        // 检查数据集状态（页面刷新后恢复步骤解锁状态）
        const checkDatasetStatus = async () => {
            if (!props.currentUser) return;
            try {
                const res = await fetch(`/api/check_dataset?username=${encodeURIComponent(props.currentUser)}`, {
                    cache: 'no-store'
                });
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                hasDataset.value = Boolean(data.has_dataset);
            } catch (e) {
                console.error("检查数据集状态失败", e);
            }
        };

        const refreshStepStatus = async () => {
            await Promise.all([checkDatasetStatus(), checkModelStatus()]);
        };

        // 检查训练占用状态（解决页面刷新后 isTraining 丢失的问题）
        const checkTrainingStatus = async () => {
            if (!props.currentUser) return;
            try {
                const res = await fetch('/api/gpu_status', { cache: 'no-store' });
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();

                const userIsTraining = data.status === 'TRAINING' && data.current_user === props.currentUser;
                isTraining.value = userIsTraining;

                // 刷新后若该用户训练仍在进行，自动重连 SSE
                if (userIsTraining && !eventSource) {
                    await startSSE({ resetChart: false });
                }
            } catch (e) {
                console.error("检查训练状态失败", e);
            }
        };


        // 发起评估请求
        const handleEvaluate = async () => {
            if (!props.currentUser) return;
            isEvaluating.value = true;
            evalResult.value = null;

            try {
                const res = await fetch('/api/evaluate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        username: props.currentUser,
                        batch_size: 8 // 测试集较小，batch给固定值即可
                    })
                });

                const data = await res.json();
                if (res.ok) {
                    evalResult.value = data.cer;
                } else {
                    alert(`评估失败: ${data.detail}`);
                }
            } catch (err) {
                alert("请求评估接口失败！");
            } finally {
                isEvaluating.value = false;
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


        const loadChartHistory = async () => {
            if (!props.currentUser) return;

            const ready = await ensureChartReady();
            if (!ready || !myChart) {
                chartError.value = "图表初始化失败，无法恢复历史曲线";
                return;
            }

            try {
                const res = await fetch(`/api/train_history?username=${encodeURIComponent(props.currentUser)}`, {
                    cache: 'no-store'
                });
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
        const startSSE = async ({ resetChart = true } = {}) => {
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

            eventSource = new EventSource(`/api/train_stream?username=${encodeURIComponent(props.currentUser)}`);

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
                    alert("🎉 恭喜！模型专属微调已成功完成！");
                    checkModelStatus();
                    return;
                }

                if (data.status === 'error') {
                    isTraining.value = false;
                    eventSource.close();
                    eventSource = null;
                    alert("⚠️ 训练中断或发生错误: " + data.message);
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
                const res = await fetch('/api/build_dataset', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        username: props.currentUser,
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
            
            isTraining.value = true;

            try {
                const res = await fetch('/api/start_finetune', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        username: props.currentUser,
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
                    await startSSE({ resetChart: true });
                } else {
                    const errorData = await res.json();
                    alert(`启动失败: ${errorData.detail}`);
                    isTraining.value = false;
                }
            } catch (err) {
                alert("网络请求失败，无法启动训练！");
                isTraining.value = false;
            }
        };

        // 窗口大小改变时，图表自适应缩放
        const handleResize = () => { if (myChart) myChart.resize(); };

        // 生命周期管理
        onMounted(async () => {
            await ensureChartReady();
            if (myChart) myChart.resize();

            window.addEventListener('resize', handleResize);

            await refreshStepStatus();
            await loadChartHistory();
            await checkTrainingStatus();
        });

        onActivated(async () => {
            await ensureChartReady();
            if (myChart) myChart.resize();

            await refreshStepStatus();
            await loadChartHistory();
            await checkTrainingStatus();
        });

        watch(
            () => props.currentUser,
            async (newUser) => {
                if (!newUser) {
                    hasDataset.value = false;
                    hasModel.value = false;
                    isTraining.value = false;
                    hasChartData.value = false;
                    trainLossData = [];
                    evalLossData = [];

                    if (myChart) {
                        myChart.setOption({ series: [{ data: [] }, { data: [] }] });
                    }

                    if (eventSource) {
                        eventSource.close();
                        eventSource = null;
                    }
                    return;
                }

                await refreshStepStatus();
                await loadChartHistory();
                await checkTrainingStatus();
            }
        );

        onUnmounted(() => {
            if (eventSource) eventSource.close();
            if (myChart) myChart.dispose();
            window.removeEventListener('resize', handleResize);
        });

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
            handleBuildDataset,
            handleStartFinetune,
            enforceMinLen,
            enforceMaxLen,
            hasModel,
            isEvaluating,
            evalResult,
            handleEvaluate
        };
    }
}
