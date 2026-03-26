const { ref, onMounted, onUnmounted, nextTick } = Vue;

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

        // 评估模块状态
        const hasModel = ref(false);      // 是否已生成模型
        const isEvaluating = ref(false);  // 是否正在评估
        const evalResult = ref(null);     // 评估分数(CER)

        // 检查模型状态
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
            if (chartRef.value && !myChart) {
                // 初始化 ECharts 实例
                myChart = echarts.init(chartRef.value);
                
                // 配置酷炫的蓝绿色渐变折线图
                const option = {
                    tooltip: { 
                        trigger: 'axis',
                        backgroundColor: 'rgba(255, 255, 255, 0.9)',
                        borderColor: '#E5E7EB',
                        textStyle: { color: '#374151', fontWeight: 'bold' }
                    },
                    grid: { left: '8%', right: '5%', bottom: '15%', top: '10%' },
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
                    }]
                };
                myChart.setOption(option);
            }
        };

        // --- 监听 SSE 流数据 ---
        const startSSE = () => {
            // 如果之前有连接，先关掉
            if (eventSource) eventSource.close();
            
            // 清空图表数据，准备接收新一轮训练数据
            if (myChart) myChart.setOption({ series: [{ data: [] }] });
            hasChartData.value = true;

            // 建立长连接
            eventSource = new EventSource(`/api/train_stream?username=${encodeURIComponent(props.currentUser)}`);

            eventSource.onmessage = (event) => {
                let data = null;
                try {
                    data = JSON.parse(event.data);
                } catch (err) {
                    console.warn("收到无法解析的训练日志，已跳过：", event.data);
                    return;
                }
                
                // 判断是否是控制信号
                if (data.status === 'finished') {
                    isTraining.value = false;
                    eventSource.close();
                    alert("🎉 恭喜！模型专属微调已成功完成！");
                    
                    checkModelStatus();
                    return;
                }
                if (data.status === 'error') {
                    isTraining.value = false;
                    eventSource.close();
                    alert("⚠️ 训练中断或发生错误: " + data.message);
                    return;
                }

                // 如果是常规的 log 数据 (包含了 step 和 loss)
                if (data.loss !== undefined && data.step !== undefined) {
                    if (!myChart) return;
                    // 获取当前图表的数据数组
                    const currentOption = myChart.getOption();
                    const currentData = currentOption.series[0].data;
                    
                    // 追加新的坐标点 [x, y]
                    currentData.push([data.step, data.loss]);
                    
                    // 增量更新图表
                    myChart.setOption({
                        series: [{ data: currentData }]
                    });
                }
            };

            eventSource.onerror = (err) => {
                console.error("SSE 连接异常，可能是网络波动或后端重启", err);
                // 这里不需要主动 close()，浏览器自带断线重连机制
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
                    startSSE();
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
        onMounted(() => {
            // 使用 nextTick 确保 DOM 已经渲染完毕再挂载图表
            nextTick(() => {
                initChart();
            });
            window.addEventListener('resize', handleResize);
            checkModelStatus();
        });

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
