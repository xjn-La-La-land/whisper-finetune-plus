const { ref, computed, onMounted, onUnmounted } = Vue;

export default {
    template: '#tpl-audio-collector',
    // 接收外壳传进来的当前登录用户名
    props: ['currentUser'], 
    setup(props) {
        // --- 业务状态 ---
        const tasks = ref([]);
        const txtInput = ref(null);
        const newTaskText = ref("");
        const editingTaskId = ref(null);
        const editingTaskText = ref("");
        const recordingTaskId = ref(null);
        const processingTaskId = ref(null);
        
        let mediaRecorder = null;
        let audioChunks = [];

        // --- 沉浸式模式状态 ---
        const focusedTaskIndex = ref(null);
        const focusedTask = computed(() => {
            if (focusedTaskIndex.value === null || tasks.value.length === 0) return null;
            return tasks.value[focusedTaskIndex.value];
        });

        // --- 进度计算 ---
        const completedCount = computed(() => tasks.value.filter(task => task.is_completed).length);
        const progressPercentage = computed(() => tasks.value.length === 0 ? 0 : Math.round((completedCount.value / tasks.value.length) * 100));

        // --- API 交互 (使用 props.currentUser) ---
        const fetchTasks = async () => {
            if (!props.currentUser) return;
            const res = await fetch(`/api/tasks?username=${encodeURIComponent(props.currentUser)}&t=${new Date().getTime()}`);
            tasks.value = await res.json();
        };

        const uploadTxt = async () => {
            const file = txtInput.value.files[0];
            if (!file) return alert("请先选择一个 TXT 文件！");
            const formData = new FormData();
            formData.append("file", file);
            await fetch(`/api/upload_txt?username=${encodeURIComponent(props.currentUser)}`, { method: 'POST', body: formData });
            txtInput.value.value = "";
            fetchTasks();
        };

        const addTask = async () => {
            if (!newTaskText.value.trim()) return;
            await fetch(`/api/task?username=${encodeURIComponent(props.currentUser)}`, { 
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: newTaskText.value }) 
            });
            newTaskText.value = "";
            fetchTasks();
        };

        const startEditing = (task) => { editingTaskId.value = task.id; editingTaskText.value = task.text_content; };
        const cancelEdit = () => { editingTaskId.value = null; editingTaskText.value = ""; };
        const saveEdit = async (taskId) => {
            if (!editingTaskText.value.trim()) return alert("文本不能为空！");
            await fetch(`/api/task/${taskId}?username=${encodeURIComponent(props.currentUser)}`, { 
                method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: editingTaskText.value }) 
            });
            editingTaskId.value = null;
            fetchTasks();
        };

        const deleteTask = async (taskId) => {
            if (!confirm("确定要删除这条录音任务吗？已录制的音频也会被永久删除！")) return;
            await fetch(`/api/task/${taskId}?username=${encodeURIComponent(props.currentUser)}`, { method: 'DELETE' });
            fetchTasks();
        };

        const clearAllTasks = async () => {
            if (!confirm(`确定要清空代号 [${props.currentUser}] 的所有任务和音频吗？`)) return;
            await fetch(`/api/tasks?username=${encodeURIComponent(props.currentUser)}`, { method: 'DELETE' });
            fetchTasks();
        };

        // --- 录音逻辑 ---
        const startRecording = async (taskId) => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                mediaRecorder.ondataavailable = (event) => { if (event.data.size > 0) audioChunks.push(event.data); };
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    await uploadAudio(taskId, audioBlob);
                };
                mediaRecorder.start();
                recordingTaskId.value = taskId;
            } catch (err) {
                alert("无法访问麦克风，请确保浏览器已授权！");
            }
        };

        const stopRecording = (taskId) => {
            if (mediaRecorder && mediaRecorder.state !== "inactive") {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
            recordingTaskId.value = null;
            processingTaskId.value = taskId;
        };

        const uploadAudio = async (taskId, audioBlob) => {
            const formData = new FormData();
            formData.append("audio", audioBlob, "record.webm");
            let uploadSucceeded = false;
            try {
                const response = await fetch(`/api/upload_audio/${taskId}?username=${encodeURIComponent(props.currentUser)}`, { method: 'POST', body: formData });
                if (!response.ok) throw new Error(`Upload failed with status ${response.status}`);
                uploadSucceeded = true;
            } catch (e) {
                alert("保存失败！");
            } finally {
                processingTaskId.value = null;
                await fetchTasks();

                // 仅在沉浸式模式中生效：当前任务上传成功后，自动切换到下一条任务
                if (uploadSucceeded && focusedTaskIndex.value !== null) {
                    const currentIndex = tasks.value.findIndex(task => task.id === taskId);
                    if (currentIndex !== -1 && focusedTaskIndex.value === currentIndex && currentIndex < tasks.value.length - 1) {
                        focusedTaskIndex.value = currentIndex + 1;
                    }
                }
            }
        };

        // --- 沉浸式模式控制逻辑 ---
        const openFocusMode = (index) => { focusedTaskIndex.value = index; };
        const closeFocusMode = () => { focusedTaskIndex.value = null; };
        const prevFocusTask = () => { if (focusedTaskIndex.value > 0) focusedTaskIndex.value--; };
        const nextFocusTask = () => { if (focusedTaskIndex.value < tasks.value.length - 1) focusedTaskIndex.value++; };

        const handleKeydown = (e) => {
            if (focusedTaskIndex.value === null) return;
            if (e.key === 'Escape') closeFocusMode();
            if (e.key === 'ArrowLeft') prevFocusTask();
            if (e.key === 'ArrowRight') nextFocusTask();
        };

        // 组件挂载时拉取数据并监听键盘
        onMounted(() => {
            fetchTasks();
            window.addEventListener('keydown', handleKeydown);
        });

        // 组件卸载（比如切换到了微调 Tab 时）清理键盘监听，防止内存泄漏
        onUnmounted(() => {
            window.removeEventListener('keydown', handleKeydown);
        });

        return { 
            tasks, txtInput, uploadTxt, startRecording, stopRecording, recordingTaskId, processingTaskId,
            newTaskText, addTask, editingTaskId, editingTaskText, startEditing, saveEdit, cancelEdit, deleteTask, clearAllTasks,
            completedCount, progressPercentage,
            focusedTaskIndex, focusedTask, openFocusMode, closeFocusMode, prevFocusTask, nextFocusTask
        }
    }
}
