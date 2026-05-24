const { ref, computed, onMounted, onUnmounted, nextTick } = Vue;
import * as dialog from './dialog.js?v=1.1';
import { apiFetch, sseUrl } from './api.js?v=1.1';

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
            // username 由后端从 JWT 提取，不再传 query；`t=...` 用于绕过浏览器缓存
            const res = await apiFetch(`/api/tasks?t=${new Date().getTime()}`);
            if (!res.ok) return;  // 401 已被 apiFetch 处理，其他错误静默
            tasks.value = await res.json();
        };

        const uploadTxt = async () => {
            const file = txtInput.value.files[0];
            if (!file) return dialog.alert("请先选择一个 TXT 文件！", { variant: 'warning' });
            const formData = new FormData();
            formData.append("file", file);
            await apiFetch('/api/upload_txt', { method: 'POST', body: formData });
            txtInput.value.value = "";
            fetchTasks();
        };

        const addTask = async () => {
            if (!newTaskText.value.trim()) return;
            await apiFetch('/api/task', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: newTaskText.value })
            });
            newTaskText.value = "";
            fetchTasks();
        };

        const startEditing = (task) => { editingTaskId.value = task.id; editingTaskText.value = task.text_content; };
        const cancelEdit = () => { editingTaskId.value = null; editingTaskText.value = ""; };
        const saveEdit = async (taskId) => {
            const newText = editingTaskText.value.trim();
            if (!newText) return dialog.alert("文本不能为空！", { variant: 'warning' });

            // 找到原任务，判断是否有录音且文本是否真的改了。
            // 如果文本变了且已经录过音，提醒用户：保存就会丢失旧录音。
            const original = tasks.value.find(t => t.id === taskId);
            const textChanged = original ? original.text_content !== newText : true;
            const hasAudio = !!(original && original.is_completed);
            if (textChanged && hasAudio) {
                const ok = await dialog.confirm(
                    "修改文本会丢弃这一条已录制的音频，需要重新录制。确定继续吗？",
                    { variant: 'warning', confirmText: '确定修改', cancelText: '保留原录音' }
                );
                if (!ok) return;
            }

            await apiFetch(`/api/task/${taskId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: newText })
            });
            editingTaskId.value = null;
            fetchTasks();
        };

        const deleteTask = async (taskId) => {
            const ok = await dialog.confirm(
                "确定要删除这条录音任务吗？已录制的音频也会被永久删除！",
                { variant: 'danger', confirmText: '删除', cancelText: '取消' }
            );
            if (!ok) return;
            await apiFetch(`/api/task/${taskId}`, { method: 'DELETE' });
            fetchTasks();
        };

        const clearAllTasks = async () => {
            const ok = await dialog.confirm(
                `确定要清空代号 [${props.currentUser}] 的所有任务和音频吗？此操作无法撤销！`,
                { variant: 'danger', confirmText: '全部清空', cancelText: '取消' }
            );
            if (!ok) return;
            await apiFetch('/api/tasks', { method: 'DELETE' });
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
                dialog.alert("无法访问麦克风，请确保浏览器已授权！", { variant: 'danger' });
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
            const shouldAdvanceFocus = (
                focusedTaskIndex.value !== null &&
                focusedTask.value &&
                focusedTask.value.id === taskId &&
                focusedTaskIndex.value < tasks.value.length - 1
            );
            try {
                const response = await apiFetch(`/api/upload_audio/${taskId}`, { method: 'POST', body: formData });
                const result = await response.json();
                if (!response.ok) throw new Error(result.detail || `Upload failed with status ${response.status}`);
                if (result.error) throw new Error(result.error);
                uploadSucceeded = true;
            } catch (e) {
                dialog.alert(e.message || "保存失败！", { variant: 'danger' });
            } finally {
                processingTaskId.value = null;
                await fetchTasks();
                await nextTick();

                // 录音上传成功后，直接复用沉浸模式“向右切换”的现有前端逻辑
                if (uploadSucceeded && shouldAdvanceFocus) {
                    nextFocusTask();
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

        // 生成受保护音频接口的 URL；token 走 query 因为 <audio> 不能附带 header
        const audioSrc = (task) => {
            if (!task || !task.is_completed) return null;
            return sseUrl(`/api/audio/${task.id}`, { v: task.updated_at });
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
            focusedTaskIndex, focusedTask, openFocusMode, closeFocusMode, prevFocusTask, nextFocusTask,
            audioSrc
        }
    }
}
