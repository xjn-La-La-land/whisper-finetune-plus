const { ref } = Vue;

export default {
    props: ['src'],
    template: '#tpl-custom-audio',
    setup(props) {
        const audioEl = ref(null);
        const isPlaying = ref(false);
        const duration = ref(0);
        const progress = ref(0);
        
        const togglePlay = () => { if (!audioEl.value) return; isPlaying.value ? audioEl.value.pause() : audioEl.value.play(); };
        const onPlay = () => { isPlaying.value = true; document.querySelectorAll('audio').forEach(el => { if (el !== audioEl.value) el.pause(); }); };
        const onTimeUpdate = () => { if (!audioEl.value || !duration.value) return; progress.value = (audioEl.value.currentTime / duration.value) * 100; };
        const onLoaded = () => { if (!audioEl.value) return; duration.value = audioEl.value.duration; };
        const onEnded = () => { isPlaying.value = false; progress.value = 0; if (audioEl.value) audioEl.value.currentTime = 0; };
        const seek = (event) => { if (!audioEl.value || !duration.value) return; const rect = event.currentTarget.getBoundingClientRect(); const newProgress = (event.clientX - rect.left) / rect.width; audioEl.value.currentTime = newProgress * duration.value; progress.value = newProgress * 100; };
        
        return { audioEl, isPlaying, progress, duration, togglePlay, onTimeUpdate, onLoaded, onEnded, onPlay, seek };
    }
}