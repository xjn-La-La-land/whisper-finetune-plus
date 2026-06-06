// wasm_whisper.js
// 浏览器端 whisper.cpp（libmain.wasm）推理封装。负责：
//   1. 浏览器能力探测（crossOriginIsolated / WASM / SIMD / IndexedDB）
//   2. 加载 libmain.js 运行时（配 print 捕获 + locateFile）
//   3. 模型下载（带进度）+ IndexedDB 缓存 + 写入 WASM FS + Module.init
//   4. transcribe(pcm) —— full_default 跑后台线程，结果靠解析 print 流（带时间戳的段 +
//      whisper_print_timings 判完成）
//   5. 音频 blob/文件 → 16k 单声道 Float32
//
// 设计依据见 vendor/whisper.cpp/examples/whisper.wasm/{emscripten.cpp,index-tmpl.html}
// 与 TODO_WHISPER_CPP_WASM.md P2-2/P2-3。

const WASM_DIR = '/static/wasm/';
const FS_MODEL_NAME = 'whisper.bin';     // 模型在 WASM FS 里的固定文件名
const IDB_NAME = 'whisper-ggml-cache';
const IDB_STORE = 'models';
const TRANSCRIBE_TIMEOUT_MS = 5 * 60 * 1000;

// 一行段落：[00:00:00.000 --> 00:00:02.000]   文本
const SEG_RE = /^\s*\[(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.*)$/;
// whisper_print_timings 的 total time 行 = 一次推理结束
const DONE_RE = /whisper_print_timings:\s+total time/;

// "HH:MM:SS.mmm" → 浮点秒（与服务端 /api/recognition 的 start/end 数字格式一致）
function _ts2sec(ts) {
    const m = ts.match(/(\d{2}):(\d{2}):(\d{2})\.(\d{3})/);
    if (!m) return 0;
    return (+m[1]) * 3600 + (+m[2]) * 60 + (+m[3]) + (+m[4]) / 1000;
}

// ---------------------------------------------------------------------------
// 能力探测
// ---------------------------------------------------------------------------
let _simdCache = null;
async function _wasmSimdOk() {
    if (_simdCache !== null) return _simdCache;
    try {
        // 最小的含 v128 指令的 wasm 模块字节；validate 通过 = 支持 SIMD
        const bytes = new Uint8Array([
            0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0,
            10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
        ]);
        _simdCache = WebAssembly.validate(bytes);
    } catch (e) {
        _simdCache = false;
    }
    return _simdCache;
}

// 返回 { ok, reason, ... }。ok=false 时 reason 是给用户看的中文说明。
export async function detectCapabilities() {
    const hasWasm = typeof WebAssembly !== 'undefined';
    const hasIDB = 'indexedDB' in window;
    const isolated = (typeof crossOriginIsolated !== 'undefined') ? crossOriginIsolated : false;
    const hasSAB = typeof SharedArrayBuffer !== 'undefined';
    const simd = hasWasm ? await _wasmSimdOk() : false;

    let ok = true;
    let reason = '';
    if (!hasWasm) { ok = false; reason = '浏览器不支持 WebAssembly'; }
    else if (!simd) { ok = false; reason = '浏览器不支持 WASM SIMD'; }
    else if (!hasIDB) { ok = false; reason = '浏览器不支持 IndexedDB（无法缓存模型）'; }
    else if (!isolated || !hasSAB) {
        ok = false;
        reason = '页面未跨源隔离（缺 SharedArrayBuffer）——多线程 WASM 不可用';
    }
    return { ok, reason, hasWasm, simd, hasIDB, crossOriginIsolated: isolated, hasSharedArrayBuffer: hasSAB };
}

// ---------------------------------------------------------------------------
// IndexedDB 模型缓存
// ---------------------------------------------------------------------------
function _idbOpen() {
    return new Promise((resolve, reject) => {
        const rq = indexedDB.open(IDB_NAME, 1);
        rq.onupgradeneeded = () => {
            const db = rq.result;
            if (!db.objectStoreNames.contains(IDB_STORE)) db.createObjectStore(IDB_STORE);
        };
        rq.onsuccess = () => resolve(rq.result);
        rq.onerror = () => reject(rq.error);
    });
}
async function _idbGet(key) {
    const db = await _idbOpen();
    try {
        return await new Promise((resolve, reject) => {
            const rq = db.transaction(IDB_STORE, 'readonly').objectStore(IDB_STORE).get(key);
            rq.onsuccess = () => resolve(rq.result || null);
            rq.onerror = () => reject(rq.error);
        });
    } finally { db.close(); }
}
async function _idbPut(key, value) {
    const db = await _idbOpen();
    try {
        await new Promise((resolve, reject) => {
            const tx = db.transaction(IDB_STORE, 'readwrite');
            tx.objectStore(IDB_STORE).put(value, key);
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error);
        });
    } finally { db.close(); }
}

// 下载（带进度）+ 缓存。返回 Uint8Array。onProgress(receivedBytes, totalBytesOrNull)。
// fetcher: 可注入带鉴权的 fetch（下载接口需 JWT，调用方传 api.js 的 apiFetch）。
async function _fetchModelBytes(url, onProgress, fetcher) {
    const doFetch = fetcher || ((u) => fetch(u));
    const res = await doFetch(url);
    if (!res.ok) throw new Error(`下载模型失败: HTTP ${res.status}`);
    const total = parseInt(res.headers.get('content-length') || '0', 10) || null;
    const reader = res.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        if (onProgress) onProgress(received, total);
    }
    const out = new Uint8Array(received);
    let pos = 0;
    for (const c of chunks) { out.set(c, pos); pos += c.length; }
    return out;
}

// ---------------------------------------------------------------------------
// libmain 运行时加载
// ---------------------------------------------------------------------------
let _runtimePromise = null;
let _lineListener = null;          // 当前 transcribe 的 print 行回调

function _onLine(text) {
    if (_lineListener) {
        try { _lineListener(text); } catch (e) { /* 忽略单行处理异常 */ }
    }
}

export function ensureRuntime() {
    if (_runtimePromise) return _runtimePromise;
    _runtimePromise = new Promise((resolve, reject) => {
        // libmain.js 是经典全局 Module 模式：先把配置挂到 window.Module，再注入脚本，
        // 脚本会就地往这个对象上补 init/free/full_default/FS_* 等。
        window.Module = {
            print: _onLine,
            printErr: _onLine,         // whisper 的 timings/段落可能走 stderr，一并收
            locateFile: (path) => WASM_DIR + path,   // libmain.wasm 等都在 /static/wasm/
            onRuntimeInitialized: () => resolve(window.Module),
            onAbort: (what) => reject(new Error('WASM 运行时中止: ' + what)),
        };
        const s = document.createElement('script');
        s.src = WASM_DIR + 'libmain.js';
        s.async = true;
        s.onerror = () => reject(new Error('加载 libmain.js 失败'));
        document.head.appendChild(s);
    });
    return _runtimePromise;
}

// ---------------------------------------------------------------------------
// 模型加载（缓存 → WASM FS → init）
// ---------------------------------------------------------------------------
let _instance = null;          // 当前 whisper context index（>=1），null 表示未加载
let _loadedKey = null;         // 当前已加载模型的 cacheKey
let _busy = false;             // transcribe 互斥

function _storeFS(bytes) {
    const M = window.Module;
    try { M.FS_unlink(FS_MODEL_NAME); } catch (e) { /* 不存在则忽略 */ }
    M.FS_createDataFile('/', FS_MODEL_NAME, bytes, true, true);
}

export function getLoadedKey() { return _loadedKey; }

// 确保指定模型已加载到 WASM。
//   cacheKey: 稳定缓存键（同一份内容用同一个键，换发布版本就换键）
//   url:      下载地址（/api/download_published_ggml?v=... 或 /api/download_base_ggml?model=...）
//   onProgress(received, total|null): 仅在真正下载（缓存未命中）时回调
// 返回 { fromCache: boolean }
export async function ensureModel({ cacheKey, url, onProgress, fetcher }) {
    await ensureRuntime();
    if (_loadedKey === cacheKey && _instance != null) return { fromCache: true };
    if (_busy) throw new Error('正在识别中，无法切换模型');

    // 取字节：先查 IndexedDB，没有再下载并写入缓存
    let bytes = null;
    let fromCache = false;
    try { bytes = await _idbGet(cacheKey); } catch (e) { bytes = null; }
    if (bytes) {
        fromCache = true;
        bytes = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    } else {
        bytes = await _fetchModelBytes(url, onProgress, fetcher);
        try { await _idbPut(cacheKey, bytes); } catch (e) { /* 配额不足等，仍可用，只是没缓存 */ }
    }

    // 释放旧 context（若有），写入新模型并 init
    if (_instance != null) {
        try { window.Module.free(_instance); } catch (e) { /* 忽略 */ }
        _instance = null;
        _loadedKey = null;
    }
    _storeFS(bytes);
    const idx = window.Module.init(FS_MODEL_NAME);
    if (!idx) throw new Error('WASM 模型初始化失败（init 返回 0）');
    _instance = idx;
    _loadedKey = cacheKey;
    return { fromCache };
}

// 删某个缓存键（重新下载时用）
export async function evictModel(cacheKey) {
    try {
        const db = await _idbOpen();
        await new Promise((resolve, reject) => {
            const tx = db.transaction(IDB_STORE, 'readwrite');
            tx.objectStore(IDB_STORE).delete(cacheKey);
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error);
        });
        db.close();
    } catch (e) { /* 忽略 */ }
    if (_loadedKey === cacheKey) {
        if (_instance != null) { try { window.Module.free(_instance); } catch (e) {} }
        _instance = null; _loadedKey = null;
    }
}

// 查某缓存键是否已在 IndexedDB（用于「已缓存」徽章）
export async function isCached(cacheKey) {
    try { return !!(await _idbGet(cacheKey)); } catch (e) { return false; }
}

// ---------------------------------------------------------------------------
// 推理
// ---------------------------------------------------------------------------
function _nthreads() {
    const hc = navigator.hardwareConcurrency || 4;
    return Math.max(1, Math.min(8, hc));   // binding 内部还会再 clamp
}

// transcribe(pcmF32, lang) → Promise<{ segments:[{start,end,text}], text }>
// onSegment(seg) 可选：每解析到一段就回调（用于边出边显示）。
export function transcribe(pcm, lang = 'zh', { onSegment } = {}) {
    if (_instance == null) return Promise.reject(new Error('模型未加载'));
    if (_busy) return Promise.reject(new Error('上一次识别还在进行'));
    _busy = true;

    const segments = [];
    return new Promise((resolve, reject) => {
        let timer = setTimeout(() => { cleanup(); reject(new Error('识别超时')); }, TRANSCRIBE_TIMEOUT_MS);
        function cleanup() { clearTimeout(timer); _lineListener = null; _busy = false; }

        _lineListener = (line) => {
            const m = line.match(SEG_RE);
            if (m) {
                const seg = { start: _ts2sec(m[1]), end: _ts2sec(m[2]), text: (m[3] || '').trim() };
                if (seg.text) { segments.push(seg); if (onSegment) onSegment(seg); }
                return;
            }
            if (DONE_RE.test(line)) {
                cleanup();
                resolve({ segments, text: segments.map((s) => s.text).join('') });
            }
        };

        // full_default 立即返回（推理在后台线程跑），结果全靠上面的 print 监听
        let ret;
        try {
            ret = window.Module.full_default(_instance, pcm, lang, _nthreads(), false);
        } catch (e) {
            cleanup();
            reject(e);
            return;
        }
        if (ret !== 0) { cleanup(); reject(new Error('full_default 返回 ' + ret)); }
    });
}

// ---------------------------------------------------------------------------
// 音频解码：blob/文件 → 16kHz 单声道 Float32（whisper 输入格式）
// ---------------------------------------------------------------------------
export async function decodeToPcm16k(blob) {
    const arrayBuf = await blob.arrayBuffer();
    const AC = window.AudioContext || window.webkitAudioContext;
    const decodeCtx = new AC();
    let audioBuffer;
    try {
        audioBuffer = await decodeCtx.decodeAudioData(arrayBuf);
    } finally {
        if (decodeCtx.close) decodeCtx.close();
    }
    const targetRate = 16000;
    const frames = Math.ceil(audioBuffer.duration * targetRate);
    const OAC = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    // 目标 1 声道 → 多声道源连上去会自动下混为单声道
    const offline = new OAC(1, Math.max(frames, 1), targetRate);
    const src = offline.createBufferSource();
    src.buffer = audioBuffer;
    src.connect(offline.destination);
    src.start(0);
    const rendered = await offline.startRendering();
    return rendered.getChannelData(0);    // Float32Array
}
