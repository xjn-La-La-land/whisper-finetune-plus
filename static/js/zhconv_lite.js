// zhconv_lite.js
// 繁体 → 简体（zh-cn）转换，纯客户端。复刻 Python `zhconv.convert(s, 'zh-cn')` 的
// 「前缀集 + 最长匹配」算法，并使用从同一个 zhconv 导出的映射表（static/vendor/zh2cn.json），
// 保证浏览器端 WASM 推理的简繁结果与服务端 /api/recognition 完全一致。
//
// 用途：本地 WASM 路径里，base 模型（tiny/base）输出的是繁体，需在 JS 里转简体
// （微调模型本就输出简体，转一遍也无害）。表 lazy-load：只有真正用到本地推理时才拉。

// --- 纯算法（导出以便单测）---------------------------------------------------

// 由映射表的所有键构建「前缀集」：键的每一个前缀都加入。对应 zhconv 里的 pfset。
export function buildPrefixSet(map) {
    const pfset = new Set();
    for (const key of map.keys()) {
        const kc = Array.from(key);              // 按码点切，避免代理对错位
        for (let i = 1; i <= kc.length; i++) pfset.add(kc.slice(0, i).join(''));
    }
    return pfset;
}

// 复刻 zhconv.convert 的主循环：每个位置沿着前缀集尽量延长，记录最长命中。
export function convertWith(map, pfset, s) {
    const cps = Array.from(s);                   // 码点数组，和 Python str 索引语义一致
    const N = cps.length;
    const out = [];
    let pos = 0;
    while (pos < N) {
        let i = pos;
        let frag = cps[pos];
        let maxword = null;
        let maxpos = 0;
        while (i < N && pfset.has(frag)) {
            if (map.has(frag)) { maxword = map.get(frag); maxpos = i; }
            i += 1;
            frag = cps.slice(pos, i + 1).join('');
        }
        if (maxword === null) { out.push(cps[pos]); pos += 1; }
        else { out.push(maxword); pos = maxpos + 1; }
    }
    return out.join('');
}

// --- 浏览器侧：lazy-load 表 + 缓存 -------------------------------------------

let _map = null;
let _pfset = null;
let _loadPromise = null;

export async function loadConverter(url = '/static/vendor/zh2cn.json') {
    if (_map) return;
    if (!_loadPromise) {
        _loadPromise = fetch(url)
            .then((r) => {
                if (!r.ok) throw new Error(`加载简繁表失败: HTTP ${r.status}`);
                return r.json();
            })
            .then((obj) => {
                _map = new Map(Object.entries(obj));
                _pfset = buildPrefixSet(_map);
            });
    }
    return _loadPromise;
}

export function isConverterLoaded() {
    return !!_map;
}

// 繁 → 简。表未加载时原样返回（调用方应先 await loadConverter）。
export function toSimplified(s) {
    if (!_map || !s) return s;
    return convertWith(_map, _pfset, s);
}
