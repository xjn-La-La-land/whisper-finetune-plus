// static/js/errors.js
// 统一的错误文案助手：把任何非 2xx Response / 抛出的异常映射成「可操作的中文」。
//
// 设计要点：
//   * 按 HTTP 状态码映射，不去解析 detail 的语义；
//   * 后端已写好中文的 detail（423/409/部分 400）原样回显，其余用固定中文兜底；
//   * 401 返回空串 —— 交给 api.js 的全局 whisper:unauthorized 处理，调用方据此跳过弹窗；
//   * 网络错误绝不回显 err.message（那是 "Failed to fetch" 泄漏处）。

// 含至少一个 CJK 字符即认为可直接展示给中文用户。
function isChinese(s) {
    return typeof s === 'string' && /[一-鿿]/.test(s);
}

// 从已读出的 body 里取后端文案（detail 优先，其次 message —— 登录/注册接口用 message）。
function pickDetail(body) {
    if (!body || typeof body !== 'object') return '';
    const d = body.detail != null ? body.detail : body.message;
    return typeof d === 'string' ? d : '';
}

/**
 * 把一个非 2xx 的 Response 映射成可操作的中文提示。
 * @param {Response} res
 * @param {string} [fallback]
 * @returns {Promise<string>} 提示文案；401 返回 ''（交全局处理，调用方应据此跳过弹窗）
 */
export async function friendlyHttpError(res, fallback = '操作失败，请稍后重试。') {
    const status = res ? res.status : 0;

    // 401：会话过期，已由 api.js 广播 whisper:unauthorized 全局接管，这里不重复提示。
    if (status === 401) return '';
    // 413：请求体过大。应用层无体积限制，413 只来自前置隧道/代理（body 多为 HTML），按状态码兜底。
    if (status === 413) return '文件太大，无法上传。请录制更短的音频后重试。';

    // 读 body（413/SSE/空体可能不是 JSON；clone 以免与调用方的 res.json() 抢 body）
    let body = null;
    try { body = await res.clone().json(); } catch (e) { body = null; }
    const detail = pickDetail(body);

    // 423 GPU 占用 / 409 冲突：后端写的就是可操作的中文，原样回显。
    if (status === 423 || status === 409) return isChinese(detail) ? detail : fallback;
    // 400：多为校验类，后端中文 detail 直接给（如模型名规则），否则兜底。
    if (status === 400) return isChinese(detail) ? detail : '请求无法处理，请检查输入后重试。';
    // 422：FastAPI 参数校验，detail 常是结构化/英文，避免泄漏，给固定中文。
    if (status === 422) return '提交的内容不符合要求，请检查后重试。';
    // 5xx：服务端错误。
    if (status >= 500) return '服务器开小差了，请稍后重试；若多次失败请联系管理员。';

    return isChinese(detail) ? detail : fallback;
}

/**
 * 把 fetch reject / 其它异常映射成固定中文网络提示。绝不回显 err.message。
 * @returns {string}
 */
export function friendlyNetworkError() {
    return '无法连接服务器，请检查网络后重试。';
}
