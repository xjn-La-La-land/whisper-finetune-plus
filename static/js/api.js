// static/js/api.js
//
// 统一的 fetch 包装。
//
// 作用：
//   * 自动给请求加 `Authorization: Bearer <token>` 头
//   * 401 时清空本地 token + 抛 401 错误（调用方可监听并跳回登录页）
//
// 用法：
//   import { apiFetch, getToken, setToken, clearToken, ssEUrl } from './api.js';
//   const res = await apiFetch('/api/tasks');
//   const data = await res.json();

const TOKEN_KEY = 'whisper_token';
const USERNAME_KEY = 'whisper_username';

export function getToken() {
    return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token, username) {
    localStorage.setItem(TOKEN_KEY, token);
    if (username) localStorage.setItem(USERNAME_KEY, username);
}

export function getStoredUsername() {
    return localStorage.getItem(USERNAME_KEY);
}

export function clearToken() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USERNAME_KEY);
}

/**
 * 监听 401。如果服务器告诉我们 token 失效，我们清空本地 token 并广播事件。
 * app.js 可以监听这个事件把用户踢回登录页。
 */
function broadcastUnauthorized() {
    clearToken();
    window.dispatchEvent(new CustomEvent('whisper:unauthorized'));
}

/**
 * apiFetch(url, options?) — 与原生 fetch 同 API，多做两件事：
 *   1. 自动从 localStorage 取 token 加到 Authorization 头
 *   2. 收到 401 时清 token 并广播事件
 *
 * 调用方仍能拿到原始 Response 对象做后续处理。
 */
export async function apiFetch(url, options = {}) {
    const token = getToken();
    const headers = new Headers(options.headers || {});
    if (token) {
        headers.set('Authorization', `Bearer ${token}`);
    }
    const res = await fetch(url, { ...options, headers });
    if (res.status === 401) {
        // 不要在这里 throw，让调用方自己决定是否提示；
        // 但要先广播 unauthorized 让 UI 回到登录态
        broadcastUnauthorized();
    }
    return res;
}

/**
 * 给 EventSource 用的 URL 构造器。
 * EventSource 不能附加 header，所以 token 走 query 参数（?token=...）。
 * 同时把其他业务参数拼上去。
 *
 * 用法：new EventSource(sseUrl('/api/train_stream', { model_name: 'foo' }))
 */
export function sseUrl(path, params = {}) {
    const u = new URL(path, window.location.origin);
    const token = getToken();
    if (token) u.searchParams.set('token', token);
    for (const [k, v] of Object.entries(params)) {
        if (v !== undefined && v !== null) u.searchParams.set(k, String(v));
    }
    // 转成相对路径返回（保留 query），避免暴露 origin
    return u.pathname + u.search;
}
