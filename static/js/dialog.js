// static/js/dialog.js
//
// 全局自定义对话框模块。替代浏览器原生 alert() / confirm()。
//
// 用法：
//   import * as dialog from './dialog.js';
//   await dialog.alert('保存成功');
//   const ok = await dialog.confirm('确定删除吗？');
//
// 相对浏览器原生的优势：
//   * 视觉与 Tailwind UI 一致，字号 / 按钮 / 间距对脑瘫儿童 + 家长更友好
//   * 支持类型变体（info / success / warning / danger）的图标和配色
//   * Promise-based，可 await，不会真正阻塞 event loop
//   * 全局键盘支持（Enter 确认 / Esc 取消）由 app.js 接管
//   * 点背景不自动关闭（避免脑瘫用户的肌肉震颤误触）

const { reactive } = Vue;

// 单例的对话框状态。挂载到 app.js 的根模板上由 <UiDialog> 渲染。
export const dialogState = reactive({
    visible: false,
    type: 'alert',            // 'alert' | 'confirm'
    variant: 'info',          // 'info' | 'success' | 'warning' | 'danger'
    title: '',
    message: '',
    confirmText: '我知道了',
    cancelText: '取消',
    _resolve: null
});

function show(opts) {
    return new Promise(resolve => {
        // 如果上一个对话框还没关，先用 false 把它 resolve 掉，避免 Promise 堆积
        if (dialogState._resolve) dialogState._resolve(false);
        Object.assign(dialogState, {
            visible: true,
            type: opts.type || 'alert',
            variant: opts.variant || 'info',
            title: opts.title || '',
            message: opts.message || '',
            confirmText: opts.confirmText || '我知道了',
            cancelText: opts.cancelText || '取消',
            _resolve: resolve
        });
    });
}

/**
 * 信息提示对话框（单按钮）。
 * @param {string} message - 主体文本
 * @param {object} [opts] - { title, confirmText, variant: 'info'|'success'|'warning'|'danger' }
 * @returns {Promise<void>} 用户点确认后 resolve
 */
export function alert(message, opts = {}) {
    return show({
        type: 'alert',
        variant: opts.variant || 'info',
        title: opts.title || '提示',
        message,
        confirmText: opts.confirmText || '我知道了'
    });
}

/**
 * 确认对话框（两按钮）。
 * @param {string} message - 主体文本
 * @param {object} [opts] - { title, confirmText, cancelText, variant }
 * @returns {Promise<boolean>} 用户点确认 → true，点取消或 Esc → false
 */
export function confirm(message, opts = {}) {
    return show({
        type: 'confirm',
        variant: opts.variant || 'warning',
        title: opts.title || '请确认',
        message,
        confirmText: opts.confirmText || '确定',
        cancelText: opts.cancelText || '取消'
    });
}

export function handleConfirm() {
    const fn = dialogState._resolve;
    dialogState.visible = false;
    dialogState._resolve = null;
    if (fn) fn(true);
}

export function handleCancel() {
    const fn = dialogState._resolve;
    dialogState.visible = false;
    dialogState._resolve = null;
    if (fn) fn(false);
}
