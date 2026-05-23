# shared_state.py
import asyncio


# GPU 状态枚举
class GPUStatus:
    IDLE = "IDLE"               # 空闲
    TRAINING = "TRAINING"       # 正在微调
    INFERENCING = "INFERENCING" # 正在运行推理服务


# 全局 GPU 状态字典：记录当前 GPU 被谁、以何种状态占用，供前端轮询展示。
#
# current_model_name 仅在 TRAINING 期间有值，用于前端刷新后重新连接到正确的 SSE 流。
# （日志路径按 {username}/{model_name} 隔离，必须告诉前端是哪个模型才能找到日志。）
GPU_STATE = {
    "status": GPUStatus.IDLE,
    "current_user": None,
    "current_model_name": None
}

# GPU 状态转移的互斥锁。
#
# 用法约定：
#   * 任何修改 GPU_STATE 的代码必须放在 `async with GPU_LOCK:` 块内
#   * 锁只在"检查 + 转移"的瞬间持有，不要在锁内做模型加载 / 推理等长耗时操作，
#     否则会阻塞 GET /api/gpu_status 等只读请求
#   * 长时间的 GPU 占用通过 GPU_STATE["status"] != IDLE 表达；
#     后来的请求在锁内看到非 IDLE 状态就立刻拒绝 (423)
#
# 修复了原先 dict-only 实现的两个 race：
#   1) start_finetune / api_recognition 的 "检查 + 上锁" 之间有 await，并发可双双通过检查
#   2) 多个 inference 请求并发时，后到者覆盖 current_user；前一个 finally 又无差别把状态
#      置 IDLE，导致第三方训练请求趁虚而入。原子化转移后此场景下后到者直接 423，不会再覆盖。
GPU_LOCK = asyncio.Lock()


# 全局推理管道级缓存 (用于判断当前显存中加载的是哪个模型)
# loaded_model_path 用于记录当前显存里的模型路径
INFERENCE_CACHE = {
    "loaded_model_path": None,
    "pipeline": None
}