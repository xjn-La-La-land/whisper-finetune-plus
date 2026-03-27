# shared_state.py

# GPU 状态枚举
class GPUStatus:
    IDLE = "IDLE"               # 空闲
    TRAINING = "TRAINING"       # 正在微调
    INFERENCING = "INFERENCING" # 正在运行推理服务

# 全局 GPU 状态锁
GPU_STATE = {
    "status": GPUStatus.IDLE,
    "current_user": None
}

# 全局推理管道级缓存 (用于判断当前显存中加载的是哪个模型)
# loaded_model_path 用于记录当前显存里的模型路径
INFERENCE_CACHE = {
    "loaded_model_path": None,
    "pipeline": None
}