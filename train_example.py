# 快速训练示例
# 这是一个简化的训练示例，展示如何使用 train_config.py

from ultralytics import YOLO


# 方法 1: 最简单的训练方式
def simple_train():
    """最简单的训练方式."""
    model = YOLO("yolo11n.pt")  # 加载预训练模型
    model.train(
        data="VisDrone.yaml",  # 数据集配置文件
        epochs=100,  # 训练轮数
        imgsz=640,  # 图像尺寸
        batch=16,  # 批次大小
        device=0,  # GPU 设备
    )


# 方法 2: 使用配置文件中的参数
def config_train():
    """使用配置文件进行训练."""
    from train_config import (
        AUGMENTATION_CONFIG,
        DATA_CONFIG,
        LOSS_WEIGHTS,
        MODEL_CONFIG,
        OPTIMIZER_CONFIG,
        TRAIN_ARGS,
    )

    # 加载模型
    model = YOLO(MODEL_CONFIG["model"])

    # 合并所有配置
    train_args = {
        **DATA_CONFIG,
        **TRAIN_ARGS,
        **OPTIMIZER_CONFIG,
        **LOSS_WEIGHTS,
        **AUGMENTATION_CONFIG,
    }

    # 开始训练
    results = model.train(**train_args)
    return results


# 方法 3: 直接调用配置文件中的训练函数
def config_file_train():
    """直接使用配置文件中的训练函数."""
    from train_config import train_model

    results = train_model()
    return results


if __name__ == "__main__":
    # 选择一种训练方式
    # simple_train()           # 最简单的方式
    # config_train()           # 使用配置参数
    config_file_train()  # 使用配置文件函数 (推荐)
