# Ultralytics YOLO 训练配置文件
# 使用方法: python train_config.py


from ultralytics import YOLO

# ============================================================================
# 基础配置
# ============================================================================

# 模型配置
MODEL_CONFIG = {
    "model": "yolo11n.pt",  # 模型文件: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
    # 或者使用 YAML 配置文件从头训练: "yolo11n.yaml"
    # 或者使用自定义权重: "path/to/your/weights.pt"
}

# 数据集配置
DATA_CONFIG = {
    "data": "VisDrone.yaml",  # 数据集配置文件路径
    # 其他选项: "coco.yaml", "coco8.yaml", 或自定义数据集 YAML 文件
}

# ============================================================================
# 训练参数
# ============================================================================

TRAIN_ARGS = {
    # 基本训练设置
    "epochs": 100,  # 训练轮数
    "batch": 16,  # 批次大小，使用 -1 自动选择最佳批次大小
    "imgsz": 640,  # 输入图像尺寸 (正方形) 或 [height, width]
    "patience": 50,  # 早停耐心值，N 轮验证无改善则停止训练
    # 设备配置
    "device": 0,  # 设备: 0, 1, 2, 3 为 GPU 编号; 'cpu' 为 CPU; 'mps' 为 Apple Silicon
    # 多 GPU 训练: [0, 1, 2, 3] 或 -1 自动选择空闲 GPU
    # 数据加载
    "workers": 8,  # 数据加载器工作线程数
    "cache": False,  # 缓存图像: False, True (RAM), 'disk' (磁盘缓存)
    # 保存设置
    "save": True,  # 保存训练检查点和预测结果
    "save_period": -1,  # 每 N 轮保存一次检查点 (-1 禁用，仅在最后保存)
    "project": "runs/train",  # 项目名称，结果保存在 project/name 目录
    "name": "visdrone_exp",  # 实验名称
    "exist_ok": False,  # 如果 project/name 存在，是否覆盖
    # 验证设置
    "val": True,  # 训练期间运行验证
    "split": "val",  # 验证集分割: 'val', 'test', 'train'
    # 其他设置
    "verbose": True,  # 打印详细日志
    "seed": 0,  # 随机种子，用于可重复性
    "deterministic": True,  # 确定性操作，可重复但可能较慢
    "amp": True,  # 自动混合精度训练 (AMP)
    "resume": False,  # 从上次检查点恢复训练
    "fraction": 1.0,  # 使用训练数据集的比例 (1.0 = 全部)
}

# ============================================================================
# 优化器配置
# ============================================================================

OPTIMIZER_CONFIG = {
    "optimizer": "auto",  # 优化器: 'SGD', 'Adam', 'AdamW', 'Adamax', 'NAdam', 'RAdam', 'RMSProp', 'auto'
    "lr0": 0.01,  # 初始学习率 (SGD=1e-2, Adam/AdamW=1e-3)
    "lrf": 0.01,  # 最终学习率比例 (最终 LR = lr0 * lrf)
    "momentum": 0.937,  # SGD 动量或 Adam beta1
    "weight_decay": 0.0005,  # 权重衰减 (L2 正则化)
    "warmup_epochs": 3.0,  # 预热轮数 (允许小数)
    "warmup_momentum": 0.8,  # 预热期间的初始动量
    "warmup_bias_lr": 0.1,  # 预热期间的偏置学习率
    "cos_lr": False,  # 使用余弦学习率调度器
}

# ============================================================================
# 损失函数权重
# ============================================================================

LOSS_WEIGHTS = {
    "box": 7.5,  # 边界框损失权重
    "cls": 0.5,  # 分类损失权重
    "dfl": 1.5,  # 分布焦点损失权重
    "pose": 12.0,  # 姿态损失权重 (仅姿态任务)
    "kobj": 1.0,  # 关键点目标损失权重 (仅姿态任务)
    "nbs": 64,  # 用于损失归一化的标称批次大小
}

# ============================================================================
# 数据增强配置
# ============================================================================

AUGMENTATION_CONFIG = {
    # HSV 颜色空间增强
    "hsv_h": 0.015,  # HSV 色调增强比例
    "hsv_s": 0.7,  # HSV 饱和度增强比例
    "hsv_v": 0.4,  # HSV 亮度增强比例
    # 几何变换
    "degrees": 0.0,  # 旋转角度 (+/-)
    "translate": 0.1,  # 平移比例 (+/-)
    "scale": 0.5,  # 缩放增益 (+/-)
    "shear": 0.0,  # 剪切角度 (+/-)
    "perspective": 0.0,  # 透视变换比例 (0-0.001 典型值)
    # 翻转
    "flipud": 0.0,  # 垂直翻转概率
    "fliplr": 0.5,  # 水平翻转概率
    "bgr": 0.0,  # RGB↔BGR 通道交换概率
    # 高级增强
    "mosaic": 1.0,  # Mosaic 增强概率
    "mixup": 0.0,  # MixUp 增强概率
    "cutmix": 0.0,  # CutMix 增强概率
    "copy_paste": 0.0,  # 分割复制粘贴概率 (仅分割任务)
    "close_mosaic": 10,  # 最后 N 轮禁用 Mosaic 增强 (0 保持启用)
    # 其他
    "auto_augment": "randaugment",  # 分类自动增强策略: 'randaugment', 'autoaugment', 'augmix'
    "erasing": 0.4,  # 分类随机擦除概率 (0-0.9)
    "multi_scale": False,  # 多尺度训练，通过改变图像尺寸
}

# ============================================================================
# 验证/测试配置
# ============================================================================

VAL_CONFIG = {
    "conf": 0.001,  # 置信度阈值 (验证默认 0.001, 预测默认 0.25)
    "iou": 0.7,  # NMS IoU 阈值
    "max_det": 300,  # 每张图像的最大检测数
    "half": False,  # 使用半精度 (FP16) 如果支持
    "plots": True,  # 训练/验证期间保存图表和图像
    "save_json": False,  # 保存结果为 COCO JSON 格式用于外部评估
}

# ============================================================================
# 训练函数
# ============================================================================


def train_model():
    """执行模型训练."""
    # 加载模型
    model = YOLO(MODEL_CONFIG["model"])

    # 合并所有配置
    train_args = {
        **DATA_CONFIG,
        **TRAIN_ARGS,
        **OPTIMIZER_CONFIG,
        **LOSS_WEIGHTS,
        **AUGMENTATION_CONFIG,
        **VAL_CONFIG,
    }

    # 开始训练
    print("=" * 60)
    print("开始训练 YOLO 模型")
    print("=" * 60)
    print(f"模型: {MODEL_CONFIG['model']}")
    print(f"数据集: {DATA_CONFIG['data']}")
    print(f"训练轮数: {TRAIN_ARGS['epochs']}")
    print(f"批次大小: {TRAIN_ARGS['batch']}")
    print(f"图像尺寸: {TRAIN_ARGS['imgsz']}")
    print(f"设备: {TRAIN_ARGS['device']}")
    print("=" * 60)

    # 执行训练
    results = model.train(**train_args)

    # 训练完成后的操作
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"最佳权重保存在: {results.save_dir}/weights/best.pt")
    print(f"最后权重保存在: {results.save_dir}/weights/last.pt")
    print("=" * 60)

    return results


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    # 执行训练
    results = train_model()

    # 可选: 训练后立即进行验证
    # model = YOLO(f"{results.save_dir}/weights/best.pt")
    # metrics = model.val(data=DATA_CONFIG["data"])
    # print(f"mAP50: {metrics.box.map50}")
    # print(f"mAP50-95: {metrics.box.map}")
