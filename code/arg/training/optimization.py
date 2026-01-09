"""
优化器和调度器创建
对应论文训练设置
"""

import torch
from torch.optim import Optimizer, AdamW, Adam, SGD
from torch.optim.lr_scheduler import LRScheduler, LinearLR, CosineAnnealingLR
from typing import List, Optional


def create_optimizer(
    parameters: List,
    config: dict,
    optimizer_type: str = "adamw"
) -> Optimizer:
    """
    创建优化器
    
    对应论文实验设置中的优化器配置
    
    Args:
        parameters: 模型参数
        config: 训练配置字典
        optimizer_type: 优化器类型
    
    Returns:
        优化器
    """
    lr = config.get('learning_rate', 1e-5)
    weight_decay = config.get('weight_decay', 0.01)
    
    if optimizer_type.lower() == "adamw":
        return AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_type.lower() == "adam":
        return Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_type.lower() == "sgd":
        return SGD(
            parameters,
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: Optimizer,
    num_training_steps: int,
    warmup_steps: int = 0,
    scheduler_type: str = "linear"
) -> Optional[LRScheduler]:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        num_training_steps: 总训练步数
        warmup_steps: 预热步数
        scheduler_type: 调度器类型
    
    Returns:
        学习率调度器
    """
    if scheduler_type == "linear":
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=num_training_steps
        )
    elif scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=0
        )
    elif scheduler_type == "constant":
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def create_multi_optimizer(
    model_params: List,
    critic_params: List,
    safety_dir_params: List,
    config: dict
) -> dict:
    """
    创建多个优化器（用于多目标优化）
    
    对应论文：主模型、Critics、Safety Direction 需要分别优化
    
    Args:
        model_params: 主模型参数
        critic_params: Critic 参数
        safety_dir_params: Safety Direction 参数
        config: 配置
    
    Returns:
        包含所有优化器的字典
    """
    model_lr = config.get('learning_rate', 1e-5)
    critic_lr = config.get('critic_learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 0.01)
    
    return {
        'model': AdamW(
            model_params,
            lr=model_lr,
            weight_decay=weight_decay
        ),
        'critics': AdamW(
            critic_params,
            lr=critic_lr,
            weight_decay=weight_decay
        ),
        'safety_direction': AdamW(
            safety_dir_params,
            lr=model_lr,
            weight_decay=0.0  # 不对 direction 做权重衰减
        ),
    }

