"""
模型配置
对应论文实验中的模型设置
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """
    模型配置
    
    对应论文实验中使用的模型（Qwen2.5-7B）
    """
    # Backbone 模型
    model_name: str = "Qwen/Qwen2.5-7B"
    """模型名称（论文使用 Qwen2.5-7B）"""
    
    hidden_size: int = 4096
    """隐藏层维度（Qwen2.5-7B 的维度）"""
    
    device: str = "cuda"
    """设备"""
    
    # 表示提取
    normalize_representations: bool = True
    """是否对表示进行 L2 归一化（论文 Appendix C.2）"""
    
    representation_layer: int = -1
    """提取表示的层（-1 表示最后一层）"""
    
    # Safety Direction
    safety_direction_init: str = "random"
    """安全方向初始化方法: random, zero, pca"""
    
    # Critics
    critic_hidden_dims: list[int] = None
    """Critic 网络隐藏层维度"""
    
    critic_dropout: float = 0.1
    """Critic Dropout 概率"""
    
    y_attr_dim: int = 2
    """安全属性类别数（Safe/Unsafe）"""
    
    y_sem_dim: int = 10
    """语义内容类别数（用于 MI 估计）"""
    
    def __post_init__(self):
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [512, 256]
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'model_name': self.model_name,
            'hidden_size': self.hidden_size,
            'device': self.device,
            'normalize_representations': self.normalize_representations,
            'representation_layer': self.representation_layer,
            'safety_direction_init': self.safety_direction_init,
            'critic_hidden_dims': self.critic_hidden_dims,
            'critic_dropout': self.critic_dropout,
            'y_attr_dim': self.y_attr_dim,
            'y_sem_dim': self.y_sem_dim,
        }

