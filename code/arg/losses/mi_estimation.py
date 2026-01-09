"""
Mutual Information Estimation 实现
对应论文 Appendix C: Mutual Information Estimation

论文原文:
"I(Z;Y) = E_{p(z,y)}[log p(y|z)] + H(Y)"
"we use a variational critic q_φ(y|z)"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict
from ..models.critics import VariationalCritic


class MutualInformationEstimator:
    """
    互信息估计器
    
    对应论文 Appendix C
    
    实现变分下界估计：
    I(Z;Y) ≥ E_{p(z,y)}[log q_φ(y|z)] + H(Y)
    
    论文公式 (11):
    I(Z;Y) ≥ E_{p(z,y)}[log q_φ(y|z)] + H(Y)
    
    由于 H(Y) 是常数，最大化 I(Z;Y) 等价于最大化 E[log q_φ(y|z)]
    即最小化预测损失 L_pred(Y;Z) = -E[log q_φ(y|z)]
    
    论文公式 (12):
    L_pred(Y;Z) = -E_{(z,y)}[log q_φ(y|z)]
    """
    
    def __init__(
        self,
        z_dim: int,
        y_attr_dim: int,
        y_sem_dim: int,
        hidden_dims: list[int] = [512, 256],
        device: str = 'cpu'
    ):
        """
        Args:
            z_dim: 表示维度
            y_attr_dim: 安全属性标签维度（通常为 2: Safe/Unsafe）
            y_sem_dim: 语义内容标签维度（可以是离散化的语义类别数）
            hidden_dims: Critic 隐藏层维度
            device: 设备
        """
        self.z_dim = z_dim
        self.y_attr_dim = y_attr_dim
        self.y_sem_dim = y_sem_dim
        self.device = torch.device(device)
        
        # 创建两个 Critic
        self.critic_attr = VariationalCritic(
            z_dim=z_dim,
            y_dim=y_attr_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        self.critic_sem = VariationalCritic(
            z_dim=z_dim,
            y_dim=y_sem_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
    
    def estimate_mi_lower_bound(
        self,
        z: Tensor,
        y: Tensor,
        critic: VariationalCritic
    ) -> Tensor:
        """
        估计互信息的变分下界
        
        对应论文公式 (11) 的实现
        
        Args:
            z: 表示向量
            y: 标签
            critic: 变分 Critic
        
        Returns:
            互信息下界（不包含常数 H(Y)）
        """
        # E[log q_φ(y|z)]
        log_probs = critic.compute_log_prob(z, y)
        return log_probs.mean()
    
    def compute_L_pred(
        self,
        z: Tensor,
        y: Tensor,
        critic: VariationalCritic
    ) -> Tensor:
        """
        计算预测损失
        
        对应论文 Appendix C 公式 (12):
        L_pred(Y;Z) = -E_{(z,y)}[log q_φ(y|z)]
        
        Args:
            z: 表示向量
            y: 标签
            critic: 变分 Critic
        
        Returns:
            预测损失
        """
        return critic.compute_loss(z, y)
    
    def compute_mi_objective(
        self,
        z: Tensor,
        y_attr: Tensor,
        y_sem: Tensor,
        lambda_info: float = 1.0
    ) -> Dict[str, Tensor]:
        """
        计算互信息目标
        
        对应论文公式 (13):
        L_MI ∝ L_pred(Y_attr; Z) - λ_info * L_pred(Y_sem; Z)
        
        最大化 I(Z; Y_attr) - λ_info * I(Z; Y_sem)
        等价于最小化 L_pred(Y_attr; Z) - λ_info * L_pred(Y_sem; Z)
        
        注意：为了通过梯度下降最大化互信息，我们需要最小化负的互信息
        
        Args:
            z: 表示向量
            y_attr: 安全属性标签
            y_sem: 语义内容标签
            lambda_info: λ_info 参数
        
        Returns:
            包含各项损失的字典
        """
        # L_pred(Y_attr; Z)
        L_pred_attr = self.compute_L_pred(z, y_attr, self.critic_attr)
        
        # L_pred(Y_sem; Z)
        L_pred_sem = self.compute_L_pred(z, y_sem, self.critic_sem)
        
        # 组合（论文公式 13）
        # 目标：最大化 I(Z; Y_attr) - λ_info * I(Z; Y_sem)
        # 等价于最小化 L_pred(Y_attr; Z) - λ_info * L_pred(Y_sem; Z)
        L_MI = L_pred_attr - lambda_info * L_pred_sem
        
        return {
            'total': L_MI,
            'L_pred_attr': L_pred_attr,
            'L_pred_sem': L_pred_sem,
        }
    
    def get_critics(self) -> Dict[str, VariationalCritic]:
        """
        获取 Critic 网络
        
        Returns:
            包含 attr 和 sem critic 的字典
        """
        return {
            'attr': self.critic_attr,
            'sem': self.critic_sem,
        }
    
    def parameters(self):
        """
        获取所有可训练参数
        
        用于优化器
        """
        return list(self.critic_attr.parameters()) + list(self.critic_sem.parameters())
    
    def to(self, device):
        """移动到指定设备"""
        self.device = device
        self.critic_attr = self.critic_attr.to(device)
        self.critic_sem = self.critic_sem.to(device)
        return self
    
    def train(self):
        """设置为训练模式"""
        self.critic_attr.train()
        self.critic_sem.train()
    
    def eval(self):
        """设置为评估模式"""
        self.critic_attr.eval()
        self.critic_sem.eval()


def create_semantic_labels(texts: list[str], num_classes: int = 10) -> Tensor:
    """
    创建语义标签（占位实现）
    
    实际应用中，这应该通过聚类或预训练的语义分类器生成
    
    Args:
        texts: 文本列表
        num_classes: 语义类别数
    
    Returns:
        语义标签, shape: (batch_size,)
    """
    # 简单的哈希函数作为占位
    labels = [hash(text) % num_classes for text in texts]
    return torch.tensor(labels, dtype=torch.long)

