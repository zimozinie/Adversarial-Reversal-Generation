"""
Auxiliary Critics for Mutual Information Estimation
对应论文 Appendix C: Mutual Information Estimation

论文原文:
"we use a variational critic q_φ(y|z)"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Literal


class VariationalCritic(nn.Module):
    """
    变分 Critic 网络
    
    对应论文 Appendix C 中的 q_φ(y|z)
    用于互信息估计的变分下界
    
    论文公式 (11):
    I(Z;Y) ≥ E_{p(z,y)}[log q_φ(y|z)] + H(Y)
    
    实现为分类器：给定表示 z，预测标签 y
    """
    
    def __init__(
        self,
        z_dim: int,
        y_dim: int,
        hidden_dims: list[int] = [512, 256],
        activation: str = 'relu',
        dropout: float = 0.1
    ):
        """
        Args:
            z_dim: 表示维度（输入）
            y_dim: 标签维度（输出，类别数）
            hidden_dims: 隐藏层维度列表
            activation: 激活函数
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.z_dim = z_dim
        self.y_dim = y_dim
        
        # 构建 MLP
        layers = []
        input_dim = z_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, y_dim))
        
        self.predictor = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, z: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            z: 表示向量, shape: (batch_size, z_dim)
        
        Returns:
            logits, shape: (batch_size, y_dim)
        """
        return self.predictor(z)
    
    def predict_proba(self, z: Tensor) -> Tensor:
        """
        预测概率分布
        
        Args:
            z: 表示向量
        
        Returns:
            概率分布, shape: (batch_size, y_dim)
        """
        logits = self.forward(z)
        return F.softmax(logits, dim=-1)
    
    def compute_log_prob(self, z: Tensor, y: Tensor) -> Tensor:
        """
        计算 log q_φ(y|z)
        
        对应论文 Appendix C 公式 (11)
        
        Args:
            z: 表示向量, shape: (batch_size, z_dim)
            y: 标签, shape: (batch_size,) - 类别索引
        
        Returns:
            log 概率, shape: (batch_size,)
        """
        logits = self.forward(z)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 选择对应标签的 log 概率
        return log_probs.gather(1, y.unsqueeze(1)).squeeze(1)
    
    def compute_loss(self, z: Tensor, y: Tensor) -> Tensor:
        """
        计算预测损失
        
        对应论文 Appendix C 公式 (12):
        L_pred(Y;Z) = -E_{(z,y)}[log q_φ(y|z)]
        
        Args:
            z: 表示向量
            y: 标签
        
        Returns:
            损失值
        """
        logits = self.forward(z)
        return F.cross_entropy(logits, y)


class InfoNCECritic(nn.Module):
    """
    InfoNCE Critic
    
    用于对比学习式的互信息估计
    当标签空间较大时的替代方案
    
    论文 Appendix C 提到：
    "or its contrastive variant such as InfoNCE when Y is large"
    """
    
    def __init__(
        self,
        z_dim: int,
        projection_dim: int = 256,
        temperature: float = 0.07
    ):
        """
        Args:
            z_dim: 表示维度
            projection_dim: 投影空间维度
            temperature: InfoNCE 温度参数
        """
        super().__init__()
        
        self.z_dim = z_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        
        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(z_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(self, z: Tensor) -> Tensor:
        """
        投影到对比学习空间
        
        Args:
            z: 表示向量, shape: (batch_size, z_dim)
        
        Returns:
            投影向量, shape: (batch_size, projection_dim)
        """
        proj = self.projector(z)
        return F.normalize(proj, p=2, dim=-1)
    
    def compute_infonce_loss(
        self,
        z1: Tensor,
        z2: Tensor,
        negative_samples: Optional[Tensor] = None
    ) -> Tensor:
        """
        计算 InfoNCE 损失
        
        Args:
            z1: 正样本表示1
            z2: 正样本表示2（与 z1 对应）
            negative_samples: 负样本表示（可选）
        
        Returns:
            InfoNCE 损失
        """
        # 投影
        proj1 = self.forward(z1)
        proj2 = self.forward(z2)
        
        # 计算相似度
        batch_size = z1.size(0)
        
        # 正样本相似度
        pos_sim = torch.sum(proj1 * proj2, dim=-1) / self.temperature
        
        # 负样本相似度（使用 batch 内其他样本）
        if negative_samples is None:
            # 使用 batch 内对比
            all_proj = torch.cat([proj1, proj2], dim=0)
            sim_matrix = torch.matmul(proj1, all_proj.T) / self.temperature
            
            # 构建标签（对角线为正样本）
            labels = torch.arange(batch_size, device=z1.device)
            
            # InfoNCE 损失
            loss = F.cross_entropy(sim_matrix, labels)
        else:
            # 使用显式负样本
            neg_proj = self.forward(negative_samples)
            neg_sim = torch.matmul(proj1, neg_proj.T) / self.temperature
            
            # 组合正负样本
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            labels = torch.zeros(batch_size, dtype=torch.long, device=z1.device)
            
            loss = F.cross_entropy(logits, labels)
        
        return loss


class CriticEnsemble(nn.Module):
    """
    Critic 集成
    
    用于同时管理多个 Critic（例如 attribute critic 和 semantic critic）
    """
    
    def __init__(
        self,
        attr_critic: VariationalCritic,
        sem_critic: VariationalCritic
    ):
        """
        Args:
            attr_critic: 安全属性 Critic
            sem_critic: 语义内容 Critic
        """
        super().__init__()
        
        self.attr_critic = attr_critic
        self.sem_critic = sem_critic
    
    def forward(
        self,
        z: Tensor,
        mode: Literal['attr', 'sem']
    ) -> Tensor:
        """
        前向传播
        
        Args:
            z: 表示向量
            mode: 'attr' 或 'sem'
        
        Returns:
            logits
        """
        if mode == 'attr':
            return self.attr_critic(z)
        elif mode == 'sem':
            return self.sem_critic(z)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def compute_mi_loss(
        self,
        z: Tensor,
        y_attr: Tensor,
        y_sem: Tensor,
        lambda_info: float = 1.0
    ) -> tuple[Tensor, dict]:
        """
        计算互信息损失
        
        对应论文公式 (7) 和 Appendix C 公式 (13):
        L_MI = I(Z; Y_attr) - λ_info * I(Z; Y_sem)
        
        Args:
            z: 表示向量
            y_attr: 安全属性标签
            y_sem: 语义内容标签
            lambda_info: λ_info 参数
        
        Returns:
            (总损失, 详细损失字典)
        """
        # 计算 L_pred(Y_attr; Z)
        loss_attr = self.attr_critic.compute_loss(z, y_attr)
        
        # 计算 L_pred(Y_sem; Z)
        loss_sem = self.sem_critic.compute_loss(z, y_sem)
        
        # 组合损失（论文 Appendix C 公式 13）
        total_loss = loss_attr - lambda_info * loss_sem
        
        return total_loss, {
            'attr': loss_attr.item(),
            'sem': loss_sem.item(),
            'total': total_loss.item()
        }

