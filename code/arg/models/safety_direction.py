"""
安全方向向量学习
对应论文 Section 3.3, 公式 (5)

论文原文:
"v_safety: a learned safety direction vector"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class SafetyDirectionLearner(nn.Module):
    """
    可学习的安全方向向量
    
    对应论文公式 (5) 中的 v_safety
    
    论文原文:
    "We learn a global safety direction vector v_safety that captures 
    the latent axis along which safe and unsafe representations separate"
    
    关键约束：
    - v_safety 必须 L2 归一化
    - 通过 L_dir 损失进行优化
    """
    
    def __init__(self, hidden_dim: int, init_method: str = 'random'):
        """
        Args:
            hidden_dim: 隐藏层维度（与 LLM backbone 一致）
            init_method: 初始化方法 ('random', 'zero', 'pca')
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.init_method = init_method
        
        # 可学习的安全方向参数
        self.v_safety = nn.Parameter(torch.zeros(hidden_dim))
        
        # 初始化
        self._initialize()
    
    def _initialize(self) -> None:
        """
        初始化 v_safety
        """
        if self.init_method == 'random':
            # 随机初始化并归一化
            nn.init.normal_(self.v_safety, mean=0.0, std=0.01)
        elif self.init_method == 'zero':
            # 零初始化（将通过训练学习）
            nn.init.zeros_(self.v_safety)
        elif self.init_method == 'pca':
            # PCA 初始化（需要数据，这里用随机代替）
            nn.init.normal_(self.v_safety, mean=0.0, std=0.01)
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")
        
        # L2 归一化
        with torch.no_grad():
            self.v_safety.data = F.normalize(self.v_safety.data, p=2, dim=0)
    
    def forward(self) -> Tensor:
        """
        返回归一化的安全方向向量
        
        Returns:
            v_safety, shape: (hidden_dim,)
        """
        # 确保始终归一化
        return F.normalize(self.v_safety, p=2, dim=0)
    
    def get_direction(self) -> Tensor:
        """
        获取安全方向向量（便捷方法）
        
        Returns:
            归一化的 v_safety
        """
        return self.forward()
    
    def compute_projection(self, representations: Tensor) -> Tensor:
        """
        计算表示在安全方向上的投影
        
        用于分析表示的安全倾向
        
        Args:
            representations: 表示向量, shape: (batch_size, hidden_dim) 或 (hidden_dim,)
        
        Returns:
            投影值, shape: (batch_size,) 或 scalar
        """
        v = self.get_direction()
        
        if representations.dim() == 1:
            # 单个表示
            return torch.dot(representations, v)
        else:
            # 批量表示
            return torch.matmul(representations, v)
    
    def compute_cosine_similarity(
        self, 
        representations: Tensor, 
        normalized: bool = True
    ) -> Tensor:
        """
        计算表示与安全方向的余弦相似度
        
        对应论文公式 (5) 中的 cos(v_safety, h(x))
        
        Args:
            representations: 表示向量
            normalized: 表示是否已归一化
        
        Returns:
            余弦相似度
        """
        v = self.get_direction()
        
        if not normalized:
            representations = F.normalize(representations, p=2, dim=-1)
        
        if representations.dim() == 1:
            # 单个表示
            return F.cosine_similarity(
                representations.unsqueeze(0),
                v.unsqueeze(0),
                dim=1
            ).squeeze()
        else:
            # 批量表示
            return F.cosine_similarity(
                representations,
                v.unsqueeze(0).expand_as(representations),
                dim=1
            )
    
    def get_norm(self) -> float:
        """
        获取 v_safety 的范数（应该接近 1.0）
        
        Returns:
            L2 范数
        """
        return torch.norm(self.v_safety, p=2).item()
    
    def visualize_direction(self) -> dict:
        """
        可视化安全方向信息
        
        Returns:
            包含方向统计信息的字典
        """
        v = self.get_direction().detach().cpu()
        
        return {
            'norm': torch.norm(v, p=2).item(),
            'mean': v.mean().item(),
            'std': v.std().item(),
            'min': v.min().item(),
            'max': v.max().item(),
            'sparsity': (v.abs() < 0.01).sum().item() / v.numel(),
        }


class SafetyDirectionAnalyzer:
    """
    安全方向分析器
    
    用于分析和可视化学到的安全方向
    """
    
    def __init__(self, learner: SafetyDirectionLearner):
        """
        Args:
            learner: SafetyDirectionLearner 实例
        """
        self.learner = learner
    
    def analyze_representation(
        self, 
        h_x: Tensor, 
        h_x_flip: Tensor
    ) -> dict:
        """
        分析表示对的安全方向对齐
        
        Args:
            h_x: 原始表示
            h_x_flip: 翻转后表示
        
        Returns:
            分析结果字典
        """
        v = self.learner.get_direction()
        
        # 计算投影
        proj_x = self.learner.compute_projection(h_x)
        proj_x_flip = self.learner.compute_projection(h_x_flip)
        
        # 计算余弦相似度
        cos_x = self.learner.compute_cosine_similarity(h_x)
        cos_x_flip = self.learner.compute_cosine_similarity(h_x_flip)
        
        # 计算差异向量的对齐
        g_x = h_x - h_x_flip
        cos_g = F.cosine_similarity(
            g_x.unsqueeze(0),
            v.unsqueeze(0),
            dim=1
        ).item()
        
        return {
            'projection_original': proj_x.item() if proj_x.dim() == 0 else proj_x.mean().item(),
            'projection_flipped': proj_x_flip.item() if proj_x_flip.dim() == 0 else proj_x_flip.mean().item(),
            'cosine_original': cos_x.item() if cos_x.dim() == 0 else cos_x.mean().item(),
            'cosine_flipped': cos_x_flip.item() if cos_x_flip.dim() == 0 else cos_x_flip.mean().item(),
            'cosine_difference': cos_g,
            'separation': abs((proj_x - proj_x_flip).item() if proj_x.dim() == 0 
                            else (proj_x - proj_x_flip).mean().item()),
        }

