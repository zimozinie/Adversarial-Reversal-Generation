"""
Safety Direction Loss 实现
对应论文 Section 3.3, 公式 (5)

论文原文:
"L_dir = -cos(v_safety, g(x)) - α * [cos(v_safety, h(x)) - cos(v_safety, h(x_flip))]"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class SafetyDirectionLoss(nn.Module):
    """
    安全方向损失
    
    对应论文公式 (5):
    L_dir = -cos(v_safety, g(x)) - α * [cos(v_safety, h(x)) - cos(v_safety, h(x_flip))]
    
    其中:
    - v_safety: 学习的安全方向向量
    - g(x) = h(x) - h(x_flip): 差异向量
    - α: 平衡系数
    
    目标:
    1. 使 g(x) 与 v_safety 对齐：-cos(v_safety, g(x))
    2. 使 safe 和 unsafe 的表示在 v_safety 方向上分离
    
    论文原文:
    "We learn a global safety direction vector v_safety that captures 
    the latent axis along which safe and unsafe representations separate"
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: 平衡系数（论文中的 α）
        """
        super().__init__()
        self.alpha = alpha
    
    def forward(
        self,
        h_x: Tensor,
        h_x_flip: Tensor,
        v_safety: Tensor,
        g_x: Optional[Tensor] = None
    ) -> dict[str, Tensor]:
        """
        计算安全方向损失
        
        Args:
            h_x: h(x) - 原始指令的表示, shape: (batch_size, hidden_dim) 或 (hidden_dim,)
            h_x_flip: h(x_flip) - 翻转后指令的表示, same shape
            v_safety: 安全方向向量, shape: (hidden_dim,)
            g_x: g(x) = h(x) - h(x_flip) (可选，如未提供则自动计算)
        
        Returns:
            包含损失组件的字典
        """
        # 计算 g(x) if not provided
        if g_x is None:
            g_x = h_x - h_x_flip
        
        # 确保 v_safety 归一化
        v_safety = F.normalize(v_safety, p=2, dim=-1)
        
        # ========== 第一项: -cos(v_safety, g(x)) ==========
        # 使差异向量 g(x) 与安全方向对齐
        cos_v_g = F.cosine_similarity(
            v_safety.unsqueeze(0) if v_safety.dim() == 1 else v_safety,
            g_x.unsqueeze(0) if g_x.dim() == 1 else g_x,
            dim=-1
        )
        term1 = -cos_v_g.mean()
        
        # ========== 第二项: -α * [cos(v_safety, h(x)) - cos(v_safety, h(x_flip))] ==========
        # 使 safe 和 unsafe 在安全方向上分离
        
        # cos(v_safety, h(x))
        cos_v_hx = F.cosine_similarity(
            v_safety.unsqueeze(0) if v_safety.dim() == 1 else v_safety,
            h_x.unsqueeze(0) if h_x.dim() == 1 else h_x,
            dim=-1
        )
        
        # cos(v_safety, h(x_flip))
        cos_v_hx_flip = F.cosine_similarity(
            v_safety.unsqueeze(0) if v_safety.dim() == 1 else v_safety,
            h_x_flip.unsqueeze(0) if h_x_flip.dim() == 1 else h_x_flip,
            dim=-1
        )
        
        term2 = -self.alpha * (cos_v_hx - cos_v_hx_flip).mean()
        
        # ========== Total Loss ==========
        total_loss = term1 + term2
        
        return {
            'total': total_loss,
            'alignment': term1,  # -cos(v_safety, g(x))
            'separation': term2,  # -α * [...]
            'cos_v_g': cos_v_g.mean(),
            'cos_v_hx': cos_v_hx.mean(),
            'cos_v_hx_flip': cos_v_hx_flip.mean(),
        }


def compute_safety_direction_loss(
    h_x: Tensor,
    h_x_flip: Tensor,
    v_safety: Tensor,
    alpha: float = 0.5
) -> Tensor:
    """
    计算安全方向损失（函数式接口）
    
    对应论文公式 (5)
    
    Args:
        h_x: h(x) - 原始表示
        h_x_flip: h(x_flip) - 翻转后表示
        v_safety: 安全方向向量
        alpha: 平衡系数
    
    Returns:
        总损失
    """
    loss_module = SafetyDirectionLoss(alpha=alpha)
    
    # 计算 g(x)
    g_x = h_x - h_x_flip
    
    result = loss_module(h_x, h_x_flip, v_safety, g_x)
    return result['total']

