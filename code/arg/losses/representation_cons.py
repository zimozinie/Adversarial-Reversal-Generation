"""
Representation Consistency Loss 实现
对应论文 Section 3.3, 公式 (6)

论文原文:
"L_cons = ||h(y) - h(y_flip)||²"
"""

import torch
import torch.nn as nn
from torch import Tensor


class RepresentationConsistencyLoss(nn.Module):
    """
    表示一致性损失
    
    对应论文公式 (6):
    L_cons = ||h(y) - h(y_flip)||²
    
    目标:
    确保 y 和 y_flip 的表示在非安全维度上一致
    即：翻转只改变安全属性，不改变语义内容的表示
    
    论文原文:
    "L_cons enforces that responses y and y_flip preserve semantic similarity 
    in the representation space outside the safety-relevant subspace"
    """
    
    def __init__(self, norm: int = 2):
        """
        Args:
            norm: 范数类型（默认 L2）
        """
        super().__init__()
        self.norm = norm
    
    def forward(
        self,
        h_y: Tensor,
        h_y_flip: Tensor
    ) -> dict[str, Tensor]:
        """
        计算表示一致性损失
        
        Args:
            h_y: h(y) - 原始响应的表示, shape: (batch_size, hidden_dim) 或 (hidden_dim,)
            h_y_flip: h(y_flip) - 翻转后响应的表示, same shape
        
        Returns:
            包含损失的字典
        """
        # 计算 L2 距离的平方
        diff = h_y - h_y_flip
        
        if self.norm == 2:
            # L2 范数的平方
            loss = torch.sum(diff ** 2, dim=-1)
        else:
            # 一般 Lp 范数
            loss = torch.norm(diff, p=self.norm, dim=-1)
        
        # 如果是批量，取平均
        if loss.dim() > 0:
            loss = loss.mean()
        
        # 计算额外的统计信息
        distance = torch.norm(diff, p=2, dim=-1).mean() if diff.dim() > 1 else torch.norm(diff, p=2)
        cosine_sim = torch.nn.functional.cosine_similarity(
            h_y.unsqueeze(0) if h_y.dim() == 1 else h_y,
            h_y_flip.unsqueeze(0) if h_y_flip.dim() == 1 else h_y_flip,
            dim=-1
        ).mean()
        
        return {
            'total': loss,
            'l2_distance': distance,
            'cosine_similarity': cosine_sim,
        }


def compute_representation_consistency_loss(
    h_y: Tensor,
    h_y_flip: Tensor,
    norm: int = 2
) -> Tensor:
    """
    计算表示一致性损失（函数式接口）
    
    对应论文公式 (6)
    
    Args:
        h_y: h(y) - 原始响应表示
        h_y_flip: h(y_flip) - 翻转后响应表示
        norm: 范数类型
    
    Returns:
        损失值
    """
    loss_module = RepresentationConsistencyLoss(norm=norm)
    result = loss_module(h_y, h_y_flip)
    return result['total']

