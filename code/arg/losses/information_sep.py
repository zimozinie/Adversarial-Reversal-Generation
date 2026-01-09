"""
Information Separation Loss 实现
对应论文 Section 3.3, 公式 (7)

论文原文:
"L_MI = I(Z; Y_attr) - λ_info * I(Z; Y_sem)"
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict
from .mi_estimation import MutualInformationEstimator


class InformationSeparationLoss(nn.Module):
    """
    信息分离损失
    
    对应论文公式 (7):
    L_MI = I(Z; Y_attr) - λ_info * I(Z; Y_sem)
    
    目标:
    - 最大化 I(Z; Y_attr): 表示应包含安全属性信息
    - 最小化 I(Z; Y_sem): 表示应排除语义内容信息（避免 shortcut）
    
    论文原文:
    "L_MI enforces information-theoretic separation: representations should encode 
    safety attributes but not be confounded by semantic content"
    
    实现通过 Appendix C 的变分下界
    """
    
    def __init__(
        self,
        mi_estimator: MutualInformationEstimator,
        lambda_info: float = 1.0
    ):
        """
        Args:
            mi_estimator: 互信息估计器
            lambda_info: λ_info 参数（论文公式 7）
        """
        super().__init__()
        self.mi_estimator = mi_estimator
        self.lambda_info = lambda_info
    
    def forward(
        self,
        z: Tensor,
        y_attr: Tensor,
        y_sem: Tensor
    ) -> Dict[str, Tensor]:
        """
        计算信息分离损失
        
        Args:
            z: 表示向量 Z = h(x), shape: (batch_size, hidden_dim)
            y_attr: 安全属性标签 Y_attr, shape: (batch_size,)
            y_sem: 语义内容标签 Y_sem, shape: (batch_size,)
        
        Returns:
            包含损失组件的字典
        """
        result = self.mi_estimator.compute_mi_objective(
            z=z,
            y_attr=y_attr,
            y_sem=y_sem,
            lambda_info=self.lambda_info
        )
        
        return {
            'total': result['total'],
            'I_Z_Yattr': -result['L_pred_attr'],  # 负号因为 L_pred 是负对数
            'I_Z_Ysem': -result['L_pred_sem'],
            'L_pred_attr': result['L_pred_attr'],
            'L_pred_sem': result['L_pred_sem'],
        }


def compute_information_separation_loss(
    z: Tensor,
    y_attr: Tensor,
    y_sem: Tensor,
    mi_estimator: MutualInformationEstimator,
    lambda_info: float = 1.0
) -> Tensor:
    """
    计算信息分离损失（函数式接口）
    
    对应论文公式 (7) 和 Appendix C 公式 (13)
    
    Args:
        z: 表示向量
        y_attr: 安全属性标签
        y_sem: 语义内容标签
        mi_estimator: 互信息估计器
        lambda_info: λ_info 参数
    
    Returns:
        总损失
    """
    loss_module = InformationSeparationLoss(
        mi_estimator=mi_estimator,
        lambda_info=lambda_info
    )
    result = loss_module(z, y_attr, y_sem)
    return result['total']

