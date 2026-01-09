"""
Complete Safety Contrastive Regularization (SCR) Loss
对应论文 Section 3.3

完整损失:
L = L_KL-SFT + λ₁ * L_dir + λ₂ * L_cons + λ₃ * L_MI
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional
from .kl_sft import KLSFTLoss
from .safety_direction import SafetyDirectionLoss
from .representation_cons import RepresentationConsistencyLoss
from .information_sep import InformationSeparationLoss
from .mi_estimation import MutualInformationEstimator


class SCRLoss(nn.Module):
    """
    Safety Contrastive Regularization Loss
    
    对应论文完整训练目标:
    L = L_KL-SFT + λ₁ * L_dir + λ₂ * L_cons + λ₃ * L_MI
    
    组件:
    1. L_KL-SFT: KL-regularized supervised fine-tuning (公式 4)
    2. L_dir: Safety direction learning (公式 5)
    3. L_cons: Representation consistency (公式 6)
    4. L_MI: Information separation (公式 7)
    """
    
    def __init__(
        self,
        beta: float = 0.1,           # KL-SFT 中的 β
        lambda1: float = 1.0,         # L_dir 权重
        lambda2: float = 1.0,         # L_cons 权重
        lambda3: float = 0.5,         # L_MI 权重
        alpha: float = 0.5,           # L_dir 中的 α
        lambda_info: float = 1.0,     # L_MI 中的 λ_info
        mi_estimator: Optional[MutualInformationEstimator] = None
    ):
        """
        Args:
            beta: KL 惩罚系数
            lambda1: L_dir 权重
            lambda2: L_cons 权重
            lambda3: L_MI 权重
            alpha: L_dir 中的平衡系数
            lambda_info: L_MI 中的信息平衡系数
            mi_estimator: 互信息估计器（如果使用 L_MI）
        """
        super().__init__()
        
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.alpha = alpha
        self.lambda_info = lambda_info
        
        # 创建各个损失模块
        self.kl_sft_loss = KLSFTLoss(beta=beta)
        self.safety_dir_loss = SafetyDirectionLoss(alpha=alpha)
        self.repr_cons_loss = RepresentationConsistencyLoss()
        
        if mi_estimator is not None:
            self.info_sep_loss = InformationSeparationLoss(
                mi_estimator=mi_estimator,
                lambda_info=lambda_info
            )
        else:
            self.info_sep_loss = None
    
    def forward(
        self,
        # KL-SFT 参数
        logits_policy: Tensor,
        logits_ref: Optional[Tensor],
        labels: Tensor,
        attention_mask: Optional[Tensor] = None,
        # SCR 参数
        h_x: Optional[Tensor] = None,
        h_x_flip: Optional[Tensor] = None,
        h_y: Optional[Tensor] = None,
        h_y_flip: Optional[Tensor] = None,
        v_safety: Optional[Tensor] = None,
        y_attr: Optional[Tensor] = None,
        y_sem: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        计算完整的 SCR 损失
        
        Args:
            # KL-SFT Loss 参数
            logits_policy: 策略模型 logits
            logits_ref: 参考模型 logits
            labels: 标签
            attention_mask: 注意力掩码
            
            # Safety Direction Loss 参数
            h_x: h(x) - 原始指令表示
            h_x_flip: h(x_flip) - 翻转后指令表示
            v_safety: 安全方向向量
            
            # Representation Consistency Loss 参数
            h_y: h(y) - 原始响应表示
            h_y_flip: h(y_flip) - 翻转后响应表示
            
            # Information Separation Loss 参数
            y_attr: 安全属性标签
            y_sem: 语义内容标签
        
        Returns:
            包含所有损失组件的字典
        """
        losses = {}
        
        # ========== 1. L_KL-SFT ==========
        kl_sft_result = self.kl_sft_loss(
            logits_policy, logits_ref, labels, attention_mask
        )
        L_kl_sft = kl_sft_result['total']
        losses.update({
            'L_kl_sft': L_kl_sft,
            'L_kl_sft_nll': kl_sft_result['nll'],
            'L_kl_sft_kl': kl_sft_result['kl'],
        })
        
        # ========== 2. L_dir (Safety Direction) ==========
        L_dir = torch.tensor(0.0, device=logits_policy.device)
        if h_x is not None and h_x_flip is not None and v_safety is not None:
            dir_result = self.safety_dir_loss(h_x, h_x_flip, v_safety)
            L_dir = dir_result['total']
            losses.update({
                'L_dir': L_dir,
                'L_dir_alignment': dir_result['alignment'],
                'L_dir_separation': dir_result['separation'],
            })
        
        # ========== 3. L_cons (Representation Consistency) ==========
        L_cons = torch.tensor(0.0, device=logits_policy.device)
        if h_y is not None and h_y_flip is not None:
            cons_result = self.repr_cons_loss(h_y, h_y_flip)
            L_cons = cons_result['total']
            losses.update({
                'L_cons': L_cons,
                'L_cons_l2_distance': cons_result['l2_distance'],
            })
        
        # ========== 4. L_MI (Information Separation) ==========
        L_mi = torch.tensor(0.0, device=logits_policy.device)
        if self.info_sep_loss is not None and h_x is not None and y_attr is not None and y_sem is not None:
            mi_result = self.info_sep_loss(h_x, y_attr, y_sem)
            L_mi = mi_result['total']
            losses.update({
                'L_mi': L_mi,
                'L_mi_pred_attr': mi_result['L_pred_attr'],
                'L_mi_pred_sem': mi_result['L_pred_sem'],
            })
        
        # ========== Total Loss ==========
        total_loss = (
            L_kl_sft 
            + self.lambda1 * L_dir
            + self.lambda2 * L_cons
            + self.lambda3 * L_mi
        )
        
        losses['total'] = total_loss
        
        return losses


def compute_total_loss(
    # KL-SFT 参数
    logits_policy: Tensor,
    logits_ref: Optional[Tensor],
    labels: Tensor,
    attention_mask: Optional[Tensor] = None,
    # SCR 参数
    h_x: Optional[Tensor] = None,
    h_x_flip: Optional[Tensor] = None,
    h_y: Optional[Tensor] = None,
    h_y_flip: Optional[Tensor] = None,
    v_safety: Optional[Tensor] = None,
    y_attr: Optional[Tensor] = None,
    y_sem: Optional[Tensor] = None,
    # 超参数
    beta: float = 0.1,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda3: float = 0.5,
    alpha: float = 0.5,
    lambda_info: float = 1.0,
    mi_estimator: Optional[MutualInformationEstimator] = None
) -> Tensor:
    """
    计算完整损失（函数式接口）
    
    对应论文完整训练目标:
    L = L_KL-SFT + λ₁ * L_dir + λ₂ * L_cons + λ₃ * L_MI
    
    Returns:
        总损失
    """
    scr_loss = SCRLoss(
        beta=beta,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        alpha=alpha,
        lambda_info=lambda_info,
        mi_estimator=mi_estimator
    )
    
    result = scr_loss(
        logits_policy=logits_policy,
        logits_ref=logits_ref,
        labels=labels,
        attention_mask=attention_mask,
        h_x=h_x,
        h_x_flip=h_x_flip,
        h_y=h_y,
        h_y_flip=h_y_flip,
        v_safety=v_safety,
        y_attr=y_attr,
        y_sem=y_sem
    )
    
    return result['total']

