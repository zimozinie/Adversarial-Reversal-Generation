"""
KL-SFT Loss 实现
对应论文 Section 3.3, 公式 (4)

论文原文:
"L_KL-SFT = E_{(x,y)~D}[-log p_θ(y|x) + β * KL(p_θ(·|x) || p_ref(·|x))]"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any


class KLSFTLoss(nn.Module):
    """
    KL-Supervised Fine-Tuning Loss
    
    对应论文公式 (4):
    L_KL-SFT = E_{(x,y)~D}[-log p_θ(y|x) + β * KL(p_θ(·|x) || p_ref(·|x))]
    
    组成部分：
    1. 负对数似然（NLL）：-log p_θ(y|x) - 监督学习项
    2. KL 散度：KL(p_θ(·|x) || p_ref(·|x)) - 防止与参考模型偏离过大
    
    论文原文：
    "to prevent utility degradation, we retain a KL-penalty to the base policy"
    """
    
    def __init__(self, beta: float = 0.1):
        """
        Args:
            beta: KL 惩罚系数（论文中的 β）
        """
        super().__init__()
        self.beta = beta
    
    def forward(
        self,
        logits_policy: Tensor,
        logits_ref: Optional[Tensor],
        labels: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        计算 KL-SFT 损失
        
        Args:
            logits_policy: 策略模型的 logits, shape: (batch_size, seq_len, vocab_size)
            logits_ref: 参考模型的 logits（可选）, same shape
            labels: 标签, shape: (batch_size, seq_len)
            attention_mask: 注意力掩码（可选）, shape: (batch_size, seq_len)
        
        Returns:
            包含损失组件的字典
        """
        batch_size, seq_len, vocab_size = logits_policy.shape
        
        # ========== 1. NLL Loss: -log p_θ(y|x) ==========
        nll_loss = F.cross_entropy(
            logits_policy.view(-1, vocab_size),
            labels.view(-1),
            reduction='none'
        ).view(batch_size, seq_len)
        
        # 应用 attention mask
        if attention_mask is not None:
            nll_loss = nll_loss * attention_mask
            nll_loss = nll_loss.sum() / attention_mask.sum()
        else:
            nll_loss = nll_loss.mean()
        
        # ========== 2. KL Divergence: KL(p_θ || p_ref) ==========
        kl_loss = torch.tensor(0.0, device=logits_policy.device)
        
        if logits_ref is not None:
            # 计算 KL 散度
            log_probs_policy = F.log_softmax(logits_policy, dim=-1)
            probs_ref = F.softmax(logits_ref, dim=-1)
            
            # KL(p_θ || p_ref) = sum(p_θ * (log p_θ - log p_ref))
            kl_div = F.kl_div(
                log_probs_policy.view(-1, vocab_size),
                probs_ref.view(-1, vocab_size),
                reduction='none'
            ).sum(dim=-1).view(batch_size, seq_len)
            
            # 应用 attention mask
            if attention_mask is not None:
                kl_div = kl_div * attention_mask
                kl_loss = kl_div.sum() / attention_mask.sum()
            else:
                kl_loss = kl_div.mean()
        
        # ========== 3. Total KL-SFT Loss ==========
        total_loss = nll_loss + self.beta * kl_loss
        
        return {
            'total': total_loss,
            'nll': nll_loss,
            'kl': kl_loss,
        }


def compute_kl_sft_loss(
    logits_policy: Tensor,
    logits_ref: Optional[Tensor],
    labels: Tensor,
    attention_mask: Optional[Tensor] = None,
    beta: float = 0.1
) -> Tensor:
    """
    计算 KL-SFT 损失（函数式接口）
    
    对应论文公式 (4)
    
    Args:
        logits_policy: 策略模型的 logits
        logits_ref: 参考模型的 logits
        labels: 标签
        attention_mask: 注意力掩码
        beta: KL 惩罚系数
    
    Returns:
        总损失
    """
    loss_module = KLSFTLoss(beta=beta)
    result = loss_module(logits_policy, logits_ref, labels, attention_mask)
    return result['total']


class SimplifiedSFTLoss(nn.Module):
    """
    简化版 SFT Loss（无 KL 惩罚）
    
    当没有参考模型时使用
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        计算标准的监督微调损失
        
        Args:
            logits: 模型 logits
            labels: 标签
            attention_mask: 注意力掩码
        
        Returns:
            损失
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            reduction='none'
        ).view(batch_size, seq_len)
        
        if attention_mask is not None:
            loss = loss * attention_mask
            loss = loss.sum() / attention_mask.sum()
        else:
            loss = loss.mean()
        
        return loss

