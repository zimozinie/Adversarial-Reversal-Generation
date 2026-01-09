"""
损失函数模块
对应论文 Section 3.3: Safety Contrastive Regularization (SCR)

完整训练目标:
L = L_KL-SFT + λ₁ * L_dir + λ₂ * L_cons + λ₃ * L_MI
"""

from .kl_sft import compute_kl_sft_loss, KLSFTLoss
from .safety_direction import compute_safety_direction_loss, SafetyDirectionLoss
from .representation_cons import compute_representation_consistency_loss, RepresentationConsistencyLoss
from .information_sep import compute_information_separation_loss, InformationSeparationLoss
from .mi_estimation import MutualInformationEstimator
from .scr_loss import SCRLoss, compute_total_loss

__all__ = [
    # KL-SFT
    'compute_kl_sft_loss',
    'KLSFTLoss',
    
    # Safety Direction
    'compute_safety_direction_loss',
    'SafetyDirectionLoss',
    
    # Representation Consistency
    'compute_representation_consistency_loss',
    'RepresentationConsistencyLoss',
    
    # Information Separation
    'compute_information_separation_loss',
    'InformationSeparationLoss',
    
    # MI Estimation
    'MutualInformationEstimator',
    
    # Complete SCR
    'SCRLoss',
    'compute_total_loss',
]

