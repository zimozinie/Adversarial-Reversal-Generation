"""
模型组件模块
对应论文 Section 3.3: Safety Contrastive Regularization (SCR)
"""

from .backbone import LLMBackbone
from .representation import RepresentationExtractor
from .safety_direction import SafetyDirectionLearner
from .critics import VariationalCritic

__all__ = [
    'LLMBackbone',
    'RepresentationExtractor',
    'SafetyDirectionLearner',
    'VariationalCritic',
]

