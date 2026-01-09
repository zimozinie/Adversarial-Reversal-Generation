"""
数据结构模块
对应论文 Section 3.1: Adversarial Reversal Data Generation
"""

from .structures import (
    SafetyAttribute,
    SemanticContent,
    ReversalPair,
)
from .dataset import ARGDataset
from .dataloader import ARGDataLoader

__all__ = [
    'SafetyAttribute',
    'SemanticContent',
    'ReversalPair',
    'ARGDataset',
    'ARGDataLoader',
]
