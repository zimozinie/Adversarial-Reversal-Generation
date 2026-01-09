"""
Multi-Agent Pipeline 模块
对应论文 Section 3.2: Multi-Agent Safety Reasoning Pipeline

论文原文:
"We develop a multi-agent safety reasoning pipeline (analysis, reversal, answer, and validation) 
with integrated SB-CoT to ensure high-fidelity attribute inversion, semantic preservation, 
and boundary-aware decisions."
"""

from .base import BaseAgent
from .analysis_agent import AnalysisAgent
from .reversal_agent import ReversalAgent
from .answer_agent import AnswerAgent
from .validation_agent import ValidationAgent
from .reversal_strategies import ReversalStrategy

__all__ = [
    'BaseAgent',
    'AnalysisAgent',
    'ReversalAgent',
    'AnswerAgent',
    'ValidationAgent',
    'ReversalStrategy',
]

