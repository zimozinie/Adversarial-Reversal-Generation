"""
评估模块
对应论文 Section 4: Evaluation

论文评估基准:
- HarmBench: Attack Success Rate (ASR)
- XSTest: Over-refusal rate
- MT-Bench: General reasoning capabilities
"""

from .eval_hooks import (
    HarmBenchEvaluator,
    XSTestEvaluator,
    MTBenchEvaluator,
    EvaluationManager
)

__all__ = [
    'HarmBenchEvaluator',
    'XSTestEvaluator',
    'MTBenchEvaluator',
    'EvaluationManager',
]

