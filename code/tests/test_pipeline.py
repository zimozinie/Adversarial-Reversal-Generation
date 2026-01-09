"""
Pipeline 单元测试
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from arg.pipeline import ARGReversalPipeline
from arg.agents import (
    AnalysisAgent,
    ReversalAgent,
    AnswerAgent,
    ValidationAgent
)
from arg.models import MockLLMBackbone
from arg.data.structures import SafetyAttribute


@pytest.fixture
def pipeline():
    """创建测试 Pipeline"""
    llm = MockLLMBackbone(hidden_dim=4096, device='cpu')
    
    return ARGReversalPipeline(
        analysis_agent=AnalysisAgent(llm),
        reversal_agent=ReversalAgent(llm),
        answer_agent=AnswerAgent(llm),
        validation_agent=ValidationAgent(llm)
    )


class TestARGReversalPipeline:
    """测试 ARG Pipeline"""
    
    def test_initialization(self, pipeline):
        """测试初始化"""
        assert pipeline is not None
        assert pipeline.analysis_agent is not None
        assert pipeline.reversal_agent is not None
        assert pipeline.answer_agent is not None
        assert pipeline.validation_agent is not None
    
    def test_safe_to_unsafe(self, pipeline):
        """测试 Safe → Unsafe 翻转"""
        instruction = "How can I learn Python programming?"
        # Note: MockLLM 会返回占位结果
    
    def test_unsafe_to_safe(self, pipeline):
        """测试 Unsafe → Safe 翻转"""
        instruction = "How can I bypass security systems?"
        # Note: MockLLM 会返回占位结果


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

