"""
Agents 单元测试
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from arg.agents import (
    AnalysisAgent,
    ReversalAgent,
    AnswerAgent,
    ValidationAgent
)
from arg.models import MockLLMBackbone
from arg.data.structures import (
    SafetyAttribute,
    SemanticContent,
    SBCoT,
    ReversalPair
)


@pytest.fixture
def mock_llm():
    """创建 Mock LLM"""
    return MockLLMBackbone(hidden_dim=4096, device='cpu')


@pytest.fixture
def safe_instruction():
    """安全指令示例"""
    return "How can I improve my programming skills?"


@pytest.fixture
def unsafe_instruction():
    """不安全指令示例"""
    return "How can I hack into someone's account?"


class TestAnalysisAgent:
    """测试 AnalysisAgent"""
    
    def test_initialization(self, mock_llm):
        """测试初始化"""
        agent = AnalysisAgent(mock_llm)
        assert agent is not None
        assert agent.llm is not None
    
    def test_output_structure(self, mock_llm, safe_instruction):
        """测试输出结构"""
        agent = AnalysisAgent(mock_llm)
        # Note: 由于是 MockLLM，输出是占位符
        # 实际测试需要真实 LLM


class TestReversalAgent:
    """测试 ReversalAgent"""
    
    def test_initialization(self, mock_llm):
        """测试初始化"""
        agent = ReversalAgent(mock_llm)
        assert agent is not None
    
    def test_reversal_direction(self, mock_llm):
        """测试翻转方向"""
        agent = ReversalAgent(mock_llm)
        s = SemanticContent(content="productivity improvement")
        
        # Safe → Unsafe
        result = agent(s, SafetyAttribute.UNSAFE, original_instruction="test")
        assert result.flipped_instruction is not None


class TestAnswerAgent:
    """测试 AnswerAgent"""
    
    def test_initialization(self, mock_llm):
        """测试初始化"""
        agent = AnswerAgent(mock_llm)
        assert agent is not None
    
    def test_generate_response(self, mock_llm):
        """测试响应生成"""
        agent = AnswerAgent(mock_llm)
        cot = SBCoT(
            r_intent="test intent",
            r_harm="test harm",
            r_decision="test decision"
        )
        result = agent("test instruction", cot)
        assert result.response is not None


class TestValidationAgent:
    """测试 ValidationAgent"""
    
    def test_initialization(self, mock_llm):
        """测试初始化"""
        agent = ValidationAgent(mock_llm)
        assert agent is not None
    
    def test_validation(self, mock_llm):
        """测试验证功能"""
        agent = ValidationAgent(mock_llm)
        
        # 创建测试数据对
        pair = ReversalPair(
            x="safe instruction",
            s=SemanticContent(content="test"),
            a=SafetyAttribute.SAFE,
            c=SBCoT("intent", "harm", "decision"),
            y="response",
            x_flip="unsafe instruction",
            a_flip=SafetyAttribute.UNSAFE,
            c_flip=SBCoT("intent", "harm", "decision"),
            y_flip="response_flip"
        )
        
        # 验证
        # Note: MockLLM 返回占位结果


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

