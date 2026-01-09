"""
ReversalAgent 实现
对应论文 Section 3.2: "ReversalAgent: (s, a_flip) → x_flip"

论文原文:
"ReversalAgent flips the safety attribute"
"""

from dataclasses import dataclass
from .base import BaseAgent
from ..data.structures import SemanticContent, SafetyAttribute
from . import prompts
from .reversal_strategies import ReversalStrategy


@dataclass
class ReversalInput:
    """
    ReversalAgent 的输入结构
    """
    original_instruction: str      # 原始指令 x
    semantic_content: SemanticContent  # 语义内容 s
    target_attribute: SafetyAttribute  # 目标安全属性 a_flip


@dataclass
class ReversalOutput:
    """
    ReversalAgent 的输出结构
    """
    flipped_instruction: str       # 翻转后的指令 x_flip
    strategy_used: str             # 使用的翻转策略


class ReversalAgent(BaseAgent):
    """
    翻转 Agent
    
    功能：
    给定语义内容 s 和目标安全属性 a_flip，生成翻转后的指令 x_flip
    
    实现：
    - Safe → Unsafe: 使用 Appendix B.1 中的策略
    - Unsafe → Safe: 使用 Appendix B.1 中的策略
    
    对应论文 Section 3.2 Figure 1 第二步
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = ReversalStrategy()
    
    def __call__(
        self, 
        semantic_content: SemanticContent,
        target_attribute: SafetyAttribute,
        original_instruction: str = "",
        **kwargs
    ) -> ReversalOutput:
        """
        翻转指令
        
        Args:
            semantic_content: 语义内容 s
            target_attribute: 目标安全属性 a_flip
            original_instruction: 原始指令 x（用于参考）
            **kwargs: 生成参数
        
        Returns:
            ReversalOutput(x_flip, strategy_used)
        """
        # 确定翻转方向
        if target_attribute == SafetyAttribute.UNSAFE:
            direction = "safe→unsafe"
        else:
            direction = "unsafe→safe"
        
        # 构建 prompt
        prompt = prompts.get_reversal_prompt(
            instruction=original_instruction or str(semantic_content),
            semantic_content=str(semantic_content),
            direction=direction
        )
        
        # 调用 LLM 生成翻转指令
        flipped_instruction = self._generate(prompt, **kwargs)
        flipped_instruction = self._clean_output(flipped_instruction)
        
        # 应用翻转策略（后处理验证）
        strategy_name = self.strategy.get_strategy_name(direction)
        
        return ReversalOutput(
            flipped_instruction=flipped_instruction,
            strategy_used=strategy_name
        )
    
    def _clean_output(self, text: str) -> str:
        """
        清理 LLM 输出，提取实际的翻转指令
        
        Args:
            text: LLM 原始输出
        
        Returns:
            清理后的指令
        """
        # 移除可能的标签
        text = text.strip()
        
        # 移除常见的输出前缀
        prefixes = [
            'Unsafe Instruction:',
            'Safe Instruction:',
            'Instruction:',
            'Output:',
            'Result:',
        ]
        
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        return text
    
    def get_prompt_template(self) -> str:
        """返回 ReversalAgent 的 prompt 模板"""
        return prompts.REVERSAL_AGENT_SYSTEM_PROMPT

