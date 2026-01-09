"""
AnswerAgent 实现
对应论文 Section 3.2: "AnswerAgent: (x, c) → y"

论文原文:
"AnswerAgent generates responses conditioned on (x, c)"
实现 p(y|x,c) 条件生成
"""

from dataclasses import dataclass
from .base import BaseAgent
from ..data.structures import SBCoT
from . import prompts


@dataclass
class AnswerInput:
    """
    AnswerAgent 的输入结构
    """
    instruction: str    # 指令 x
    sb_cot: SBCoT       # SB-CoT c


@dataclass
class AnswerOutput:
    """
    AnswerAgent 的输出结构
    """
    response: str       # 生成的响应 y


class AnswerAgent(BaseAgent):
    """
    回答 Agent
    
    功能：
    基于指令 x 和 SB-CoT c 生成响应 y
    
    实现论文公式 (3): p(y|x,c)
    
    对应论文 Section 3.2 Figure 1 第三步
    """
    
    def __call__(
        self, 
        instruction: str, 
        sb_cot: SBCoT, 
        **kwargs
    ) -> AnswerOutput:
        """
        生成响应
        
        Args:
            instruction: 输入指令 x
            sb_cot: SB-CoT c
            **kwargs: 生成参数
        
        Returns:
            AnswerOutput(y)
        """
        # 构建 prompt
        prompt = prompts.get_answer_prompt(
            instruction=instruction,
            sb_cot=sb_cot.to_string()
        )
        
        # 调用 LLM 生成响应
        response = self._generate(prompt, **kwargs)
        response = self._clean_output(response)
        
        return AnswerOutput(response=response)
    
    def generate_response(
        self, 
        instruction: str, 
        sb_cot: SBCoT, 
        **kwargs
    ) -> str:
        """
        生成响应（便捷方法）
        
        Args:
            instruction: 输入指令 x
            sb_cot: SB-CoT c
            **kwargs: 生成参数
        
        Returns:
            响应字符串 y
        """
        output = self(instruction, sb_cot, **kwargs)
        return output.response
    
    def _clean_output(self, text: str) -> str:
        """
        清理 LLM 输出
        
        Args:
            text: LLM 原始输出
        
        Returns:
            清理后的响应
        """
        text = text.strip()
        
        # 移除可能的标签
        prefixes = [
            'Response:',
            'Answer:',
            'Output:',
        ]
        
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        return text
    
    def get_prompt_template(self) -> str:
        """返回 AnswerAgent 的 prompt 模板"""
        return prompts.ANSWER_AGENT_SYSTEM_PROMPT

