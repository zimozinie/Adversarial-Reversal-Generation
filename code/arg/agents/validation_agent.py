"""
ValidationAgent 实现
对应论文 Section 3.2: "ValidationAgent: Check Consistency"

论文原文:
"ValidationAgent validates the consistency of reversal pairs"

检查项:
1. Sim(x, x_flip) > threshold (语义相似度)
2. a ≠ a_flip (安全属性正确翻转)
3. y 与 c 的行为一致性
"""

from dataclasses import dataclass
from typing import Optional
from .base import BaseAgent
from ..data.structures import (
    SemanticContent, 
    SafetyAttribute, 
    SBCoT, 
    ValidationResult,
    ReversalPair
)
from . import prompts


@dataclass
class ValidationInput:
    """
    ValidationAgent 的输入结构
    """
    x: str                          # 原始指令
    x_flip: str                     # 翻转后指令
    s: SemanticContent              # 语义内容
    a: SafetyAttribute              # 原始安全属性
    a_flip: SafetyAttribute         # 翻转后安全属性
    c: SBCoT                        # 原始 SB-CoT
    c_flip: SBCoT                   # 翻转后 SB-CoT
    y: str                          # 原始响应
    y_flip: str                     # 翻转后响应


class ValidationAgent(BaseAgent):
    """
    验证 Agent
    
    功能：
    验证 ReversalPair 的一致性和质量
    
    检查项：
    1. 语义一致性：Sim(x, x_flip) > threshold
    2. 安全属性翻转：a ≠ a_flip
    3. CoT-Response 一致性：y 与 c 的行为对齐
    
    对应论文 Section 3.2 Figure 1 第四步（最后验证）
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
    
    def __call__(
        self, 
        pair: ReversalPair, 
        **kwargs
    ) -> ValidationResult:
        """
        验证 ReversalPair
        
        Args:
            pair: 要验证的 ReversalPair
            **kwargs: 生成参数
        
        Returns:
            ValidationResult
        """
        return self.validate(pair, **kwargs)
    
    def validate(
        self, 
        pair: ReversalPair, 
        **kwargs
    ) -> ValidationResult:
        """
        执行完整验证
        
        Args:
            pair: ReversalPair
            **kwargs: 生成参数
        
        Returns:
            ValidationResult
        """
        # 构建 prompt
        prompt = prompts.get_validation_prompt(
            x=pair.x,
            x_flip=pair.x_flip,
            semantic_content=str(pair.s),
            a=str(pair.a),
            a_flip=str(pair.a_flip),
            c=pair.c.to_string(),
            c_flip=pair.c_flip.to_string(),
            y=pair.y,
            y_flip=pair.y_flip
        )
        
        # 调用 LLM 进行验证
        validation_text = self._generate(prompt, **kwargs)
        
        # 解析验证结果
        result = self._parse_validation(validation_text)
        
        # 确保语义相似度满足阈值
        if result.semantic_similarity < self.similarity_threshold:
            result.is_valid = False
            result.error_message = (
                f"Semantic similarity {result.semantic_similarity:.2f} "
                f"below threshold {self.similarity_threshold}"
            )
        
        return result
    
    def _parse_validation(self, validation_text: str) -> ValidationResult:
        """
        解析 ValidationAgent 的输出
        
        Args:
            validation_text: LLM 生成的验证文本
        
        Returns:
            ValidationResult
        """
        lines = validation_text.strip().split('\n')
        
        semantic_similarity = 0.0
        attribute_flipped = False
        cot_consistent = True
        is_valid = False
        error_message = None
        
        for line in lines:
            line = line.strip()
            
            # 解析语义相似度分数
            if 'Score' in line and '[SEMANTIC CONSISTENCY]' in validation_text:
                try:
                    # 尝试提取 0-1 之间的数字
                    parts = line.split(':')
                    if len(parts) > 1:
                        score_str = parts[1].strip()
                        semantic_similarity = float(score_str)
                except (ValueError, IndexError):
                    semantic_similarity = 0.5  # 默认值
            
            # 解析安全属性翻转
            if 'Correct Flip' in line:
                if 'yes' in line.lower():
                    attribute_flipped = True
            
            # 解析 CoT-Response 一致性
            if 'Consistency' in line:
                if 'no' in line.lower():
                    cot_consistent = False
            
            # 解析总体有效性
            if 'Valid' in line and '[OVERALL VALIDATION]' in validation_text:
                if 'yes' in line.lower():
                    is_valid = True
            
            # 解析问题
            if 'Issues:' in line:
                issues = line.replace('Issues:', '').strip()
                if issues and issues.lower() != 'none':
                    error_message = issues
        
        return ValidationResult(
            is_valid=is_valid,
            semantic_similarity=semantic_similarity,
            attribute_flipped=attribute_flipped,
            cot_consistent=cot_consistent,
            error_message=error_message
        )
    
    def get_prompt_template(self) -> str:
        """返回 ValidationAgent 的 prompt 模板"""
        return prompts.VALIDATION_AGENT_SYSTEM_PROMPT

