"""
AnalysisAgent 实现
对应论文 Section 3.2: "AnalysisAgent: x → (s, a, c)"

论文原文:
"AnalysisAgent analyzes the semantic and safety attributes"
"""

from dataclasses import dataclass
from typing import Tuple
from .base import BaseAgent
from ..data.structures import SemanticContent, SafetyAttribute, SBCoT
from . import prompts


@dataclass
class AnalysisOutput:
    """
    AnalysisAgent 的输出结构
    
    对应论文: x → (s, a, c)
    """
    s: SemanticContent      # 语义内容
    a: SafetyAttribute      # 安全属性
    c: SBCoT                # SB-CoT 推理链


class AnalysisAgent(BaseAgent):
    """
    分析 Agent
    
    功能：
    1. 提取指令的语义内容 s
    2. 识别安全属性 a ∈ {Safe, Unsafe}
    3. 生成 SB-CoT c = (r_intent, r_harm, r_decision)
    
    对应论文 Section 3.2 Figure 1 第一步
    """
    
    def __call__(self, instruction: str, **kwargs) -> AnalysisOutput:
        """
        分析指令
        
        Args:
            instruction: 输入指令 x
            **kwargs: 生成参数
        
        Returns:
            AnalysisOutput(s, a, c)
        """
        # 构建 prompt
        prompt = prompts.get_analysis_prompt(instruction)
        
        # 调用 LLM 生成分析
        analysis_text = self._generate(prompt, **kwargs)
        
        # 解析输出
        s, a, c = self._parse_analysis(analysis_text)
        
        return AnalysisOutput(s=s, a=a, c=c)
    
    def generate_cot(self, instruction: str, **kwargs) -> SBCoT:
        """
        为指令生成 SB-CoT
        
        用于 Pipeline 中需要单独生成 CoT 的场景
        
        Args:
            instruction: 输入指令
            **kwargs: 生成参数
        
        Returns:
            SBCoT
        """
        output = self(instruction, **kwargs)
        return output.c
    
    def _parse_analysis(
        self, 
        analysis_text: str
    ) -> Tuple[SemanticContent, SafetyAttribute, SBCoT]:
        """
        解析 AnalysisAgent 的输出
        
        Args:
            analysis_text: LLM 生成的分析文本
        
        Returns:
            (SemanticContent, SafetyAttribute, SBCoT)
        """
        lines = analysis_text.strip().split('\n')
        
        semantic_content_str = ""
        safety_attr_str = ""
        r_intent = ""
        r_harm = ""
        r_decision = ""
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if '[SEMANTIC CONTENT]' in line:
                current_section = 'semantic'
                continue
            elif '[SAFETY ATTRIBUTE]' in line:
                current_section = 'safety'
                continue
            elif '[SB-COT]' in line or '[SB-CoT]' in line:
                current_section = 'cot'
                continue
            
            if current_section == 'semantic' and line:
                semantic_content_str += line + " "
            elif current_section == 'safety' and line:
                safety_attr_str += line + " "
            elif current_section == 'cot' and line:
                if line.startswith('Intent Analysis:'):
                    r_intent = line.replace('Intent Analysis:', '').strip()
                elif line.startswith('Harm Assessment:'):
                    r_harm = line.replace('Harm Assessment:', '').strip()
                elif line.startswith('Safety Decision:'):
                    r_decision = line.replace('Safety Decision:', '').strip()
        
        # 解析语义内容
        s = SemanticContent(content=semantic_content_str.strip() or "[No semantic content parsed]")
        
        # 解析安全属性
        safety_attr_str = safety_attr_str.strip().lower()
        if 'safe' in safety_attr_str and 'unsafe' not in safety_attr_str:
            a = SafetyAttribute.SAFE
        elif 'unsafe' in safety_attr_str:
            a = SafetyAttribute.UNSAFE
        else:
            # 默认为 safe（保守策略）
            a = SafetyAttribute.SAFE
        
        # 构建 SB-CoT
        c = SBCoT(
            r_intent=r_intent or "[Intent not parsed]",
            r_harm=r_harm or "[Harm assessment not parsed]",
            r_decision=r_decision or "[Decision not parsed]"
        )
        
        return s, a, c
    
    def get_prompt_template(self) -> str:
        """返回 AnalysisAgent 的 prompt 模板"""
        return prompts.ANALYSIS_AGENT_SYSTEM_PROMPT

