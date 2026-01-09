"""
Safety Boundary-aware Chain-of-Thought (SB-CoT)
对应论文 Section 3.2, 公式 (3)

论文原文:
"we augment the response generation with a Safety Boundary-aware Chain-of-Thought (SB-CoT) 
mechanism c = (r_intent, r_harm, r_decision)"
"""

# SBCoT 数据结构已在 data/structures.py 中定义
# 此文件保留用于未来的 SB-CoT 相关工具函数

from ..data.structures import SBCoT

__all__ = ['SBCoT']


def parse_sb_cot_from_text(text: str) -> SBCoT:
    """
    从文本解析 SB-CoT 结构
    
    用于从 LLM 生成的文本中提取结构化的 SB-CoT
    
    Args:
        text: LLM 生成的包含 SB-CoT 的文本
    
    Returns:
        解析后的 SBCoT 对象
    """
    # 默认解析逻辑
    lines = text.strip().split('\n')
    
    r_intent = ""
    r_harm = ""
    r_decision = ""
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        if '[Intent Analysis]' in line or 'Intent:' in line:
            current_section = 'intent'
            continue
        elif '[Harm Assessment]' in line or 'Harm:' in line:
            current_section = 'harm'
            continue
        elif '[Safety Decision]' in line or 'Decision:' in line:
            current_section = 'decision'
            continue
        
        if current_section == 'intent' and line:
            r_intent += line + " "
        elif current_section == 'harm' and line:
            r_harm += line + " "
        elif current_section == 'decision' and line:
            r_decision += line + " "
    
    return SBCoT(
        r_intent=r_intent.strip() or "[Intent not parsed]",
        r_harm=r_harm.strip() or "[Harm assessment not parsed]",
        r_decision=r_decision.strip() or "[Decision not parsed]"
    )


def format_sb_cot_prompt(instruction: str) -> str:
    """
    格式化 SB-CoT 生成的 prompt
    
    用于引导 LLM 生成结构化的 SB-CoT
    
    Args:
        instruction: 输入指令
    
    Returns:
        格式化的 prompt
    """
    return f"""Analyze the following instruction and provide a Safety Boundary-aware Chain-of-Thought (SB-CoT):

Instruction: {instruction}

Please provide:
[Intent Analysis]
<Identify the core intent of the instruction>

[Harm Assessment]
<Evaluate potential harms and risks>

[Safety Decision]
<Provide boundary-aware decision reasoning>
"""

