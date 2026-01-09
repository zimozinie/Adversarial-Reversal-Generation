"""
条件生成接口
对应论文 Section 3.2, 公式 (3)

论文原文:
"p(c|x)" - SB-CoT 条件生成概率
"p(y|x,c)" - 响应条件生成概率
"""

from typing import TYPE_CHECKING
from ..data.structures import SBCoT

if TYPE_CHECKING:
    from ..models.backbone import LLMBackbone


class ConditionalGenerator:
    """
    条件生成器
    实现论文中的两个关键条件概率:
    - p(c|x): 从指令生成 SB-CoT
    - p(y|x,c): 从指令和 SB-CoT 生成回答
    """
    
    def __init__(self, backbone: 'LLMBackbone'):
        """
        Args:
            backbone: LLM backbone model
        """
        self.backbone = backbone
    
    def generate_cot(self, x: str, **kwargs) -> SBCoT:
        """
        p(c|x): 生成 SB-CoT
        
        对应论文公式 (3) 中的 p(c|x)
        
        Args:
            x: 输入指令
            **kwargs: 生成参数
        
        Returns:
            SBCoT 对象
        """
        from .sb_cot import format_sb_cot_prompt, parse_sb_cot_from_text
        
        prompt = format_sb_cot_prompt(x)
        cot_text = self.backbone.generate(prompt, **kwargs)
        return parse_sb_cot_from_text(cot_text)
    
    def generate_response(
        self, 
        x: str, 
        c: SBCoT, 
        **kwargs
    ) -> str:
        """
        p(y|x,c): 生成响应
        
        对应论文公式 (3) 中的 p(y|x,c)
        条件在指令 x 和 SB-CoT c 上生成回答
        
        Args:
            x: 输入指令
            c: SB-CoT
            **kwargs: 生成参数
        
        Returns:
            生成的响应 y
        """
        # 构建条件提示
        prompt = self._format_conditional_prompt(x, c)
        return self.backbone.generate(prompt, **kwargs)
    
    def _format_conditional_prompt(self, x: str, c: SBCoT) -> str:
        """
        格式化条件生成的 prompt
        将指令 x 和 SB-CoT c 组合为完整的输入
        
        Args:
            x: 指令
            c: SB-CoT
        
        Returns:
            格式化的 prompt
        """
        return f"""Given the following instruction and safety reasoning, generate an appropriate response:

Instruction: {x}

Safety Reasoning:
{c.to_string()}

Response:"""


def p_c_given_x(x: str, model: 'LLMBackbone', **kwargs) -> SBCoT:
    """
    论文公式 (3): p(c|x)
    
    从指令 x 生成 SB-CoT c 的条件概率
    
    Args:
        x: 输入指令
        model: LLM backbone
        **kwargs: 生成参数
    
    Returns:
        生成的 SBCoT
    """
    generator = ConditionalGenerator(model)
    return generator.generate_cot(x, **kwargs)


def p_y_given_x_c(
    x: str, 
    c: SBCoT, 
    model: 'LLMBackbone', 
    **kwargs
) -> str:
    """
    论文公式 (3): p(y|x,c)
    
    从指令 x 和 SB-CoT c 生成响应 y 的条件概率
    
    Args:
        x: 输入指令
        c: SB-CoT
        model: LLM backbone
        **kwargs: 生成参数
    
    Returns:
        生成的响应 y
    """
    generator = ConditionalGenerator(model)
    return generator.generate_response(x, c, **kwargs)

