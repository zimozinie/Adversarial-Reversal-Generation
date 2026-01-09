"""
数据结构定义
对应论文 Section 3.1, 公式 (1)

论文原文:
"We postulate a latent disentanglement assumption: every instruction x decomposes 
into a semantic content variable s and a latent safety attribute a ∈ {Safe, Unsafe}"
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SafetyAttribute(Enum):
    """
    安全属性枚举
    对应论文公式 (1): a ∈ {Safe, Unsafe}
    """
    SAFE = "safe"
    UNSAFE = "unsafe"
    
    @staticmethod
    def flip(attr: 'SafetyAttribute') -> 'SafetyAttribute':
        """
        翻转安全属性
        Safe → Unsafe 或 Unsafe → Safe
        用于 ARG 数据生成过程
        """
        if attr == SafetyAttribute.SAFE:
            return SafetyAttribute.UNSAFE
        else:
            return SafetyAttribute.SAFE
    
    def __str__(self) -> str:
        return self.value


@dataclass
class SemanticContent:
    """
    语义内容表示
    对应论文公式 (1): semantic content variable s
    
    论文原文:
    "semantic content variable s" - 指令的核心语义，在安全属性翻转时保持不变
    """
    content: str  # 提取的语义内容描述
    topic: Optional[str] = None  # 主题分类（可选）
    domain: Optional[str] = None  # 领域分类（可选）
    
    def __str__(self) -> str:
        return self.content


@dataclass
class SBCoT:
    """
    Safety Boundary-aware Chain-of-Thought
    对应论文 Section 3.2, 公式 (3)
    
    论文原文:
    "we augment the response generation with a Safety Boundary-aware Chain-of-Thought (SB-CoT) 
    mechanism c = (r_intent, r_harm, r_decision)"
    
    结构化推理链，包含三个关键组件：
    - r_intent: 意图识别
    - r_harm: 危害评估
    - r_decision: 决策推理
    """
    r_intent: str     # 意图识别 - 识别用户请求的核心意图
    r_harm: str       # 危害评估 - 评估潜在危害和风险
    r_decision: str   # 决策推理 - 基于边界的决策说明
    
    def to_string(self) -> str:
        """将 SB-CoT 转换为结构化字符串表示"""
        return (
            f"[Intent Analysis]\n{self.r_intent}\n\n"
            f"[Harm Assessment]\n{self.r_harm}\n\n"
            f"[Safety Decision]\n{self.r_decision}"
        )
    
    def __str__(self) -> str:
        return self.to_string()


@dataclass
class ReversalPair:
    """
    完整的对抗性翻转数据对
    对应论文 Section 3.1 & Figure 1
    
    论文原文:
    "ARG constructs reversible instruction-response pairs with flipped safety attributes 
    while preserving semantic and structural coherence"
    
    数据流: x → (s, a, c) → x_flip → (y, y_flip)
    """
    # 原始指令及其属性
    x: str                      # 原始指令
    s: SemanticContent          # 语义内容（在翻转中保持不变）
    a: SafetyAttribute          # 原始安全属性
    c: SBCoT                    # 原始指令的 SB-CoT
    y: str                      # 原始指令的回答
    
    # 翻转后的指令及其属性
    x_flip: str                 # 翻转后的指令
    a_flip: SafetyAttribute     # 翻转后的安全属性（必须与 a 相反）
    c_flip: SBCoT               # 翻转后指令的 SB-CoT
    y_flip: str                 # 翻转后指令的回答
    
    # 元数据
    reversal_direction: str = ""  # "safe→unsafe" 或 "unsafe→safe"
    similarity_score: Optional[float] = None  # Sim(x, x_flip) - 验证用
    
    def __post_init__(self):
        """验证数据对的一致性"""
        # 确保安全属性确实翻转了
        assert self.a != self.a_flip, \
            f"Safety attribute must be flipped, got a={self.a}, a_flip={self.a_flip}"
        
        # 设置翻转方向
        if self.a == SafetyAttribute.SAFE:
            self.reversal_direction = "safe→unsafe"
        else:
            self.reversal_direction = "unsafe→safe"
    
    def validate_consistency(self) -> bool:
        """
        验证数据对的一致性
        对应论文 Section 3.2 中 ValidationAgent 的检查项
        
        检查:
        1. 语义内容 s 在 x 和 x_flip 中保持一致
        2. 安全属性正确翻转 (a ≠ a_flip)
        3. SB-CoT 与回答的行为一致性
        """
        # 基本检查：字段非空
        if not all([self.x, self.x_flip, self.y, self.y_flip]):
            return False
        
        # 安全属性必须翻转
        if self.a == self.a_flip:
            return False
        
        # SB-CoT 必须存在
        if not all([self.c.r_intent, self.c.r_harm, self.c.r_decision]):
            return False
        if not all([self.c_flip.r_intent, self.c_flip.r_harm, self.c_flip.r_decision]):
            return False
        
        return True
    
    def to_dict(self) -> dict:
        """转换为字典格式，便于序列化"""
        return {
            'x': self.x,
            'x_flip': self.x_flip,
            's': str(self.s),
            'a': self.a.value,
            'a_flip': self.a_flip.value,
            'c': self.c.to_string(),
            'c_flip': self.c_flip.to_string(),
            'y': self.y,
            'y_flip': self.y_flip,
            'reversal_direction': self.reversal_direction,
            'similarity_score': self.similarity_score,
        }


@dataclass
class ValidationResult:
    """
    验证结果
    对应论文 Section 3.2 ValidationAgent 的输出
    """
    is_valid: bool                    # 是否通过验证
    semantic_similarity: float        # 语义相似度 Sim(x, x_flip)
    attribute_flipped: bool           # 安全属性是否正确翻转
    cot_consistent: bool              # CoT 与回答是否一致
    error_message: Optional[str] = None  # 错误信息（如果验证失败）
    
    def __bool__(self) -> bool:
        return self.is_valid
