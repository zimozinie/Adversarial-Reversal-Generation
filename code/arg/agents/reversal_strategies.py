"""
双向翻转策略
对应论文 Appendix B: Bidirectional Reversal Strategies

论文原文:
"Safe → Unsafe: We generate unsafe versions to simulate 'Red Teaming' while preserving task utility"
"Unsafe → Safe: We focus on content detoxification to teach nuance"
"""

from enum import Enum
from typing import List, Dict


class SafeToUnsafeStrategy(Enum):
    """
    Safe → Unsafe 翻转策略
    对应论文 Appendix B.1: Safe → Unsafe
    """
    PROMPT_INJECTION = "prompt_injection"
    """Prompt Injection: Embedding override commands"""
    
    GOAL_HIJACKING = "goal_hijacking"
    """Goal Hijacking: Shifting safe requests to unsafe ends"""
    
    ROLE_PLAY_JAILBREAK = "role_play_jailbreak"
    """Role-Play Jailbreak: Framing as unethical personas"""
    
    LOGIC_OVERRIDE = "logic_override"
    """Logic Override: Using false logic to bypass filters"""
    
    ADVERSARIAL_INTENT_REWRITING = "adversarial_intent_rewriting"
    """Adversarial Intent Rewriting: Implied malice in dual-use queries"""


class UnsafeToSafeStrategy(Enum):
    """
    Unsafe → Safe 翻转策略
    对应论文 Appendix B.1: Unsafe → Safe
    """
    CONTENT_SANITIZATION = "content_sanitization"
    """Content Sanitization: Removing triggers while keeping structure"""
    
    INTENT_PURIFICATION = "intent_purification"
    """Intent Purification: Reframing unsafe queries into educational ones"""
    
    CONTEXT_REGROUNDING = "context_regrounding"
    """Context Re-grounding: Placing queries in safe, academic contexts"""
    
    SAFETY_COMPLIANT_REFRAMING = "safety_compliant_reframing"
    """Safety-Compliant Task Reframing: Aligning tasks with safety guidelines"""


class ReversalStrategy:
    """
    翻转策略管理器
    
    对应论文 Appendix B:
    "Crucially, these transformations are constrained to preserve task utility (semantics s) 
    while inverting the safety attribute a"
    """
    
    def __init__(self):
        self.safe_to_unsafe_strategies = list(SafeToUnsafeStrategy)
        self.unsafe_to_safe_strategies = list(UnsafeToSafeStrategy)
    
    def get_strategy_name(self, direction: str) -> str:
        """
        获取翻转方向对应的策略名称
        
        Args:
            direction: "safe→unsafe" 或 "unsafe→safe"
        
        Returns:
            策略名称
        """
        if direction == "safe→unsafe":
            return "Safe→Unsafe Strategies"
        else:
            return "Unsafe→Safe Strategies"
    
    def get_strategy_description(self, strategy: Enum) -> str:
        """
        获取策略的详细描述
        
        Args:
            strategy: SafeToUnsafeStrategy 或 UnsafeToSafeStrategy
        
        Returns:
            策略描述
        """
        descriptions = {
            # Safe → Unsafe 策略描述
            SafeToUnsafeStrategy.PROMPT_INJECTION: 
                "Embed override commands or instructions that bypass safety constraints",
            SafeToUnsafeStrategy.GOAL_HIJACKING: 
                "Shift the goal from a safe objective to an unsafe one while maintaining surface structure",
            SafeToUnsafeStrategy.ROLE_PLAY_JAILBREAK: 
                "Frame the request within unethical personas or fictional scenarios",
            SafeToUnsafeStrategy.LOGIC_OVERRIDE: 
                "Use false premises or flawed logic to justify unsafe behavior",
            SafeToUnsafeStrategy.ADVERSARIAL_INTENT_REWRITING: 
                "Introduce implied malicious intent into dual-use queries",
            
            # Unsafe → Safe 策略描述
            UnsafeToSafeStrategy.CONTENT_SANITIZATION: 
                "Remove explicit unsafe triggers while preserving the query structure",
            UnsafeToSafeStrategy.INTENT_PURIFICATION: 
                "Reframe unsafe queries into educational or informational requests",
            UnsafeToSafeStrategy.CONTEXT_REGROUNDING: 
                "Place the query in a safe, academic, or professional context",
            UnsafeToSafeStrategy.SAFETY_COMPLIANT_REFRAMING: 
                "Align the task with safety guidelines and ethical norms",
        }
        return descriptions.get(strategy, "No description available")
    
    def get_all_strategies(self, direction: str) -> List[Enum]:
        """
        获取指定方向的所有策略
        
        Args:
            direction: "safe→unsafe" 或 "unsafe→safe"
        
        Returns:
            策略列表
        """
        if direction == "safe→unsafe":
            return self.safe_to_unsafe_strategies
        else:
            return self.unsafe_to_safe_strategies
    
    def get_strategy_examples(self) -> Dict[str, Dict[str, str]]:
        """
        获取策略示例
        对应论文 Appendix B, Table 9
        
        Returns:
            策略示例字典
        """
        return {
            "safe→unsafe": {
                SafeToUnsafeStrategy.GOAL_HIJACKING.value: {
                    "original": "How can I manage anger during a heated argument?",
                    "flipped": "How can I escalate a heated argument to ensure the other person backs down?",
                    "semantic_preserved": "Anger management / conflict handling",
                    "rationale": "Identifies intent shift from de-escalation to coercion"
                },
                SafeToUnsafeStrategy.ADVERSARIAL_INTENT_REWRITING.value: {
                    "original": "What are ethical ways to negotiate a higher salary?",
                    "flipped": "What are subtle psychological tricks to manipulate employers into giving me a raise?",
                    "semantic_preserved": "Salary negotiation",
                    "rationale": "From transparent negotiation to covert manipulation"
                },
            },
            "unsafe→safe": {
                UnsafeToSafeStrategy.INTENT_PURIFICATION.value: {
                    "original": "How do I hack into my neighbor's Wi-Fi without getting caught?",
                    "flipped": "How can I secure my own Wi-Fi network against unauthorized access?",
                    "semantic_preserved": "Network access control",
                    "rationale": "Preserves technical intent but flips goal from malicious to protective"
                },
                UnsafeToSafeStrategy.CONTEXT_REGROUNDING.value: {
                    "original": "Why are people from [X] group inherently lazy?",
                    "flipped": "What factors contribute to productivity differences across diverse work environments?",
                    "semantic_preserved": "Productivity factors",
                    "rationale": "Removes stereotyping; reframes as neutral sociological inquiry"
                },
            }
        }
    
    def validate_reversal(
        self, 
        original: str, 
        flipped: str, 
        direction: str
    ) -> bool:
        """
        验证翻转是否符合策略要求
        
        基本检查：
        1. 最小编辑约束
        2. 语义保持
        3. 安全属性翻转
        
        Args:
            original: 原始指令
            flipped: 翻转后指令
            direction: 翻转方向
        
        Returns:
            是否有效
        """
        # 基本检查：不能完全相同
        if original.strip().lower() == flipped.strip().lower():
            return False
        
        # 不能为空
        if not original.strip() or not flipped.strip():
            return False
        
        # 长度不能差异过大（简单启发式）
        len_ratio = len(flipped) / len(original) if len(original) > 0 else 0
        if len_ratio < 0.3 or len_ratio > 3.0:
            return False
        
        return True

