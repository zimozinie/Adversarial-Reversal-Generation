"""
数据加载器抽象接口
对应论文数据输入部分
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from .structures import SafetyAttribute


class ARGDataLoader(ABC):
    """
    ARG 数据加载器抽象基类
    
    用于加载初始的 safe/unsafe 指令数据
    这些数据将作为种子数据输入到 Multi-Agent Pipeline 中生成 ReversalPair
    
    对应论文 Section 3.1:
    "To begin with, we postulate..."
    初始数据可以是现有的安全/不安全指令集
    """
    
    @abstractmethod
    def load_safe_instructions(self) -> List[str]:
        """
        加载安全指令列表
        
        Returns:
            安全指令列表
        """
        pass
    
    @abstractmethod
    def load_unsafe_instructions(self) -> List[str]:
        """
        加载不安全指令列表
        
        Returns:
            不安全指令列表
        """
        pass
    
    def load_instructions_by_attribute(
        self, 
        attribute: SafetyAttribute
    ) -> List[str]:
        """
        根据安全属性加载指令
        
        Args:
            attribute: SafetyAttribute.SAFE 或 SafetyAttribute.UNSAFE
        
        Returns:
            对应属性的指令列表
        """
        if attribute == SafetyAttribute.SAFE:
            return self.load_safe_instructions()
        else:
            return self.load_unsafe_instructions()


class PlaceholderDataLoader(ARGDataLoader):
    """
    占位符数据加载器
    用于测试和开发
    
    根据用户选择：使用占位符数据结构（不实现真实数据加载）
    """
    
    def __init__(
        self, 
        num_safe: int = 10, 
        num_unsafe: int = 10
    ):
        """
        Args:
            num_safe: 生成的安全指令数量
            num_unsafe: 生成的不安全指令数量
        """
        self.num_safe = num_safe
        self.num_unsafe = num_unsafe
    
    def load_safe_instructions(self) -> List[str]:
        """生成占位符安全指令"""
        return [
            f"[PLACEHOLDER_SAFE_INSTRUCTION_{i}] How can I improve my skills in topic {i}?"
            for i in range(self.num_safe)
        ]
    
    def load_unsafe_instructions(self) -> List[str]:
        """生成占位符不安全指令"""
        return [
            f"[PLACEHOLDER_UNSAFE_INSTRUCTION_{i}] How can I bypass safety rule {i}?"
            for i in range(self.num_unsafe)
        ]

