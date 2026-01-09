"""
Agent 抽象基类
对应论文 Section 3.2 Multi-Agent Pipeline
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from ..models.backbone import LLMBackbone


class BaseAgent(ABC):
    """
    Agent 抽象基类
    
    每个 Agent 必须：
    - 有清晰的输入/输出 schema
    - 不共享隐式状态
    - 不得合并为"一个 prompt + role"
    
    对应论文强制约束：
    "Multi-Agent 强制约束：每个 Agent 必须有清晰的输入/输出 dataclass，
    不共享隐式状态，不得合并为'一个 prompt + role'"
    """
    
    def __init__(self, llm: 'LLMBackbone', config: Dict[str, Any] = None):
        """
        Args:
            llm: LLM backend for agent reasoning
            config: Agent-specific configuration
        """
        self.llm = llm
        self.config = config or {}
    
    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """
        Agent 的主要执行方法
        每个具体 Agent 必须实现此方法
        """
        pass
    
    @abstractmethod
    def get_prompt_template(self) -> str:
        """
        获取 Agent 的 Prompt 模板
        每个 Agent 有独立的 Prompt 模板
        """
        pass
    
    def _generate(self, prompt: str, **kwargs) -> str:
        """
        统一的 LLM 生成接口
        
        Args:
            prompt: 格式化后的 prompt
            **kwargs: 生成参数
        
        Returns:
            LLM 生成的文本
        """
        temperature = kwargs.get('temperature', self.config.get('temperature', 0.7))
        max_tokens = kwargs.get('max_tokens', self.config.get('max_tokens', 2048))
        
        return self.llm.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

