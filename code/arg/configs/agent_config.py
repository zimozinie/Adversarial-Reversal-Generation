"""
Agent 配置
对应论文 Section 3.2 Multi-Agent Pipeline 的参数
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentConfig:
    """
    Agent 配置
    
    用于配置 Multi-Agent Pipeline 中各个 Agent 的行为
    """
    
    # ========== 通用 Agent 参数 ==========
    temperature: float = 0.7
    """LLM 生成温度"""
    
    max_tokens: int = 2048
    """最大生成 token 数"""
    
    # ========== AnalysisAgent 参数 ==========
    analysis_prompt_template: Optional[str] = None
    """自定义分析 prompt 模板"""
    
    # ========== ReversalAgent 参数 ==========
    reversal_prompt_template: Optional[str] = None
    """自定义翻转 prompt 模板"""
    
    reversal_strategy: str = "auto"
    """翻转策略: auto, random, specific"""
    
    # ========== AnswerAgent 参数 ==========
    answer_prompt_template: Optional[str] = None
    """自定义回答 prompt 模板"""
    
    # ========== ValidationAgent 参数 ==========
    validation_prompt_template: Optional[str] = None
    """自定义验证 prompt 模板"""
    
    similarity_threshold: float = 0.7
    """语义相似度阈值（Sim(x, x_flip) > threshold）"""
    
    validation_mode: str = "strict"
    """验证模式: strict, lenient"""
    
    # ========== Pipeline 参数 ==========
    max_retries: int = 3
    """最大重试次数（验证失败时）"""
    
    verbose: bool = True
    """是否打印详细日志"""
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'similarity_threshold': self.similarity_threshold,
            'validation_mode': self.validation_mode,
            'max_retries': self.max_retries,
            'verbose': self.verbose,
        }

