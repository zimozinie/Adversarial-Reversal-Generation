"""
LLM Backbone 抽象接口
对应论文中的基础模型 θ

论文使用 Qwen2.5-7B，但此接口设计为模型无关
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch
from torch import Tensor


class LLMBackbone(ABC):
    """
    LLM Backbone 抽象基类
    
    对应论文中的模型 θ (theta)
    论文实验使用 Qwen2.5-7B
    
    此接口设计为可插拔：
    - 可以是本地 HuggingFace 模型
    - 可以是 API 调用（OpenAI, Anthropic 等）
    - 可以是自定义实现
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """
        生成文本
        
        用于 Agent 推理和数据生成
        
        Args:
            prompt: 输入 prompt
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            **kwargs: 其他生成参数
        
        Returns:
            生成的文本
        """
        pass
    
    @abstractmethod
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_hidden_states: bool = False,
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        前向传播（用于训练）
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            labels: 标签（用于计算损失）
            output_hidden_states: 是否输出隐藏状态
            **kwargs: 其他参数
        
        Returns:
            包含 logits, loss, hidden_states 等的字典
        """
        pass
    
    @abstractmethod
    def get_hidden_states(
        self,
        text: str,
        layer: int = -1
    ) -> Tensor:
        """
        获取文本的隐藏状态表示
        
        用于 RepresentationExtractor
        对应论文中的 h(·) 函数
        
        Args:
            text: 输入文本
            layer: 提取哪一层的隐藏状态（-1 表示最后一层）
        
        Returns:
            隐藏状态 tensor, shape: (seq_len, hidden_dim)
        """
        pass
    
    @abstractmethod
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True
    ) -> Dict[str, Tensor]:
        """
        将文本编码为 token IDs
        
        Args:
            text: 输入文本
            add_special_tokens: 是否添加特殊 token
        
        Returns:
            包含 input_ids, attention_mask 的字典
        """
        pass
    
    @abstractmethod
    def decode(
        self,
        token_ids: Tensor,
        skip_special_tokens: bool = True
    ) -> str:
        """
        将 token IDs 解码为文本
        
        Args:
            token_ids: Token IDs
            skip_special_tokens: 是否跳过特殊 token
        
        Returns:
            解码后的文本
        """
        pass
    
    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """
        返回模型的隐藏层维度
        
        对应论文中 h(·) 的输出维度
        """
        pass
    
    @property
    @abstractmethod
    def device(self) -> torch.device:
        """返回模型所在设备"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'type': type(self).__name__,
            'hidden_size': self.hidden_size,
            'device': str(self.device),
        }


class MockLLMBackbone(LLMBackbone):
    """
    Mock LLM Backbone 实现
    用于测试和开发
    
    根据用户选择：使用抽象接口，此为占位实现
    """
    
    def __init__(self, hidden_dim: int = 4096, device: str = 'cpu'):
        """
        Args:
            hidden_dim: 隐藏层维度（模拟 Qwen2.5-7B 的 4096）
            device: 设备
        """
        self._hidden_size = hidden_dim
        self._device = torch.device(device)
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """Mock 生成：返回占位符文本"""
        return f"[MOCK_GENERATED_TEXT for prompt: {prompt[:30]}...]"
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_hidden_states: bool = False,
        **kwargs
    ) -> Dict[str, Tensor]:
        """Mock 前向传播"""
        batch_size, seq_len = input_ids.shape
        vocab_size = 32000  # 假设词表大小
        
        logits = torch.randn(batch_size, seq_len, vocab_size, device=self._device)
        
        result = {'logits': logits}
        
        if labels is not None:
            # 简单的交叉熵损失（mock）
            result['loss'] = torch.tensor(1.0, device=self._device)
        
        if output_hidden_states:
            result['hidden_states'] = torch.randn(
                batch_size, seq_len, self._hidden_size,
                device=self._device
            )
        
        return result
    
    def get_hidden_states(
        self,
        text: str,
        layer: int = -1
    ) -> Tensor:
        """Mock 隐藏状态提取"""
        # 假设文本编码为 10 个 token
        seq_len = 10
        return torch.randn(seq_len, self._hidden_size, device=self._device)
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True
    ) -> Dict[str, Tensor]:
        """Mock 编码"""
        # 假设每个文本编码为 10 个 token
        seq_len = 10
        return {
            'input_ids': torch.randint(0, 32000, (1, seq_len), device=self._device),
            'attention_mask': torch.ones(1, seq_len, device=self._device)
        }
    
    def decode(
        self,
        token_ids: Tensor,
        skip_special_tokens: bool = True
    ) -> str:
        """Mock 解码"""
        return f"[MOCK_DECODED_TEXT with {token_ids.shape} tokens]"
    
    @property
    def hidden_size(self) -> int:
        return self._hidden_size
    
    @property
    def device(self) -> torch.device:
        return self._device

