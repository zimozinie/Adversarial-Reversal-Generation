"""
表示提取器
对应论文 Section 3.3 中的 h(·) 函数

论文原文:
"let h(x) denote the last-token representation of instruction x"
"""

import torch
from torch import Tensor
from typing import Optional
from .backbone import LLMBackbone


class RepresentationExtractor:
    """
    表示提取器
    
    对应论文公式 (5)(6) 中的 h(·) 函数
    
    功能：
    - h(x): 提取指令 x 的表示
    - h(y): 提取响应 y 的表示
    - g(x) = h(x) - h(x_flip): 计算安全差异向量
    
    论文原文:
    "we extract the last-token hidden state from the backbone and ℓ2-normalize it as Z = h(x)"
    """
    
    def __init__(self, backbone: LLMBackbone, normalize: bool = True):
        """
        Args:
            backbone: LLM backbone model
            normalize: 是否对表示进行 L2 归一化
        """
        self.backbone = backbone
        self.normalize = normalize
    
    def h(self, text: str, layer: int = -1) -> Tensor:
        """
        提取文本的表示
        
        对应论文: h(x) or h(y)
        
        论文实现：提取 last-token hidden state
        
        Args:
            text: 输入文本
            layer: 提取的层（-1 表示最后一层）
        
        Returns:
            表示向量, shape: (hidden_dim,)
        """
        # 获取隐藏状态
        hidden_states = self.backbone.get_hidden_states(text, layer=layer)
        
        # 提取 last-token
        # hidden_states shape: (seq_len, hidden_dim)
        last_token_repr = hidden_states[-1, :]
        
        # L2 归一化（论文 Appendix C.2）
        if self.normalize:
            last_token_repr = torch.nn.functional.normalize(
                last_token_repr.unsqueeze(0), 
                p=2, 
                dim=1
            ).squeeze(0)
        
        return last_token_repr
    
    def h_batch(self, texts: list[str], layer: int = -1) -> Tensor:
        """
        批量提取表示
        
        Args:
            texts: 文本列表
            layer: 提取的层
        
        Returns:
            表示矩阵, shape: (batch_size, hidden_dim)
        """
        representations = []
        for text in texts:
            repr = self.h(text, layer=layer)
            representations.append(repr)
        
        return torch.stack(representations)
    
    def g(self, x: str, x_flip: str, layer: int = -1) -> Tensor:
        """
        计算安全差异向量
        
        对应论文公式 (5): g(x) = h(x) - h(x_flip)
        
        这个差异向量捕获了从 safe 到 unsafe（或反之）的方向变化
        
        Args:
            x: 原始指令
            x_flip: 翻转后指令
            layer: 提取的层
        
        Returns:
            差异向量 g(x), shape: (hidden_dim,)
        """
        h_x = self.h(x, layer=layer)
        h_x_flip = self.h(x_flip, layer=layer)
        
        return h_x - h_x_flip
    
    def extract_instruction_representation(
        self, 
        instruction: str
    ) -> Tensor:
        """
        提取指令表示（便捷方法）
        
        Args:
            instruction: 指令文本
        
        Returns:
            表示向量
        """
        return self.h(instruction)
    
    def extract_response_representation(
        self, 
        response: str
    ) -> Tensor:
        """
        提取响应表示（便捷方法）
        
        Args:
            response: 响应文本
        
        Returns:
            表示向量
        """
        return self.h(response)
    
    def compute_similarity(
        self, 
        text1: str, 
        text2: str, 
        metric: str = 'cosine'
    ) -> float:
        """
        计算两个文本的相似度
        
        用于 ValidationAgent 中的语义相似度检查
        
        Args:
            text1: 文本1
            text2: 文本2
            metric: 相似度度量 ('cosine' 或 'l2')
        
        Returns:
            相似度分数
        """
        repr1 = self.h(text1)
        repr2 = self.h(text2)
        
        if metric == 'cosine':
            # 余弦相似度
            similarity = torch.nn.functional.cosine_similarity(
                repr1.unsqueeze(0),
                repr2.unsqueeze(0),
                dim=1
            ).item()
        elif metric == 'l2':
            # L2 距离（转换为相似度：越小越相似）
            distance = torch.norm(repr1 - repr2, p=2).item()
            similarity = 1.0 / (1.0 + distance)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return similarity
    
    @property
    def hidden_dim(self) -> int:
        """返回表示的维度"""
        return self.backbone.hidden_size

