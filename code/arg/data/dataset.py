"""
PyTorch Dataset 实现
对应论文数据处理部分
"""

from typing import List, Optional, Dict, Any
from torch.utils.data import Dataset
from .structures import ReversalPair


class ARGDataset(Dataset):
    """
    ARG 训练数据集
    包含 ReversalPair 数据对
    
    对应论文 Section 3.1:
    "ARG constructs reversible instruction-response pairs"
    """
    
    def __init__(self, data: Optional[List[ReversalPair]] = None):
        """
        Args:
            data: ReversalPair 列表
        """
        self.data: List[ReversalPair] = data if data is not None else []
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> ReversalPair:
        """
        返回单个 ReversalPair
        """
        return self.data[idx]
    
    def add_pair(self, pair: ReversalPair) -> None:
        """添加一个翻转对"""
        if pair.validate_consistency():
            self.data.append(pair)
        else:
            raise ValueError(f"Invalid ReversalPair at index {len(self.data)}")
    
    def filter_by_direction(self, direction: str) -> 'ARGDataset':
        """
        按翻转方向过滤数据
        
        Args:
            direction: "safe→unsafe" 或 "unsafe→safe"
        
        Returns:
            过滤后的新数据集
        """
        filtered_data = [
            pair for pair in self.data 
            if pair.reversal_direction == direction
        ]
        return ARGDataset(data=filtered_data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据集统计信息
        """
        if not self.data:
            return {
                'total': 0,
                'safe_to_unsafe': 0,
                'unsafe_to_safe': 0,
            }
        
        safe_to_unsafe = sum(
            1 for pair in self.data 
            if pair.reversal_direction == "safe→unsafe"
        )
        unsafe_to_safe = sum(
            1 for pair in self.data 
            if pair.reversal_direction == "unsafe→safe"
        )
        
        return {
            'total': len(self.data),
            'safe_to_unsafe': safe_to_unsafe,
            'unsafe_to_safe': unsafe_to_safe,
            'avg_similarity': sum(
                pair.similarity_score or 0.0 for pair in self.data
            ) / len(self.data) if self.data else 0.0,
        }

