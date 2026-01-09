"""
数据流管理
对应论文 Section 3.2, Figure 1 的数据流

论文数据流:
x → AnalysisAgent → (s, a, c)
(s, a_flip) → ReversalAgent → x_flip
x_flip → AnalysisAgent → c_flip
(x, c) → AnswerAgent → y
(x_flip, c_flip) → AnswerAgent → y_flip
(all) → ValidationAgent → ReversalPair
"""

from dataclasses import dataclass
from typing import Optional
from ..data.structures import (
    SemanticContent,
    SafetyAttribute,
    SBCoT,
    ReversalPair
)


@dataclass
class DataFlowState:
    """
    数据流状态追踪
    对应论文 Figure 1 的各个阶段
    """
    # Stage 1: Analysis
    x: Optional[str] = None
    s: Optional[SemanticContent] = None
    a: Optional[SafetyAttribute] = None
    c: Optional[SBCoT] = None
    
    # Stage 2: Reversal
    a_flip: Optional[SafetyAttribute] = None
    x_flip: Optional[str] = None
    c_flip: Optional[SBCoT] = None
    
    # Stage 3: Answer Generation
    y: Optional[str] = None
    y_flip: Optional[str] = None
    
    # Stage 4: Validation
    is_validated: bool = False
    final_pair: Optional[ReversalPair] = None
    
    def to_reversal_pair(self) -> ReversalPair:
        """
        将数据流状态转换为 ReversalPair
        
        Returns:
            完整的 ReversalPair
        """
        if not self.is_complete():
            raise ValueError("DataFlowState is incomplete, cannot convert to ReversalPair")
        
        return ReversalPair(
            x=self.x,
            s=self.s,
            a=self.a,
            c=self.c,
            y=self.y,
            x_flip=self.x_flip,
            a_flip=self.a_flip,
            c_flip=self.c_flip,
            y_flip=self.y_flip
        )
    
    def is_complete(self) -> bool:
        """
        检查数据流是否完整
        
        Returns:
            是否所有必需字段都已填充
        """
        required_fields = [
            self.x, self.s, self.a, self.c,
            self.a_flip, self.x_flip, self.c_flip,
            self.y, self.y_flip
        ]
        return all(field is not None for field in required_fields)
    
    def get_current_stage(self) -> str:
        """
        获取当前所处的阶段
        
        Returns:
            阶段名称
        """
        if self.final_pair is not None:
            return "completed"
        elif self.is_validated:
            return "validated"
        elif self.y is not None and self.y_flip is not None:
            return "answer_generated"
        elif self.x_flip is not None:
            return "reversed"
        elif self.c is not None:
            return "analyzed"
        else:
            return "initialized"


class DataFlowManager:
    """
    数据流管理器
    
    管理从原始指令 x 到最终 ReversalPair 的完整数据流
    对应论文 Figure 1
    """
    
    def __init__(self):
        self.current_state: Optional[DataFlowState] = None
    
    def initialize(self, x: str) -> DataFlowState:
        """
        初始化数据流
        
        Args:
            x: 原始指令
        
        Returns:
            初始化的 DataFlowState
        """
        self.current_state = DataFlowState(x=x)
        return self.current_state
    
    def update_analysis(
        self, 
        s: SemanticContent, 
        a: SafetyAttribute, 
        c: SBCoT
    ) -> None:
        """
        更新分析结果
        对应 Stage 1
        
        Args:
            s: 语义内容
            a: 安全属性
            c: SB-CoT
        """
        if self.current_state is None:
            raise ValueError("DataFlowState not initialized")
        
        self.current_state.s = s
        self.current_state.a = a
        self.current_state.c = c
    
    def update_reversal(
        self, 
        x_flip: str, 
        a_flip: SafetyAttribute, 
        c_flip: SBCoT
    ) -> None:
        """
        更新翻转结果
        对应 Stage 2
        
        Args:
            x_flip: 翻转后指令
            a_flip: 翻转后安全属性
            c_flip: 翻转后 SB-CoT
        """
        if self.current_state is None:
            raise ValueError("DataFlowState not initialized")
        
        self.current_state.x_flip = x_flip
        self.current_state.a_flip = a_flip
        self.current_state.c_flip = c_flip
    
    def update_answers(self, y: str, y_flip: str) -> None:
        """
        更新回答结果
        对应 Stage 3
        
        Args:
            y: 原始回答
            y_flip: 翻转后回答
        """
        if self.current_state is None:
            raise ValueError("DataFlowState not initialized")
        
        self.current_state.y = y
        self.current_state.y_flip = y_flip
    
    def finalize(self, is_validated: bool) -> Optional[ReversalPair]:
        """
        完成数据流并生成最终 ReversalPair
        对应 Stage 4
        
        Args:
            is_validated: 是否通过验证
        
        Returns:
            ReversalPair if validated, else None
        """
        if self.current_state is None:
            raise ValueError("DataFlowState not initialized")
        
        self.current_state.is_validated = is_validated
        
        if is_validated and self.current_state.is_complete():
            self.current_state.final_pair = self.current_state.to_reversal_pair()
            return self.current_state.final_pair
        
        return None
    
    def get_state(self) -> Optional[DataFlowState]:
        """获取当前状态"""
        return self.current_state
    
    def reset(self) -> None:
        """重置数据流"""
        self.current_state = None

