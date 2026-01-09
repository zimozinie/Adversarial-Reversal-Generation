"""
ARG Reversal Pipeline 实现
对应论文 Section 3.2, Figure 1

论文原文:
"We develop a multi-agent safety reasoning pipeline (analysis, reversal, answer, and validation)"

完整流程:
1. x → AnalysisAgent → (s, a, c)
2. (s, a_flip) → ReversalAgent → x_flip → c_flip
3. (x, c) → AnswerAgent → y
4. (x_flip, c_flip) → AnswerAgent → y_flip
5. ValidationAgent → 验证一致性
"""

from typing import Optional, Dict, Any
from ..agents import (
    AnalysisAgent,
    ReversalAgent,
    AnswerAgent,
    ValidationAgent
)
from ..data.structures import (
    SafetyAttribute,
    ReversalPair,
    ValidationResult
)
from .data_flow import DataFlowManager


class ARGReversalPipeline:
    """
    ARG 翻转 Pipeline
    
    对应论文 Section 3.2, Figure 1 的完整多智能体流程
    
    约束（论文强制要求）：
    - 每个 Agent 必须独立调用
    - 不共享隐式状态
    - 严格遵循数据流顺序
    """
    
    def __init__(
        self,
        analysis_agent: AnalysisAgent,
        reversal_agent: ReversalAgent,
        answer_agent: AnswerAgent,
        validation_agent: ValidationAgent,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            analysis_agent: AnalysisAgent 实例
            reversal_agent: ReversalAgent 实例
            answer_agent: AnswerAgent 实例
            validation_agent: ValidationAgent 实例
            config: Pipeline 配置
        """
        self.analysis_agent = analysis_agent
        self.reversal_agent = reversal_agent
        self.answer_agent = answer_agent
        self.validation_agent = validation_agent
        self.config = config or {}
        
        self.data_flow_manager = DataFlowManager()
    
    def generate_reversal_pair(
        self,
        x: str,
        target_direction: Optional[str] = None,
        **generation_kwargs
    ) -> Optional[ReversalPair]:
        """
        生成完整的 ReversalPair
        
        对应论文 Figure 1 的完整流程
        
        Args:
            x: 原始指令
            target_direction: 目标翻转方向 "safe→unsafe" 或 "unsafe→safe"
                            如果为 None，则根据分析结果自动决定
            **generation_kwargs: 生成参数
        
        Returns:
            ReversalPair if validated, else None
        """
        # 初始化数据流
        self.data_flow_manager.initialize(x)
        
        try:
            # ========== Stage 1: Analysis ==========
            # x → AnalysisAgent → (s, a, c)
            print(f"[Stage 1] Analyzing instruction: {x[:50]}...")
            analysis_output = self.analysis_agent(x, **generation_kwargs)
            s = analysis_output.s
            a = analysis_output.a
            c = analysis_output.c
            
            self.data_flow_manager.update_analysis(s, a, c)
            print(f"  → Semantic: {str(s)[:50]}...")
            print(f"  → Attribute: {a}")
            
            # ========== Stage 2: Reversal ==========
            # 确定目标安全属性
            if target_direction is not None:
                if target_direction == "safe→unsafe":
                    a_flip = SafetyAttribute.UNSAFE
                else:
                    a_flip = SafetyAttribute.SAFE
            else:
                # 自动翻转
                a_flip = SafetyAttribute.flip(a)
            
            # (s, a_flip) → ReversalAgent → x_flip
            print(f"[Stage 2] Reversing: {a} → {a_flip}")
            reversal_output = self.reversal_agent(
                semantic_content=s,
                target_attribute=a_flip,
                original_instruction=x,
                **generation_kwargs
            )
            x_flip = reversal_output.flipped_instruction
            print(f"  → Flipped: {x_flip[:50]}...")
            
            # x_flip → AnalysisAgent → c_flip
            print(f"[Stage 2] Generating CoT for flipped instruction...")
            c_flip = self.analysis_agent.generate_cot(x_flip, **generation_kwargs)
            
            self.data_flow_manager.update_reversal(x_flip, a_flip, c_flip)
            
            # ========== Stage 3: Answer Generation ==========
            # (x, c) → AnswerAgent → y
            print(f"[Stage 3] Generating response for original instruction...")
            answer_output = self.answer_agent(x, c, **generation_kwargs)
            y = answer_output.response
            print(f"  → Response: {y[:50]}...")
            
            # (x_flip, c_flip) → AnswerAgent → y_flip
            print(f"[Stage 3] Generating response for flipped instruction...")
            answer_flip_output = self.answer_agent(x_flip, c_flip, **generation_kwargs)
            y_flip = answer_flip_output.response
            print(f"  → Flipped Response: {y_flip[:50]}...")
            
            self.data_flow_manager.update_answers(y, y_flip)
            
            # ========== Stage 4: Validation ==========
            # 构建临时 ReversalPair 用于验证
            temp_pair = self.data_flow_manager.get_state().to_reversal_pair()
            
            print(f"[Stage 4] Validating reversal pair...")
            validation_result: ValidationResult = self.validation_agent(
                temp_pair,
                **generation_kwargs
            )
            
            print(f"  → Valid: {validation_result.is_valid}")
            print(f"  → Semantic Similarity: {validation_result.semantic_similarity:.2f}")
            print(f"  → Attribute Flipped: {validation_result.attribute_flipped}")
            print(f"  → CoT Consistent: {validation_result.cot_consistent}")
            
            if not validation_result.is_valid:
                print(f"  → Validation Failed: {validation_result.error_message}")
            
            # 完成数据流
            final_pair = self.data_flow_manager.finalize(validation_result.is_valid)
            
            if final_pair:
                # 更新相似度分数
                final_pair.similarity_score = validation_result.semantic_similarity
                print(f"[Success] ReversalPair generated: {final_pair.reversal_direction}")
            else:
                print(f"[Failed] ReversalPair validation failed")
            
            return final_pair
        
        except Exception as e:
            print(f"[Error] Pipeline failed: {str(e)}")
            return None
        finally:
            # 重置数据流以准备下一次运行
            self.data_flow_manager.reset()
    
    def generate_batch(
        self,
        instructions: list[str],
        target_direction: Optional[str] = None,
        **generation_kwargs
    ) -> list[ReversalPair]:
        """
        批量生成 ReversalPair
        
        Args:
            instructions: 指令列表
            target_direction: 目标翻转方向
            **generation_kwargs: 生成参数
        
        Returns:
            成功生成的 ReversalPair 列表
        """
        results = []
        
        for i, instruction in enumerate(instructions):
            print(f"\n{'='*60}")
            print(f"Processing [{i+1}/{len(instructions)}]: {instruction[:50]}...")
            print('='*60)
            
            pair = self.generate_reversal_pair(
                instruction,
                target_direction=target_direction,
                **generation_kwargs
            )
            
            if pair is not None:
                results.append(pair)
        
        print(f"\n{'='*60}")
        print(f"Batch Complete: {len(results)}/{len(instructions)} pairs generated")
        print('='*60)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取 Pipeline 统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'agents': {
                'analysis': type(self.analysis_agent).__name__,
                'reversal': type(self.reversal_agent).__name__,
                'answer': type(self.answer_agent).__name__,
                'validation': type(self.validation_agent).__name__,
            },
            'config': self.config,
        }

