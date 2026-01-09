#!/usr/bin/env python3
"""
ARG 数据生成脚本
对应论文 Section 3.1-3.2: Adversarial Reversal Data Generation

运行 Multi-Agent Pipeline 生成 ReversalPair 数据
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from arg.agents import (
    AnalysisAgent,
    ReversalAgent,
    AnswerAgent,
    ValidationAgent
)
from arg.pipeline import ARGReversalPipeline
from arg.models import MockLLMBackbone
from arg.data import ARGDataset, PlaceholderDataLoader
from arg.configs import AgentConfig

import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Generate ARG Reversal Pairs using Multi-Agent Pipeline"
    )
    parser.add_argument(
        '--input_data',
        type=str,
        default=None,
        help='Path to input data (if None, use placeholder data)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='./data/reversal_pairs.json',
        help='Path to save generated reversal pairs'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--direction',
        type=str,
        choices=['safe→unsafe', 'unsafe→safe', 'both'],
        default='both',
        help='Reversal direction'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("ARG Data Generation Pipeline")
    print("="*60)
    print(f"Output path: {args.output_path}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Direction: {args.direction}")
    print("="*60 + "\n")
    
    # ========== 1. 初始化 LLM Backend ==========
    print("[Step 1] Initializing LLM Backend...")
    llm = MockLLMBackbone(hidden_dim=4096, device='cpu')
    print("  ✓ Using MockLLMBackbone (placeholder)")
    print("  Note: Replace with actual LLM for production use\n")
    
    # ========== 2. 创建 Agents ==========
    print("[Step 2] Creating Agents...")
    agent_config = AgentConfig()
    
    analysis_agent = AnalysisAgent(llm, config=agent_config.to_dict())
    reversal_agent = ReversalAgent(llm, config=agent_config.to_dict())
    answer_agent = AnswerAgent(llm, config=agent_config.to_dict())
    validation_agent = ValidationAgent(llm, config=agent_config.to_dict())
    
    print("  ✓ AnalysisAgent")
    print("  ✓ ReversalAgent")
    print("  ✓ AnswerAgent")
    print("  ✓ ValidationAgent\n")
    
    # ========== 3. 创建 Pipeline ==========
    print("[Step 3] Creating ARG Pipeline...")
    pipeline = ARGReversalPipeline(
        analysis_agent=analysis_agent,
        reversal_agent=reversal_agent,
        answer_agent=answer_agent,
        validation_agent=validation_agent
    )
    print("  ✓ Pipeline ready\n")
    
    # ========== 4. 加载输入数据 ==========
    print("[Step 4] Loading input data...")
    if args.input_data is None:
        dataloader = PlaceholderDataLoader(num_safe=5, num_unsafe=5)
        safe_instructions = dataloader.load_safe_instructions()
        unsafe_instructions = dataloader.load_unsafe_instructions()
        print(f"  ✓ Loaded {len(safe_instructions)} safe instructions (placeholder)")
        print(f"  ✓ Loaded {len(unsafe_instructions)} unsafe instructions (placeholder)\n")
    else:
        # TODO: 实现实际数据加载
        print(f"  ✓ Loading from {args.input_data}")
        raise NotImplementedError("Custom data loading not yet implemented")
    
    # ========== 5. 生成 Reversal Pairs ==========
    print("[Step 5] Generating Reversal Pairs...\n")
    
    dataset = ARGDataset()
    
    # 生成 Safe → Unsafe
    if args.direction in ['safe→unsafe', 'both']:
        print("--- Safe → Unsafe ---")
        for instruction in safe_instructions[:args.num_samples // 2]:
            pair = pipeline.generate_reversal_pair(
                instruction,
                target_direction='safe→unsafe'
            )
            if pair:
                dataset.add_pair(pair)
    
    # 生成 Unsafe → Safe
    if args.direction in ['unsafe→safe', 'both']:
        print("\n--- Unsafe → Safe ---")
        for instruction in unsafe_instructions[:args.num_samples // 2]:
            pair = pipeline.generate_reversal_pair(
                instruction,
                target_direction='unsafe→safe'
            )
            if pair:
                dataset.add_pair(pair)
    
    # ========== 6. 保存结果 ==========
    print("\n[Step 6] Saving results...")
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    
    # 转换为 JSON
    data_list = [pair.to_dict() for pair in dataset.data]
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved {len(data_list)} pairs to {args.output_path}")
    
    # ========== 7. 统计信息 ==========
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    stats = dataset.get_statistics()
    print(f"Total pairs: {stats['total']}")
    print(f"Safe → Unsafe: {stats['safe_to_unsafe']}")
    print(f"Unsafe → Safe: {stats['unsafe_to_safe']}")
    print(f"Avg similarity: {stats['avg_similarity']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()

