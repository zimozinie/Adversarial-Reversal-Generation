#!/usr/bin/env python3
"""
ARG 评估脚本
对应论文 Section 4: Evaluation

评估基准:
- HarmBench: Attack Success Rate
- XSTest: Over-refusal rate
- MT-Bench: General reasoning
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from arg.evaluation import EvaluationManager
from arg.models import MockLLMBackbone

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ARG model on safety and utility benchmarks"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--benchmarks',
        type=str,
        nargs='+',
        default=['harmbench', 'xstest', 'mtbench'],
        choices=['harmbench', 'xstest', 'mtbench'],
        help='Benchmarks to run'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./evaluation_results.json',
        help='Path to save evaluation results'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("ARG Model Evaluation")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint or 'None (using base model)'}")
    print(f"Benchmarks: {', '.join(args.benchmarks)}")
    print("="*60 + "\n")
    
    # ========== 1. 加载模型 ==========
    print("[Step 1] Loading model...")
    model = MockLLMBackbone(hidden_dim=4096, device='cpu')
    
    if args.checkpoint:
        # TODO: 实现检查点加载
        print(f"  Loading checkpoint: {args.checkpoint}")
        print("  TODO: Implement checkpoint loading")
    
    print("  ✓ Model ready\n")
    
    # ========== 2. 创建评估管理器 ==========
    print("[Step 2] Setting up evaluators...")
    eval_manager = EvaluationManager()
    print("  ✓ Evaluators ready\n")
    
    # ========== 3. 运行评估 ==========
    print("[Step 3] Running evaluations...")
    results = eval_manager.evaluate_all(model)
    
    # ========== 4. 打印摘要 ==========
    eval_manager.print_summary(results)
    
    # ========== 5. 保存结果 ==========
    print(f"\n[Step 4] Saving results to {args.output}...")
    import json
    results_dict = {
        name: {
            'benchmark': result.benchmark_name,
            'metrics': result.metrics,
            'details': result.details
        }
        for name, result in results.items()
    }
    
    with open(args.output, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"  ✓ Results saved\n")
    
    print("="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == '__main__':
    main()

