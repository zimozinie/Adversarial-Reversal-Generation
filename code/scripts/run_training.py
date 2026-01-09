#!/usr/bin/env python3
"""
ARG 训练脚本
对应论文完整训练流程

训练目标:
L = L_KL-SFT + λ₁ * L_dir + λ₂ * L_cons + λ₃ * L_MI
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from arg.models import MockLLMBackbone
from arg.configs import ModelConfig, TrainingConfig
from arg.training import ARGTrainer
from arg.data import ARGDataset

import argparse
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Train ARG model with Safety Contrastive Regularization"
    )
    parser.add_argument(
        '--train_data',
        type=str,
        default='./data/reversal_pairs.json',
        help='Path to training data (reversal pairs)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='Output directory'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--lambda1',
        type=float,
        default=1.0,
        help='L_dir weight'
    )
    parser.add_argument(
        '--lambda2',
        type=float,
        default=1.0,
        help='L_cons weight'
    )
    parser.add_argument(
        '--lambda3',
        type=float,
        default=0.5,
        help='L_MI weight'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("ARG Training")
    print("="*60)
    print(f"Training data: {args.train_data}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Loss weights: λ₁={args.lambda1}, λ₂={args.lambda2}, λ₃={args.lambda3}")
    print("="*60 + "\n")
    
    # ========== 1. 配置 ==========
    print("[Step 1] Setting up configurations...")
    
    model_config = ModelConfig(
        model_name="Qwen/Qwen2.5-7B",
        hidden_size=4096,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        output_dir=args.output_dir
    )
    
    training_config.validate()
    print("  ✓ Configurations validated\n")
    
    # ========== 2. 加载模型 ==========
    print("[Step 2] Loading model...")
    model = MockLLMBackbone(
        hidden_dim=model_config.hidden_size,
        device=model_config.device
    )
    print(f"  ✓ Model loaded: {type(model).__name__}")
    print(f"  Device: {model_config.device}\n")
    
    # ========== 3. 加载数据 ==========
    print("[Step 3] Loading training data...")
    # TODO: 实现实际数据加载
    train_dataset = ARGDataset()
    print(f"  ✓ Loaded {len(train_dataset)} training samples")
    print("  Note: Using placeholder dataset\n")
    
    # ========== 4. 创建 Trainer ==========
    print("[Step 4] Creating trainer...")
    trainer = ARGTrainer(
        model=model,
        model_config=model_config,
        training_config=training_config,
        train_dataset=train_dataset
    )
    print("  ✓ Trainer initialized")
    print(f"  Components:")
    print(f"    - RepresentationExtractor")
    print(f"    - SafetyDirectionLearner")
    print(f"    - MutualInformationEstimator")
    print(f"    - SCRLoss\n")
    
    # ========== 5. 训练 ==========
    print("[Step 5] Starting training...\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nTraining failed: {str(e)}")
        raise
    
    # ========== 6. 保存最终模型 ==========
    print("\n[Step 6] Saving final model...")
    trainer.save_checkpoint(epoch=training_config.num_epochs - 1)
    print("  ✓ Model saved\n")
    
    print("="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == '__main__':
    main()

