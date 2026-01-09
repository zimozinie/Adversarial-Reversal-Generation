"""
ARG Trainer 实现
对应论文完整训练流程的封装
"""

import torch
import os
from pathlib import Path
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader

from ..models import (
    LLMBackbone,
    RepresentationExtractor,
    SafetyDirectionLearner
)
from ..losses import SCRLoss
from ..losses.mi_estimation import MutualInformationEstimator
from ..data.dataset import ARGDataset
from ..configs import ModelConfig, TrainingConfig
from .optimization import create_multi_optimizer
from .training_loop import train_one_epoch, evaluate


class ARGTrainer:
    """
    ARG 训练器
    
    封装完整的训练流程:
    1. 模型初始化
    2. 损失函数设置
    3. 优化器创建
    4. 训练循环
    5. 模型保存/加载
    
    对应论文完整训练目标:
    L = L_KL-SFT + λ₁ * L_dir + λ₂ * L_cons + λ₃ * L_MI
    """
    
    def __init__(
        self,
        model: LLMBackbone,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        train_dataset: Optional[ARGDataset] = None,
        eval_dataset: Optional[ARGDataset] = None,
        reference_model: Optional[LLMBackbone] = None
    ):
        """
        Args:
            model: LLM backbone
            model_config: 模型配置
            training_config: 训练配置
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
            reference_model: 参考模型（用于 KL 散度）
        """
        self.model = model
        self.reference_model = reference_model
        self.model_config = model_config
        self.training_config = training_config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # 设置设备
        self.device = torch.device(model_config.device)
        
        # 初始化组件
        self._initialize_components()
        
        # 创建优化器
        self._create_optimizers()
        
        # 日志
        self.global_step = 0
        self.current_epoch = 0
    
    def _initialize_components(self):
        """初始化训练所需的所有组件"""
        
        # 1. 表示提取器
        self.rep_extractor = RepresentationExtractor(
            backbone=self.model,
            normalize=self.model_config.normalize_representations
        )
        
        # 2. 安全方向学习器
        self.safety_dir_learner = SafetyDirectionLearner(
            hidden_dim=self.model_config.hidden_size,
            init_method=self.model_config.safety_direction_init
        ).to(self.device)
        
        # 3. 互信息估计器
        self.mi_estimator = MutualInformationEstimator(
            z_dim=self.model_config.hidden_size,
            y_attr_dim=self.model_config.y_attr_dim,
            y_sem_dim=self.model_config.y_sem_dim,
            hidden_dims=self.model_config.critic_hidden_dims,
            device=str(self.device)
        )
        
        # 4. SCR 损失
        self.scr_loss = SCRLoss(
            beta=self.training_config.beta,
            lambda1=self.training_config.lambda1,
            lambda2=self.training_config.lambda2,
            lambda3=self.training_config.lambda3,
            alpha=self.training_config.alpha,
            lambda_info=self.training_config.lambda_info,
            mi_estimator=self.mi_estimator
        ).to(self.device)
    
    def _create_optimizers(self):
        """创建优化器"""
        # 收集参数组
        model_params = list(self.model.parameters()) if hasattr(self.model, 'parameters') else []
        critic_params = self.mi_estimator.parameters()
        safety_dir_params = list(self.safety_dir_learner.parameters())
        
        # 创建多个优化器
        self.optimizers = create_multi_optimizer(
            model_params=model_params,
            critic_params=critic_params,
            safety_dir_params=safety_dir_params,
            config=self.training_config.to_dict()
        )
    
    def train(self):
        """
        执行完整训练
        
        对应论文训练流程
        """
        # 验证配置
        self.training_config.validate()
        
        # 创建数据加载器
        train_dataloader = self._create_dataloader(self.train_dataset, shuffle=True)
        eval_dataloader = None
        if self.eval_dataset is not None:
            eval_dataloader = self._create_dataloader(self.eval_dataset, shuffle=False)
        
        # 创建输出目录
        os.makedirs(self.training_config.output_dir, exist_ok=True)
        os.makedirs(self.training_config.checkpoint_dir, exist_ok=True)
        
        # 训练循环
        for epoch in range(self.training_config.num_epochs):
            self.current_epoch = epoch
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.training_config.num_epochs}")
            print('='*60)
            
            # 训练一个 epoch
            train_metrics = train_one_epoch(
                model=self.model,
                dataloader=train_dataloader,
                scr_loss=self.scr_loss,
                rep_extractor=self.rep_extractor,
                safety_dir_learner=self.safety_dir_learner,
                mi_estimator=self.mi_estimator,
                optimizers=self.optimizers,
                config=self.training_config.to_dict(),
                epoch=epoch,
                device=self.device
            )
            
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            
            # 评估
            if eval_dataloader is not None and (epoch + 1) % self.training_config.eval_steps == 0:
                eval_metrics = evaluate(
                    model=self.model,
                    dataloader=eval_dataloader,
                    scr_loss=self.scr_loss,
                    rep_extractor=self.rep_extractor,
                    safety_dir_learner=self.safety_dir_learner,
                    mi_estimator=self.mi_estimator,
                    device=self.device
                )
                print(f"Eval Loss: {eval_metrics['eval_loss']:.4f}")
            
            # 保存检查点
            if (epoch + 1) % self.training_config.save_steps == 0:
                self.save_checkpoint(epoch)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
    
    def _create_dataloader(
        self,
        dataset: ARGDataset,
        shuffle: bool = True
    ) -> DataLoader:
        """创建数据加载器"""
        return DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=shuffle,
            num_workers=0,  # 简化实现
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """数据整理函数"""
        # TODO: 实现具体的数据整理逻辑
        return batch
    
    def save_checkpoint(self, epoch: int):
        """
        保存检查点
        
        Args:
            epoch: 当前 epoch
        """
        checkpoint_path = os.path.join(
            self.training_config.checkpoint_dir,
            f"checkpoint_epoch_{epoch}.pt"
        )
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'safety_direction': self.safety_dir_learner.state_dict(),
            'mi_estimator': {
                'critic_attr': self.mi_estimator.critic_attr.state_dict(),
                'critic_sem': self.mi_estimator.critic_sem.state_dict(),
            },
            'optimizers': {
                name: opt.state_dict()
                for name, opt in self.optimizers.items()
            },
            'config': {
                'model': self.model_config.to_dict(),
                'training': self.training_config.to_dict(),
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        self.safety_dir_learner.load_state_dict(checkpoint['safety_direction'])
        self.mi_estimator.critic_attr.load_state_dict(checkpoint['mi_estimator']['critic_attr'])
        self.mi_estimator.critic_sem.load_state_dict(checkpoint['mi_estimator']['critic_sem'])
        
        for name, opt in self.optimizers.items():
            if name in checkpoint['optimizers']:
                opt.load_state_dict(checkpoint['optimizers'][name])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model': self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {},
            'safety_direction_norm': self.safety_dir_learner.get_norm(),
            'device': str(self.device),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
        }

