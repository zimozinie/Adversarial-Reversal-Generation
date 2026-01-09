"""
训练配置
对应论文 Section 4 实验设置和超参数
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """
    训练配置
    
    对应论文完整损失函数:
    L = L_KL-SFT + λ₁ * L_dir + λ₂ * L_cons + λ₃ * L_MI
    """
    
    # ========== SCR 权重（需要通过 ablation 确定最优值）==========
    lambda1: float = 1.0
    """L_dir 权重"""
    
    lambda2: float = 1.0
    """L_cons 权重"""
    
    lambda3: float = 0.5
    """L_MI 权重"""
    
    # ========== KL-SFT 参数（公式 4）==========
    beta: float = 0.1
    """KL 惩罚系数"""
    
    use_reference_model: bool = True
    """是否使用参考模型计算 KL 散度"""
    
    # ========== Safety Direction Learning 参数（公式 5）==========
    alpha: float = 0.5
    """L_dir 中的平衡系数 α"""
    
    # ========== Information Separation 参数（公式 7）==========
    lambda_info: float = 1.0
    """L_MI 中的信息平衡系数 λ_info"""
    
    # ========== 优化器参数 ==========
    learning_rate: float = 1e-5
    """学习率（论文 Section 4）"""
    
    critic_learning_rate: float = 1e-4
    """Critic 学习率（通常比主模型高）"""
    
    weight_decay: float = 0.01
    """权重衰减"""
    
    optimizer: str = "adamw"
    """优化器类型"""
    
    max_grad_norm: float = 1.0
    """梯度裁剪"""
    
    # ========== 训练超参数 ==========
    num_epochs: int = 3
    """训练轮数（论文 Section 4）"""
    
    batch_size: int = 8
    """批次大小"""
    
    gradient_accumulation_steps: int = 4
    """梯度累积步数"""
    
    warmup_steps: int = 100
    """预热步数"""
    
    logging_steps: int = 10
    """日志记录间隔"""
    
    save_steps: int = 500
    """模型保存间隔"""
    
    eval_steps: int = 100
    """评估间隔"""
    
    # ========== 生成参数 ==========
    temperature: float = 1.0
    """生成温度"""
    
    max_length: int = 512
    """最大生成长度"""
    
    # ========== 数据参数 ==========
    max_seq_length: int = 2048
    """最大序列长度"""
    
    train_data_path: Optional[str] = None
    """训练数据路径"""
    
    val_data_path: Optional[str] = None
    """验证数据路径"""
    
    # ========== 输出设置 ==========
    output_dir: str = "./outputs"
    """输出目录"""
    
    checkpoint_dir: str = "./checkpoints"
    """检查点目录"""
    
    log_dir: str = "./logs"
    """日志目录"""
    
    # ========== 其他 ==========
    seed: int = 42
    """随机种子"""
    
    mixed_precision: str = "fp16"
    """混合精度训练: fp16, bf16, or None"""
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'lambda1': self.lambda1,
            'lambda2': self.lambda2,
            'lambda3': self.lambda3,
            'beta': self.beta,
            'alpha': self.alpha,
            'lambda_info': self.lambda_info,
            'learning_rate': self.learning_rate,
            'critic_learning_rate': self.critic_learning_rate,
            'weight_decay': self.weight_decay,
            'optimizer': self.optimizer,
            'max_grad_norm': self.max_grad_norm,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'warmup_steps': self.warmup_steps,
            'temperature': self.temperature,
            'max_length': self.max_length,
            'max_seq_length': self.max_seq_length,
            'seed': self.seed,
            'mixed_precision': self.mixed_precision,
        }
    
    def validate(self) -> bool:
        """验证配置的合法性"""
        assert self.lambda1 >= 0, "lambda1 must be non-negative"
        assert self.lambda2 >= 0, "lambda2 must be non-negative"
        assert self.lambda3 >= 0, "lambda3 must be non-negative"
        assert self.beta >= 0, "beta must be non-negative"
        assert 0 <= self.alpha <= 1, "alpha must be in [0, 1]"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        return True

