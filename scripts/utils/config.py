import yaml
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator

class BaseConfigData(BaseModel):
    """Base Configuration with shared hyperparameters for training."""
    # Model configuration
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # LoRA configuration
    lora_r: int = Field(8, gt=0)
    lora_alpha: int = Field(16, gt=0)
    lora_dropout: float = Field(0.05, ge=0.0, le=1.0)
    lora_target_modules: Optional[List[str]] = None
    use_rslora: bool = True

    # Training configuration
    output_dir: str
    num_train_epochs: int = Field(1, gt=0)
    per_device_train_batch_size: int = Field(1, gt=0)
    per_device_eval_batch_size: int = Field(1, gt=0)
    gradient_accumulation_steps: int = Field(4, gt=0)
    gradient_checkpointing: bool = True
    learning_rate: float = Field(3.0e-4, gt=0.0)
    warmup_steps: int = Field(10, ge=0)
    weight_decay: float = Field(0.01, ge=0.0)
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    logging_steps: int = Field(5, gt=0)
    eval_strategy: str = "no"
    eval_steps: Optional[int] = Field(50, gt=0)
    save_strategy: str = "steps"
    save_steps: Optional[int] = Field(100, gt=0)
    save_total_limit: int = Field(2, gt=0)
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    max_length: int = Field(512, gt=0)
    report_to: str = "none"
    remove_unused_columns: bool = False
    seed: int = 42
    dataloader_num_workers: int = Field(0, ge=0)

    # Dataset path parameters
    dataset_path: str
    max_train_samples: int = Field(2000, gt=0)
    max_eval_samples: int = Field(200, gt=0)

    @field_validator("lora_target_modules", mode="before")
    @classmethod
    def set_target_modules(cls, v):
        if v is None:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        return v


class SFTConfigData(BaseConfigData):
    """SFT-specific configurations."""
    output_dir: str = "./outputs/sft"
    dataset_path: str = "./data/sft_dataset"
    packing: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfigData":
        with open(path) as f:
            cfg_dict = yaml.safe_load(f)
        return cls(**cfg_dict)


class DPOConfigData(BaseConfigData):
    """DPO-specific configurations."""
    output_dir: str = "./outputs/dpo"
    dataset_path: str = "./data/dpo_dataset"
    sft_adapter_path: str = "./outputs/sft/adapter"
    beta: float = Field(0.3, gt=0.0)
    loss_type: str = "sigmoid"
    precompute_ref_log_probs: bool = True
    max_prompt_length: int = Field(384, gt=0)

    @classmethod
    def from_yaml(cls, path: str) -> "DPOConfigData":
        with open(path) as f:
            cfg_dict = yaml.safe_load(f)
        return cls(**cfg_dict)
