#!/usr/bin/env python3
"""
Phase 2: Direct Preference Optimization (DPO).

Starting from the SFT checkpoint, further tune the model to distinguish
between "good" and "bad" structured JSON outputs for the same input text.
Optimized for 4GB VRAM using single-model DPO and reference probability precomputation.
"""

import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import yaml

# Prevent CUDA allocator fragmentation on low-VRAM GPUs (must be set before torch import).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from pydantic import BaseModel, Field, field_validator
from transformers import set_seed
from trl import DPOConfig, DPOTrainer

# Enforce project path import for scripts/utils
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from scripts.utils.model_loader import ModelLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DPOConfigData(BaseModel):
    """DPO configuration validated using Pydantic."""
    # Model (starting from SFT)
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    sft_adapter_path: str = "./outputs/sft/adapter"
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # LoRA (should match SFT)
    lora_r: int = Field(8, gt=0)
    lora_alpha: int = Field(16, gt=0)
    lora_dropout: float = Field(0.05, ge=0.0, le=1.0)
    lora_target_modules: Optional[List[str]] = None
    use_rslora: bool = True

    # DPO Specific
    beta: float = Field(0.1, gt=0.0)
    loss_type: str = "sigmoid"
    precompute_ref_log_probs: bool = True   # yaml overrides; kept True as default matches config

    # Training
    output_dir: str = "./outputs/dpo"
    num_train_epochs: int = Field(1, gt=0)
    per_device_train_batch_size: int = Field(1, gt=0)
    per_device_eval_batch_size: int = Field(1, gt=0)
    gradient_accumulation_steps: int = Field(4, gt=0)
    gradient_checkpointing: bool = True
    learning_rate: float = Field(5.0e-6, gt=0.0)
    warmup_steps: int = Field(10, ge=0)
    weight_decay: float = Field(0.01, ge=0.0)
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    logging_steps: int = Field(5, gt=0)
    eval_strategy: str = "steps"
    eval_steps: int = Field(25, gt=0)
    save_strategy: str = "no"
    save_steps: int = Field(50, gt=0)
    save_total_limit: int = Field(2, gt=0)
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    max_length: int = Field(768, gt=0)
    max_prompt_length: int = Field(384, gt=0)
    report_to: str = "none"
    remove_unused_columns: bool = False
    seed: int = 42

    # Dataset
    dataset_path: str = "./data/dpo_dataset"
    max_train_samples: int = Field(1800, gt=0)
    max_eval_samples: int = Field(200, gt=0)

    @field_validator("lora_target_modules", mode="before")
    @classmethod
    def set_target_modules(cls, v):
        if v is None:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        return v


def load_config(config_path: str) -> DPOConfigData:
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)
    return DPOConfigData(**cfg_dict)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/dpo_config.yaml"
    logger.info(f"Loading DPO config from: {config_path}")
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Save config
    with open(Path(cfg.output_dir) / "dpo_config.yaml", "w") as f:
        yaml.dump(cfg.model_dump(), f, default_flow_style=False)

    device = torch.cuda.current_device()
    logger.info(f"Using device: {torch.cuda.get_device_name(device)}")
    logger.info(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")

    # ================================================================
    # 1. LOAD MODEL & TOKENIZER USING UNIFIED MODEL LOADER
    # ================================================================
    model, tokenizer = ModelLoader.load_quantized_model_and_tokenizer(
        model_name_or_path=cfg.model_name_or_path,
        adapter_path=cfg.sft_adapter_path,
        use_4bit=cfg.use_4bit,
        bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        is_trainable=True,
        attn_implementation="sdpa",
        padding_side="right",
    )

    # ================================================================
    # 4. LOAD DPO DATASET
    # ================================================================
    logger.info(f"Loading DPO dataset from: {cfg.dataset_path}")

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(Path(cfg.dataset_path) / "train.jsonl"),
            "test": str(Path(cfg.dataset_path) / "eval.jsonl"),
        },
    )

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    if cfg.max_train_samples and len(train_dataset) > cfg.max_train_samples:
        train_dataset = train_dataset.select(range(cfg.max_train_samples))
    if cfg.max_eval_samples and len(eval_dataset) > cfg.max_eval_samples:
        eval_dataset = eval_dataset.select(range(cfg.max_eval_samples))

    logger.info(f"  Train: {len(train_dataset)} examples")
    logger.info(f"  Eval:  {len(eval_dataset)} examples")

    # ================================================================
    # 5. FORMAT FUNCTION FOR DPO
    # ================================================================
    def format_dpo(example):
        """Format DPO example: prompt as text, chosen/rejected as completion-only text.
        DPOTrainer expects prompt=full_prompt_text, chosen=completion_only, rejected=completion_only.
        The dataset stores chosen/rejected as [{"role":"assistant","content":"..."}] lists.
        """
        prompt_text = tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        chosen_text = example["chosen"][0]["content"]
        rejected_text = example["rejected"][0]["content"]
        return {
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
        }

    train_dataset = train_dataset.map(format_dpo)
    eval_dataset = eval_dataset.map(format_dpo)

    from scripts.utils.model_loader import get_dtype
    compute_dtype = get_dtype(cfg.bnb_4bit_compute_dtype)

    # ================================================================
    # 6. DPO CONFIG (TRL DPOConfig inherits from TrainingArguments)
    # ================================================================
    training_args = DPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        optim=cfg.optim,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        report_to=cfg.report_to,
        remove_unused_columns=cfg.remove_unused_columns,
        seed=cfg.seed,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        logging_first_step=True,
        # Pinned memory locks RAM pages that cannot be swapped — keeps False to avoid
        # exhausting the limited swap on low-RAM laptops and triggering the OOM killer.
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        # DPO specific parameters
        beta=cfg.beta,
        loss_type=cfg.loss_type,
        max_length=cfg.max_length,
        precompute_ref_log_probs=cfg.precompute_ref_log_probs,
        precompute_ref_batch_size=1,  # low batch size for ref precompute to avoid VRAM spike
    )

    # ================================================================
    # 7. DPO TRAINER (single model, ref_model=None, peft_config=None)
    # ================================================================
    logger.info("=" * 60)
    logger.info("INITIALIZING DPO TRAINER")
    logger.info(f"  beta={cfg.beta}, loss_type={cfg.loss_type}")
    logger.info(f"  precompute_ref_log_probs={cfg.precompute_ref_log_probs}")
    logger.info("=" * 60)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=None,
    )

    # ================================================================
    # 8. TRAIN
    # ================================================================
    logger.info("STARTING DPO TRAINING")
    trainer.train()

    # Free optimizer / gradient state immediately after training to reclaim VRAM.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ================================================================
    # 9. SAVE
    # ================================================================
    logger.info("Saving DPO model...")
    final_path = Path(cfg.output_dir) / "final_model"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    adapter_path = Path(cfg.output_dir) / "adapter"
    trainer.model.save_pretrained(str(adapter_path), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter_path))

    if trainer.state.log_history:
        log_path = Path(cfg.output_dir) / "training_logs.json"
        with open(log_path, "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        logger.info(f"Training logs saved to {log_path}")

    logger.info("DPO training complete!")
    logger.info(f"  Model saved to: {final_path}")
    logger.info(f"  Adapter saved to: {adapter_path}")

    if torch.cuda.is_available():
        logger.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
