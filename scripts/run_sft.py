#!/usr/bin/env python3
"""
Phase 1: Supervised Fine-Tuning (SFT) with LoRA/QLoRA.

Fine-tunes a small language model on structured JSON extraction from
unstructured text — a task where prompting alone is often insufficient
due to output schema inconsistency.

Model: Qwen2.5-1.5B-Instruct (fits in ~4GB VRAM with QLoRA)
Quantization: 4-bit NF4 with double quantization
Adapter: LoRA (rank=16, alpha=32) on all attention + feed-forward layers

Compatible with TRL >= 1.5.0 (SFTConfig + SFTTrainer)
"""

import json
import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from scripts.utils.config import SFTConfigData
from scripts.utils.model_loader import ModelLoader, get_dtype

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    # --- Parse config ---
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/sft_config.yaml"
    logger.info(f"Loading config from: {config_path}")
    cfg = SFTConfigData.from_yaml(config_path)
    set_seed(cfg.seed)

    # --- Prepare output dir ---
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    with open(Path(cfg.output_dir) / "sft_config.yaml", "w") as f:
        yaml.dump(cfg.model_dump(), f, default_flow_style=False)

    device = torch.cuda.current_device()
    logger.info(f"Using device: {torch.cuda.get_device_name(device)}")
    logger.info(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")

    # ================================================================
    # 1. LOAD MODEL & TOKENIZER USING UNIFIED MODEL LOADER
    # ================================================================
    model, tokenizer = ModelLoader.load_quantized_model_and_tokenizer(
        model_name_or_path=cfg.model_name_or_path,
        adapter_path=None,
        use_4bit=cfg.use_4bit,
        bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        is_trainable=True,
        attn_implementation="sdpa",
        padding_side="right",
        trust_remote_code=False,
    )


    # ================================================================
    # 3. CONFIGURE LORA (TRL 1.5.0 applies PEFT internally)
    # ================================================================
    logger.info(f"Configuring LoRA:")
    logger.info(f"  r={cfg.lora_r}, alpha={cfg.lora_alpha}, dropout={cfg.lora_dropout}")
    logger.info(f"  targets: {cfg.lora_target_modules}")
    logger.info(f"  rslora: {cfg.use_rslora}")

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=cfg.use_rslora,
    )

    # ================================================================
    # 4. LOAD DATASET
    # ================================================================
    logger.info(f"Loading dataset from: {cfg.dataset_path}")

    train_dataset = load_dataset(
        "json",
        data_files=str(Path(cfg.dataset_path) / "train.jsonl"),
        split="train",
    )
    eval_dataset = load_dataset(
        "json",
        data_files=str(Path(cfg.dataset_path) / "eval.jsonl"),
        split="train",
    )

    # Subsample if needed
    if cfg.max_train_samples and len(train_dataset) > cfg.max_train_samples:
        train_dataset = train_dataset.select(range(cfg.max_train_samples))
    if cfg.max_eval_samples and len(eval_dataset) > cfg.max_eval_samples:
        eval_dataset = eval_dataset.select(range(cfg.max_eval_samples))

    logger.info(f"  Train: {len(train_dataset)} examples")
    logger.info(f"  Eval:  {len(eval_dataset)} examples")

    # ================================================================
    # 5. FORMATTING FUNCTION (TRL 1.5.0: callable passed to trainer)
    # ================================================================
    def formatting_func(example):
        """Convert messages into a single text string using the chat template."""
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    # ================================================================
    # 6. SFT CONFIG (TRL 1.5.0 uses SFTConfig, not TrainingArguments)
    # ================================================================
    compute_dtype = get_dtype(cfg.bnb_4bit_compute_dtype)

    # Build args dynamically based on eval_strategy
    sft_args = dict(
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
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        report_to=cfg.report_to,
        remove_unused_columns=cfg.remove_unused_columns,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=True,
        seed=cfg.seed,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        logging_first_step=True,
    )

    # SFTConfig-specific params
    sft_args["max_length"] = cfg.max_length
    sft_args["packing"] = cfg.packing

    # Only add eval/save-metric args when eval is enabled
    if cfg.eval_strategy != "no":
        sft_args["eval_steps"] = cfg.eval_steps
        sft_args["load_best_model_at_end"] = cfg.load_best_model_at_end
        sft_args["metric_for_best_model"] = cfg.metric_for_best_model
        sft_args["greater_is_better"] = cfg.greater_is_better

    training_args = SFTConfig(**sft_args)

    # Set pad/eos tokens in SFTConfig
    training_args.pad_token = tokenizer.pad_token
    training_args.eos_token = tokenizer.eos_token

    # ================================================================
    # 7. SFT TRAINER (TRL 1.5.0: formatting_func + processing_class)
    # ================================================================
    logger.info("Initializing SFTTrainer (TRL 1.5.0)...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        formatting_func=formatting_func,
    )

    # ================================================================
    # 8. TRAIN
    # ================================================================
    logger.info("=" * 60)
    logger.info("STARTING SFT TRAINING")
    logger.info("=" * 60)

    trainer.train()

    # ================================================================
    # 9. SAVE FINAL MODEL
    # ================================================================
    logger.info("Saving final model...")
    final_path = Path(cfg.output_dir) / "final_model"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # Also save the adapter only
    adapter_path = Path(cfg.output_dir) / "adapter"
    trainer.model.save_pretrained(str(adapter_path), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter_path))

    # Save training metrics
    if trainer.state.log_history:
        log_path = Path(cfg.output_dir) / "training_logs.json"
        with open(log_path, "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        logger.info(f"Training logs saved to {log_path}")

    logger.info("SFT training complete!")
    logger.info(f"  Best model: {trainer.state.best_model_checkpoint}")
    if trainer.state.best_metric is not None:
        logger.info(f"  Best metric ({cfg.metric_for_best_model}): {trainer.state.best_metric:.4f}")
    logger.info(f"  Model saved to: {final_path}")
    logger.info(f"  Adapter saved to: {adapter_path}")

    # Print memory summary
    if torch.cuda.is_available():
        logger.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
