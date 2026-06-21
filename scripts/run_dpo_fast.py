#!/usr/bin/env python3
"""DPO training — single PeftModel with ref_model=None (TRL creates internal ref adapter)."""

import json
import logging
import os
from pathlib import Path

# Prevent CUDA allocator fragmentation on low-VRAM GPUs (must be set before torch import).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_ADAPTER = Path("outputs/sft/adapter")
DPO_DIR = Path("outputs/dpo_fast")   # separate dir — do NOT overwrite outputs/dpo/adapter
DPO_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# 1. SINGLE MODEL — load base + SFT adapter with is_trainable=True
#    Use fp16 for lm_head output (171k vocab → huge logits in fp32)
# ================================================================
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                          bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
                                              torch_dtype=torch.float16,
                                              device_map="auto", trust_remote_code=False)
model.config.use_cache = False
model = PeftModel.from_pretrained(model, str(SFT_ADAPTER), is_trainable=True)
model.train()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
logger.info(f"Model: {total:,} params, {trainable:,} trainable")

# ================================================================
# 2. TOKENIZER
# ================================================================
tokenizer = AutoTokenizer.from_pretrained(str(SFT_ADAPTER), trust_remote_code=False)
tokenizer.pad_token = tokenizer.eos_token

# ================================================================
# 3. DATASET (200 train, 30 eval)
# ================================================================
dataset = load_dataset("json", data_files=str(Path("data/dpo_dataset/train.jsonl")), split="train").select(range(200))
eval_data = load_dataset("json", data_files=str(Path("data/dpo_dataset/eval.jsonl")), split="train").select(range(30))

def fmt(ex):
    # DPOTrainer expects: prompt = full chat prompt (with generation header),
    # chosen/rejected = completion-only strings (NOT full dialog)
    return {
        "prompt": tokenizer.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True),
        "chosen": ex["chosen"][0]["content"],
        "rejected": ex["rejected"][0]["content"],
    }
dataset = dataset.map(fmt)
eval_data = eval_data.map(fmt)

# ================================================================
# 4. DPO CONFIG
# ================================================================
args = DPOConfig(
    output_dir=str(DPO_DIR),
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    learning_rate=5e-6,  # match run_dpo.py — 1e-4 causes reward hacking collapse
    warmup_steps=5,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=25,
    save_strategy="no",
    report_to="none",
    bf16=False, fp16=True,  # fp16 AMP for smaller logits (GTX1650 supports natively)
    max_length=768,  # 256 was too short — JSON completions need 300+ tokens
    beta=0.1,
    loss_type="sigmoid",
    dataloader_num_workers=0,
    # Pinned memory cannot be swapped — keep False on low-RAM laptops to prevent OOM kills.
    dataloader_pin_memory=False,
    remove_unused_columns=False,
)

# ================================================================
# 5. DPOTrainer with ref_model=None
#    TRL v1.5.0 detects PeftModel and auto-creates "ref" adapter
# ================================================================
trainer = DPOTrainer(
    model=model, ref_model=None, args=args, peft_config=None,
    train_dataset=dataset, eval_dataset=eval_data,
    processing_class=tokenizer,
)

# ================================================================
# 6. TRAIN
# ================================================================
logger.info("=" * 60)
logger.info("STARTING DPO (50 steps, single-model adapter ref)")
logger.info("=" * 60)

torch.cuda.reset_peak_memory_stats()
trainer.train()

# — Save —
adapter_path = DPO_DIR / "adapter"
trainer.model.save_pretrained(str(adapter_path))
tokenizer.save_pretrained(str(adapter_path))

if trainer.state.log_history:
    with open(DPO_DIR / "training_logs.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

logger.info(f"Peak GPU: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
logger.info(f"DPO complete! Adapter saved to {adapter_path}")
