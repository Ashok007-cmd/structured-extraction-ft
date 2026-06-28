#!/usr/bin/env python3
"""Test if single-model DPO works with is_trainable=True."""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
SFT = "outputs/sft/adapter"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                          bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
                                              device_map="auto", trust_remote_code=False)
model.config.use_cache = False
model = PeftModel.from_pretrained(model, SFT, is_trainable=True)
model.train()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Total: {total:,}, Trainable: {trainable:,}")

# Check that LoRA params require grad
lora_grad = all(p.requires_grad for n, p in model.named_parameters() if "lora_" in n)
print(f"All LoRA params require grad: {lora_grad}")

# Forward pass test
tok = AutoTokenizer.from_pretrained(SFT, trust_remote_code=False)
tok.pad_token = tok.eos_token
inputs = tok("Hello, how are you?", return_tensors="pt").to("cuda")
out = model(**inputs)
print(f"Forward pass OK, loss shape: {out.loss.shape if out.loss is not None else 'no loss'}")
loss = out.logits.mean()
loss.backward()
print("Backward pass OK")
print(f"GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"LoRA grad sample: {dict(list(model.named_parameters()))['base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight'].grad is not None}")
print("SUCCESS")
