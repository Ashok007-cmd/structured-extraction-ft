import logging
import torch
from pathlib import Path
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logger = logging.getLogger(__name__)

def get_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str, torch.bfloat16)

class ModelLoader:
    @staticmethod
    def load_quantized_model_and_tokenizer(
        model_name_or_path: str,
        adapter_path: Optional[str] = None,
        use_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "bfloat16",
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        is_trainable: bool = False,
        attn_implementation: str = "sdpa",
        padding_side: str = "right",
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Unified Model Loader for Quantized Qwen Base Model and optional LoRA adapter.
        Uses PyTorch SDPA (Scaled Dot Product Attention) for fast execution.
        """
        compute_dtype = get_dtype(bnb_4bit_compute_dtype)

        # 1. Config Quantization
        bnb_config = None
        if use_4bit:
            logger.info("Configuring bitsandbytes 4-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            )

        # 2. Load Tokenizer
        tokenizer_load_path = adapter_path if (adapter_path and Path(adapter_path).exists()) else model_name_or_path
        logger.info(f"Loading tokenizer from: {tokenizer_load_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_load_path,
            trust_remote_code=True,
            padding_side=padding_side,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 3. Load Causal LM Base Model
        logger.info(f"Loading base model: {model_name_or_path} (attn={attn_implementation})")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=compute_dtype,
            attn_implementation=attn_implementation,
        )
        model.config.use_cache = not is_trainable

        # 4. Attach Adapter if provided
        if adapter_path and Path(adapter_path).exists():
            logger.info(f"Attaching PEFT adapter: {adapter_path} (trainable={is_trainable})")
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=is_trainable)
            if is_trainable:
                model.train()
            else:
                model.eval()

        return model, tokenizer
