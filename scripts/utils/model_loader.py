import logging
import torch
from pathlib import Path
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)

def get_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = mapping.get(dtype_str, torch.bfloat16)
    if dtype == torch.bfloat16 and torch.cuda.is_available():
        if not torch.cuda.is_bf16_supported():
            logger.warning(
                "bfloat16 is not natively supported by this GPU. "
                "Falling back to float16 for stability and performance."
            )
            return torch.float16
    return dtype

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
        trust_remote_code: bool = False,
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
            trust_remote_code=trust_remote_code,
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
            trust_remote_code=trust_remote_code,
            torch_dtype=compute_dtype,
            attn_implementation=attn_implementation,
        )
        
        if is_trainable:
            model.config.use_cache = False
            if use_4bit:
                logger.info("Preparing model for k-bit training")
                model = prepare_model_for_kbit_training(model)
        else:
            model.config.use_cache = True

        # 4. Attach Adapter if provided
        if adapter_path and Path(adapter_path).exists():
            logger.info(f"Attaching PEFT adapter: {adapter_path} (trainable={is_trainable})")
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=is_trainable)
            if is_trainable:
                model.train()
            else:
                model.eval()

        return model, tokenizer

