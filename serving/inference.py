"""Model loading and generation logic for the extraction API.

Wraps the same `ModelLoader` used by the training scripts so the serving
runtime stays consistent with how the model was trained and evaluated.
"""

import gc
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import jsonschema
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.utils.model_loader import ModelLoader
from serving.settings import Settings

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent.parent.resolve() / "data" / "schemas" / "extraction_schema.json"

EXTRACTION_PROMPT = (
    "Extract structured information from the following text as a single "
    "JSON object with keys: event_type, entities, dates, financials, "
    "relationships, metrics. Respond with JSON only.\n\nText:\n{text}"
)


def _load_schema() -> Optional[dict]:
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH) as f:
            return json.load(f)
    return None


def extract_json(text: str) -> Optional[dict]:
    """Extract a JSON object from raw model output, handling markdown fences
    and truncated trailing braces."""
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if json_match:
        text = json_match.group(1)

    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    elif start != -1:
        # Truncated output with no closing brace — try to recover.
        last_brace = text.rfind("}")
        text = text[start:last_brace + 1] if last_brace != -1 else text[start:]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class ExtractionModel:
    """Holds a loaded model + tokenizer and runs extraction inference."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.schema = _load_schema()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        adapter_path = self.settings.adapter_path
        if adapter_path and not Path(adapter_path).exists():
            logger.warning(
                "Adapter path %s does not exist; serving base model only", adapter_path
            )
            adapter_path = None

        logger.info(
            "Loading model %s (adapter=%s)", self.settings.model_name_or_path, adapter_path
        )
        self.model, self.tokenizer = ModelLoader.load_quantized_model_and_tokenizer(
            model_name_or_path=self.settings.model_name_or_path,
            adapter_path=adapter_path,
            use_4bit=self.settings.use_4bit,
            bnb_4bit_compute_dtype=self.settings.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=self.settings.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.settings.bnb_4bit_use_double_quant,
            is_trainable=False,
            attn_implementation="sdpa",
            padding_side="left",
        )
        logger.info("Model loaded successfully")

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def extract(self, text: str) -> Tuple[Optional[dict], str, bool, float]:
        """Run extraction on `text`.

        Returns (parsed_json_or_none, raw_output, schema_valid, latency_ms).
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        start = time.perf_counter()

        messages = [{"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.settings.max_input_tokens,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": self.settings.max_new_tokens,
            "do_sample": self.settings.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.settings.do_sample:
            gen_kwargs["temperature"] = self.settings.temperature

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        parsed = extract_json(raw_output)

        schema_valid = False
        if parsed is not None and self.schema is not None:
            try:
                jsonschema.validate(instance=parsed, schema=self.schema)
                schema_valid = True
            except jsonschema.ValidationError:
                schema_valid = False
        elif parsed is not None and self.schema is None:
            schema_valid = True

        latency_ms = (time.perf_counter() - start) * 1000
        return parsed, raw_output, schema_valid, latency_ms
