#!/usr/bin/env python3
"""
Phase 3: Evaluation Pipeline.

Compares performance of:
  1. Base model (no fine-tuning) — zero-shot prompting
  2. SFT model — after supervised fine-tuning
  3. DPO model — after preference tuning

Metrics:
  - JSON parse success rate (% valid JSON)
  - Schema compliance score (all required fields present)
  - Entity recall (how many ground-truth entities extracted)
  - Entity precision (how many extracted entities are real)
  - Date normalization accuracy
  - Financial amount accuracy
  - Structural fidelity (correct nesting vs flat)

Also generates loss curves and comparison tables.
"""

import gc
import json
import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import jsonschema
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Enforce project path import for scripts/utils
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from scripts.utils.model_loader import ModelLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load extraction JSON schema for validation
SCHEMA_PATH = Path(__file__).parent.parent.resolve() / "data" / "schemas" / "extraction_schema.json"
EXTRACTION_SCHEMA = None
if SCHEMA_PATH.exists():
    try:
        with open(SCHEMA_PATH) as f:
            EXTRACTION_SCHEMA = json.load(f)
        logger.info(f"Loaded validation schema from: {SCHEMA_PATH}")
    except Exception as e:
        logger.warning(f"Failed to load JSON schema: {e}")



@dataclass
class EvalConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    sft_adapter_path: str = "./outputs/sft/adapter"
    dpo_adapter_path: str = "./outputs/dpo/adapter"
    dataset_path: str = "./data/sft_dataset"
    eval_split: str = "eval"
    max_eval_samples: int = 30
    max_new_tokens: int = 512   # full extraction JSON needs ~370 tokens; 200 truncated every
                                # output mid-object (0% valid). 512 gives headroom to close.
    seed: int = 42


# Schema validation — ground truth keys and their types
REQUIRED_SECTIONS = {
    "event_type": str,
    "entities": list,
}

REQUIRED_ENTITY_FIELDS = {"type", "name"}
REQUIRED_FINANCIAL_FIELDS = {"type", "amount"}
REQUIRED_DATE_FIELDS = {"raw", "normalized"}
REQUIRED_RELATIONSHIP_FIELDS = {"type", "subject", "object"}


def extract_json(text: str) -> Optional[Dict]:
    """Extract JSON object from model output, handling markdown fences."""
    # Try to find JSON between ```json and ``` markers
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        text = json_match.group(1)

    # Try direct JSON parse
    text = text.strip()
    # Find first { and last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def compute_metrics(predicted: Dict, ground_truth: Dict) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a single prediction.
    """
    metrics = {}
    # --- Schema compliance using jsonschema ---
    schema_compliance = 1.0
    if EXTRACTION_SCHEMA is not None:
        try:
            jsonschema.validate(instance=predicted, schema=EXTRACTION_SCHEMA)
        except jsonschema.ValidationError as ve:
            schema_compliance = 0.0
            logger.info(f"Schema validation failed for {predicted.get('event_type')}: {ve.message}")
    metrics["schema_compliance"] = schema_compliance

    # --- Section presence ---
    gt_sections = set(ground_truth.keys()) - {"_note"}  # exclude any injected noise
    pred_sections = set(predicted.keys()) - {"_note"}

    shared_sections = gt_sections & pred_sections
    section_recall = len(shared_sections) / len(gt_sections) if gt_sections else 1.0
    section_precision = len(shared_sections) / len(pred_sections) if pred_sections else 0.0
    metrics["section_recall"] = section_recall
    metrics["section_precision"] = section_precision

    # --- Entity recall & precision ---
    gt_entities = set()
    gt_entities_raw = ground_truth.get("entities", [])
    if isinstance(gt_entities_raw, list):
        for e in gt_entities_raw:
            if isinstance(e, dict):
                name = e.get("name", "")
                if name:
                    gt_entities.add(name.lower())

    pred_entities = set()
    pred_entities_raw = predicted.get("entities", [])
    if isinstance(pred_entities_raw, list):
        for e in pred_entities_raw:
            if isinstance(e, dict):
                name = e.get("name", "")
                if name:
                    pred_entities.add(name.lower())
            elif isinstance(e, str):
                pred_entities.add(e.lower())

    # Cross-reference with text: entities not in ground truth but plausible from text?
    # For simplicity, we treat ground truth as authoritative.
    true_positives = gt_entities & pred_entities
    entity_recall = len(true_positives) / len(gt_entities) if gt_entities else 1.0
    entity_precision = len(true_positives) / len(pred_entities) if pred_entities else 0.0
    entity_f1 = (2 * entity_recall * entity_precision /
                 (entity_recall + entity_precision)) if (entity_recall + entity_precision) > 0 else 0.0

    metrics["entity_recall"] = entity_recall
    metrics["entity_precision"] = entity_precision
    metrics["entity_f1"] = entity_f1

    # --- Date normalization accuracy ---
    gt_dates = set()
    gt_dates_raw = ground_truth.get("dates", [])
    if isinstance(gt_dates_raw, list):
        for d in gt_dates_raw:
            if isinstance(d, dict):
                gt_dates.add((d.get("raw", ""), d.get("normalized", "")))

    pred_dates = set()
    pred_dates_raw = predicted.get("dates", [])
    if isinstance(pred_dates_raw, list):
        for d in pred_dates_raw:
            if isinstance(d, dict) and "raw" in d and "normalized" in d:
                pred_dates.add((d.get("raw", ""), d.get("normalized", "")))

    correct_dates = gt_dates & pred_dates
    date_recall = len(correct_dates) / len(gt_dates) if gt_dates else 1.0
    date_precision = len(correct_dates) / len(pred_dates) if pred_dates else 0.0
    metrics["date_normalization_recall"] = date_recall
    metrics["date_normalization_precision"] = date_precision

    # --- Financial amount accuracy ---
    gt_amounts = set()
    gt_amounts_raw = ground_truth.get("financials", [])
    if isinstance(gt_amounts_raw, list):
        for f in gt_amounts_raw:
            if isinstance(f, dict) and "amount" in f:
                gt_amounts.add(str(f["amount"]))

    pred_amounts = set()
    pred_amounts_raw = predicted.get("financials", [])
    if isinstance(pred_amounts_raw, list):
        for f in pred_amounts_raw:
            if isinstance(f, dict) and "amount" in f:
                pred_amounts.add(str(f["amount"]))

    correct_amounts = gt_amounts & pred_amounts
    amount_recall = len(correct_amounts) / len(gt_amounts) if gt_amounts else 1.0
    amount_precision = len(correct_amounts) / len(pred_amounts) if pred_amounts else 0.0
    metrics["financial_recall"] = amount_recall
    metrics["financial_precision"] = amount_precision

    # --- Structural fidelity: correct nesting ---
    # Check that entities are nested objects, not flattened strings
    has_correct_structure = 1.0
    if "entities" in predicted:
        entities_list = predicted["entities"]
        if isinstance(entities_list, list):
            if entities_list and isinstance(entities_list[0], str):
                has_correct_structure = 0.0  # flattened
            elif entities_list and isinstance(entities_list[0], dict):
                # Check entity has required fields
                has_type = all(isinstance(e, dict) and "type" in e for e in entities_list)
                has_name = all(isinstance(e, dict) and "name" in e for e in entities_list)
                has_correct_structure = 1.0 if (has_type and has_name) else 0.5
            else:
                has_correct_structure = 0.0
        else:
            has_correct_structure = 0.0
    metrics["structural_fidelity"] = has_correct_structure

    # --- Hallucination score: entities in prediction but not in ground truth ---
    hallucinated = pred_entities - gt_entities
    metrics["hallucination_count"] = len(hallucinated)

    return metrics


@dataclass
class ModelResults:
    name: str
    valid_json_rate: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    per_sample: List[Dict] = field(default_factory=list)
    samples_with_errors: List[Dict] = field(default_factory=list)


def evaluate_model(
    name: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset,
    max_new_tokens: int = 256,
    max_samples: int = 100,
) -> ModelResults:
    """Evaluate a model on the dataset and return structured results."""
    logger.info(f"Evaluating: {name}")

    results = ModelResults(name=name, valid_json_rate=0.0)
    total_json_valid = 0
    all_metrics = []

    for idx, example in enumerate(dataset):
        if idx >= max_samples:
            break

        # Format prompt
        ground_truth = None
        if "output_json" in example:
            ground_truth = example["output_json"]
        elif "messages" in example:
            # Try to extract ground truth from assistant message
            for msg in example["messages"]:
                if msg["role"] == "assistant":
                    ground_truth = extract_json(msg["content"])
                    break

        # Build prompt from messages (exclude assistant part)
        prompt_messages = []
        if "messages" in example:
            for msg in example["messages"]:
                if msg["role"] == "assistant":
                    break
                prompt_messages.append(msg)
        else:
            continue

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize and generate
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated part (not the input)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # If the response looks like truncated JSON, attempt to close it
        stripped = response.strip()
        if stripped.startswith("{") and not stripped.endswith("}"):
            last_brace = stripped.rfind("}")
            response = stripped[:last_brace + 1] if last_brace != -1 else stripped

        # Extract JSON
        predicted_dict = extract_json(response)
        is_valid = predicted_dict is not None
        if is_valid:
            total_json_valid += 1

        sample_result = {
            "index": idx,
            "valid_json": is_valid,
            "response_preview": response[:200],
        }

        # Compute metrics if we have ground truth and valid prediction
        if ground_truth is not None and predicted_dict is not None:
            sample_metrics = compute_metrics(predicted_dict, ground_truth)
            all_metrics.append(sample_metrics)
            sample_result["metrics"] = sample_metrics

            if sample_metrics.get("entity_f1", 1.0) < 0.5:
                results.samples_with_errors.append(sample_result)

        results.per_sample.append(sample_result)

        if (idx + 1) % 25 == 0:
            logger.info(f"  [{name}] Processed {idx + 1}/{min(len(dataset), max_samples)}")

    # Aggregate
    n = len(results.per_sample)
    results.valid_json_rate = total_json_valid / n if n > 0 else 0.0

    # Average metrics across all samples
    if all_metrics:
        keys = all_metrics[0].keys()
        for key in keys:
            values = [m[key] for m in all_metrics if key in m]
            results.metrics[key] = sum(values) / len(values) if values else 0.0

    return results


def load_model_and_tokenizer(
    base_model_path: str,
    adapter_path: Optional[str] = None,
    use_4bit: bool = True,
) -> tuple:
    """Load model with optional LoRA adapter using unified loader."""
    return ModelLoader.load_quantized_model_and_tokenizer(
        model_name_or_path=base_model_path,
        adapter_path=adapter_path,
        use_4bit=use_4bit,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        is_trainable=False,
        attn_implementation="sdpa",
        padding_side="left",
    )


def print_comparison_table(results: List[ModelResults]):
    """Print a clean comparison table."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    # Collect all metric keys
    all_keys = set()
    for r in results:
        all_keys.update(r.metrics.keys())
    all_keys = sorted(all_keys)

    # Header
    header = f"{'Model':<20}"
    header += f"{'Valid JSON%':<14}"
    for key in all_keys:
        header += f"{key:<22}"
    print(header)
    print("-" * len(header))

    # Rows
    for r in results:
        row = f"{r.name:<20}"
        row += f"{r.valid_json_rate * 100:<14.1f}"
        for key in all_keys:
            val = r.metrics.get(key, 0.0)
            row += f"{val:<22.4f}"
        print(row)

    print("=" * 80)


def _run_single_model_eval(config_file: Optional[str], model_key: str, output_path: str):
    """Evaluate exactly one model (base/sft/dpo) and write results to output_path.

    Invoked in a subprocess so GPU memory is fully reclaimed after this function exits.
    """
    config = EvalConfig()
    if config_file:
        with open(config_file) as f:
            overrides = yaml.safe_load(f)
            for k, v in overrides.items():
                setattr(config, k, v)

    logger.info(f"Loading evaluation dataset from: {config.dataset_path}")
    dataset = load_dataset(
        "json",
        data_files=str(Path(config.dataset_path) / f"{config.eval_split}.jsonl"),
        split="train",
    )
    if len(dataset) > config.max_eval_samples:
        dataset = dataset.select(range(config.max_eval_samples))
    logger.info(f"  {len(dataset)} evaluation samples")

    model_map = {
        "base": ("Base Model", None),
        "sft": ("SFT Model", config.sft_adapter_path),
        "dpo": ("DPO Model", config.dpo_adapter_path),
    }
    name, adapter_path = model_map[model_key]

    if adapter_path and not Path(adapter_path).exists():
        logger.warning(f"Adapter path not found, skipping {name}: {adapter_path}")
        result = {"name": name, "valid_json_rate": 0.0, "metrics": {}, "total_samples": 0, "error_samples": 0, "skipped": True}
        with open(output_path, "w") as f:
            json.dump(result, f)
        return

    logger.info(f"Loading {name}...")
    try:
        model, tokenizer = load_model_and_tokenizer(
            base_model_path=config.model_name_or_path,
            adapter_path=adapter_path,
        )
        results = evaluate_model(
            name, model, tokenizer, dataset,
            max_new_tokens=config.max_new_tokens,
            max_samples=config.max_eval_samples,
        )
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Evaluation failed for {name}: {e}")
        result = {"name": name, "valid_json_rate": 0.0, "metrics": {}, "total_samples": 0, "error_samples": 0, "error": str(e)}
        with open(output_path, "w") as f:
            json.dump(result, f)
        return

    result = {
        "name": results.name,
        "valid_json_rate": results.valid_json_rate,
        "metrics": results.metrics,
        "total_samples": len(results.per_sample),
        "error_samples": len(results.samples_with_errors),
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"{name} results written to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", nargs="?", default=None)
    parser.add_argument("--model", choices=["base", "sft", "dpo"], default=None,
                        help="Evaluate a single model in subprocess mode")
    parser.add_argument("--output", default=None, help="Output JSON path for --model mode")
    args = parser.parse_args()

    # ── Subprocess mode: evaluate one model and exit ──────────────────────────
    if args.model:
        if not args.output:
            logger.error("--output is required when --model is set")
            sys.exit(1)
        _run_single_model_eval(args.config_file, args.model, args.output)
        return

    # ── Orchestrator mode: spawn one subprocess per model ────────────────────
    # Each subprocess fully releases GPU memory when it exits, eliminating OOM.
    logger.info("Starting evaluation — spawning one subprocess per model to isolate GPU memory")

    config_args = [args.config_file] if args.config_file else []
    tmp_dir = tempfile.mkdtemp(prefix="eval_")

    model_keys = ["base", "sft", "dpo"]
    result_paths = {k: str(Path(tmp_dir) / f"{k}_result.json") for k in model_keys}

    for key in model_keys:
        cmd = [sys.executable, str(Path(__file__).resolve())] + config_args + [
            "--model", key, "--output", result_paths[key]
        ]
        logger.info(f"Spawning subprocess for {key} model evaluation")
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            logger.warning(f"Subprocess for '{key}' exited with code {ret.returncode}")

    # ── Merge results ─────────────────────────────────────────────────────────
    all_results_raw = []
    for key in model_keys:
        path = result_paths[key]
        if Path(path).exists():
            with open(path) as f:
                all_results_raw.append(json.load(f))
        else:
            logger.warning(f"No result file for {key} model")

    # Build ModelResults objects for comparison table
    table_results = []
    for r in all_results_raw:
        mr = ModelResults(
            name=r["name"],
            valid_json_rate=r.get("valid_json_rate", 0.0),
            metrics=r.get("metrics", {}),
        )
        table_results.append(mr)

    print_comparison_table(table_results)

    # Save merged results
    out_path = Path("outputs") / "evaluation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results_raw, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # Load dataset just for sample count (no model loaded)
    config = EvalConfig()
    if args.config_file:
        with open(args.config_file) as f:
            for k, v in yaml.safe_load(f).items():
                setattr(config, k, v)
    dataset = load_dataset(
        "json",
        data_files=str(Path(config.dataset_path) / f"{config.eval_split}.jsonl"),
        split="train",
    )
    num_samples = min(len(dataset), config.max_eval_samples)

    report_data = {"num_eval_samples": num_samples, "models": all_results_raw}
    report_path = Path("docs") / "eval_data.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Report data saved to {report_path}")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
