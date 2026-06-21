#!/usr/bin/env python3
"""
End-to-End Fine-Tuning Pipeline: SFT → DPO → Evaluation.

Runs the complete pipeline in separate processes to guarantee absolute
reclamation of GPU memory (VRAM) between the training stages.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Enforce offline mode for Hugging Face
os.environ["HF_HUB_OFFLINE"] = "1"
# Reduce VRAM fragmentation on low-memory GPUs (GTX 1650 / 4 GB)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Suppress tokenizer forking noise and avoid accidental extra worker processes.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def run_command(cmd: list, description: str):
    logger.info("=" * 60)
    logger.info(f"STARTING: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("=" * 60)

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error(f"Failed: {description} (code: {result.returncode})")
        sys.exit(result.returncode)

    logger.info(f"COMPLETED: {description} in {elapsed/60:.2f} minutes\n")


def skip_if_exists(path: str, description: str, force: bool) -> bool:
    """Return True if `description` should be skipped because `path` already exists.

    Re-running training stages from scratch is the main OOM trigger on a 4 GB
    GPU, so completed stages are skipped unless the caller passes --force.
    """
    if force:
        return False
    if Path(path).exists():
        logger.info(f"SKIPPING: {description} — output already exists at {path} "
                    f"(use --force to rebuild)")
        return True
    return False


def _warn_if_low_memory() -> None:
    """Emit a warning when system RAM is below the safe threshold for training."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        available_gb = vm.available / 1e9
        swap = psutil.swap_memory()
        swap_gb = swap.total / 1e9
        logger.info(
            f"System memory: {available_gb:.1f} GB RAM available, "
            f"{swap_gb:.1f} GB swap total"
        )
        if available_gb < 4.0:
            logger.warning(
                "LOW MEMORY WARNING: less than 4 GB RAM available. "
                "Training may exhaust system memory and trigger an OOM shutdown. "
                "Close other applications before continuing."
            )
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description="End-to-end SFT → DPO → Eval pipeline")
    parser.add_argument("--force", action="store_true",
                        help="Re-run every stage even if its output already exists")
    parser.add_argument("--run-tests", action="store_true",
                        help="Run the full pytest suite as a pre-check (heavy imports; "
                             "off by default to avoid OOM on low-memory machines)")
    args = parser.parse_args()

    logger.info("Starting End-to-End Model Fine-Tuning Pipeline")
    _warn_if_low_memory()
    t_start = time.time()

    # 1. Pre-Check: Run Unit Tests (opt-in — importing the full suite spikes RAM)
    if args.run_tests:
        run_command(
            ["pytest", "tests/", "-p", "no:cacheprovider"],
            "Pre-check: Running Unit Tests"
        )
    else:
        logger.info("SKIPPING: pytest pre-check (pass --run-tests to enable)")

    # 2. Dataset Generation
    if not skip_if_exists("data/sft_dataset", "Phase 0: Dataset Generation", args.force):
        run_command(
            [sys.executable, "data/generate_dataset.py", "--sft-size", "5000", "--dpo-size", "2000"],
            "Phase 0: Dataset Generation"
        )

    # 3. Supervised Fine-Tuning (SFT)
    if not skip_if_exists("outputs/sft/adapter", "Phase 1: Supervised Fine-Tuning (SFT)", args.force):
        run_command(
            [sys.executable, "scripts/run_sft.py", "configs/sft_config.yaml"],
            "Phase 1: Supervised Fine-Tuning (SFT)"
        )

    # 4. Direct Preference Optimization (DPO)
    if not skip_if_exists("outputs/dpo/adapter", "Phase 2: Direct Preference Optimization (DPO)", args.force):
        run_command(
            [sys.executable, "scripts/run_dpo.py", "configs/dpo_config.yaml"],
            "Phase 2: Direct Preference Optimization (DPO)"
        )

    # 5. Multi-Metric Evaluation
    if not skip_if_exists("outputs/evaluation_results.json", "Phase 3: Multi-Metric Evaluation", args.force):
        run_command(
            [sys.executable, "scripts/evaluate.py"],
            "Phase 3: Multi-Metric Evaluation"
        )

    # 6. Report Generation
    run_command(
        [sys.executable, "scripts/generate_report.py"],
        "Phase 4: Generating Final Report"
    )

    total_time = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {total_time/60:.2f} minutes!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
