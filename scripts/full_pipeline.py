#!/usr/bin/env python3
"""
End-to-End Fine-Tuning Pipeline: SFT → DPO → Evaluation.

Runs the complete pipeline in separate processes to guarantee absolute
reclamation of GPU memory (VRAM) between the training stages.
"""

import sys
import os
import logging
import subprocess
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


def main():
    logger.info("Starting End-to-End Model Fine-Tuning Pipeline")
    t_start = time.time()

    # 1. Pre-Check: Run Unit Tests
    run_command(
        ["pytest", "tests/"],
        "Pre-check: Running Unit Tests"
    )

    # 2. Dataset Generation
    run_command(
        [sys.executable, "data/generate_dataset.py", "--sft-size", "5000", "--dpo-size", "2000"],
        "Phase 0: Dataset Generation"
    )

    # 3. Supervised Fine-Tuning (SFT)
    run_command(
        [sys.executable, "scripts/run_sft.py", "configs/sft_config.yaml"],
        "Phase 1: Supervised Fine-Tuning (SFT)"
    )

    # 4. Direct Preference Optimization (DPO)
    run_command(
        [sys.executable, "scripts/run_dpo.py", "configs/dpo_config.yaml"],
        "Phase 2: Direct Preference Optimization (DPO)"
    )

    # 5. Multi-Metric Evaluation
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
