import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml
import pytest

# Enforce project path import for scripts/utils
sys.path.append(str(Path(__file__).parent.parent.resolve()))


@patch("scripts.run_dpo.load_config")
@patch("scripts.run_dpo.ModelLoader.load_quantized_model_and_tokenizer")
@patch("scripts.run_dpo.load_dataset")
@patch("scripts.run_dpo.DPOTrainer")
def test_run_dpo_main(mock_trainer_cls, mock_load_dataset, mock_load_quantized, mock_load_config, tmp_path):
    # Mock components
    mock_tok = MagicMock()
    mock_model = MagicMock()
    mock_load_quantized.return_value = (mock_model, mock_tok)

    mock_ds = MagicMock()
    mock_ds.__len__.return_value = 100
    mock_ds.select.return_value = mock_ds
    mock_ds.map.return_value = mock_ds
    mock_load_dataset.return_value = {"train": mock_ds, "test": mock_ds}

    mock_trainer = MagicMock()
    mock_trainer.state.log_history = []
    mock_trainer_cls.return_value = mock_trainer

    # Load the real config (bypassing the patched load_config) but redirect
    # output_dir to a temp directory so main() can write its config-save/log
    # files without touching the repo.
    from scripts.run_dpo import DPOConfigData
    with open("configs/dpo_config.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    real_cfg = DPOConfigData(**cfg_dict)
    real_cfg.output_dir = str(tmp_path)
    mock_load_config.return_value = real_cfg

    # Import and run main
    from scripts.run_dpo import main

    # We pass a test config or mock CLI arg
    with patch("sys.argv", ["run_dpo.py", "configs/dpo_config.yaml"]):
        main()

    # Verify main calls
    mock_load_quantized.assert_called_once()
    mock_load_dataset.assert_called_once()
    mock_trainer_cls.assert_called_once()
    mock_trainer.train.assert_called_once()
    mock_trainer.save_model.assert_called_once()
