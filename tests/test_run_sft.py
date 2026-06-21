import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

# Enforce project path import for scripts/utils
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from scripts.utils.config import SFTConfigData


@patch("scripts.run_sft.SFTConfigData.from_yaml")
@patch("scripts.run_sft.ModelLoader.load_quantized_model_and_tokenizer")
@patch("scripts.run_sft.load_dataset")
@patch("scripts.run_sft.SFTTrainer")
def test_run_sft_main(mock_trainer_cls, mock_load_dataset, mock_load_quantized, mock_from_yaml, tmp_path):
    # Mock components
    mock_tok = MagicMock()
    mock_model = MagicMock()
    mock_load_quantized.return_value = (mock_model, mock_tok)

    mock_ds = MagicMock()
    mock_ds.__len__.return_value = 100
    mock_ds.select.return_value = mock_ds
    mock_load_dataset.return_value = mock_ds

    mock_trainer = MagicMock()
    mock_trainer.state.log_history = []
    mock_trainer.state.best_model_checkpoint = "mock-best"
    mock_trainer.state.best_metric = 0.5
    mock_trainer_cls.return_value = mock_trainer

    # Load the real config (bypassing the patched from_yaml) but redirect
    # output_dir to a temp directory so main() can write its config-save/log
    # files without touching the repo.
    with open("configs/sft_config.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    real_cfg = SFTConfigData(**cfg_dict)
    real_cfg.output_dir = str(tmp_path)
    mock_from_yaml.return_value = real_cfg

    # Import and run main
    from scripts.run_sft import main

    # We pass a test config or mock CLI arg
    with patch("sys.argv", ["run_sft.py", "configs/sft_config.yaml"]):
        main()

    # Verify main calls
    mock_load_quantized.assert_called_once()
    mock_load_dataset.assert_called()
    mock_trainer_cls.assert_called_once()
    mock_trainer.train.assert_called_once()
    mock_trainer.save_model.assert_called_once()
