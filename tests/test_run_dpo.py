import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Enforce project path import for scripts/utils
sys.path.append(str(Path(__file__).parent.parent.resolve()))

@patch("scripts.run_dpo.ModelLoader.load_quantized_model_and_tokenizer")
@patch("scripts.run_dpo.load_dataset")
@patch("scripts.run_dpo.DPOTrainer")
def test_run_dpo_main(mock_trainer_cls, mock_load_dataset, mock_load_quantized):
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
    
    # Import and run main
    from scripts.run_dpo import main
    
    # We pass a test config or mock CLI arg
    with patch("sys.argv", ["run_dpo.py", "configs/dpo_config.yaml"]):
        # Mock file write to avoid modifying actual config saves during test
        with patch("builtins.open", MagicMock()):
            with patch("scripts.run_dpo.Path.mkdir", MagicMock()):
                main()
        
    # Verify main calls
    mock_load_quantized.assert_called_once()
    mock_load_dataset.assert_called_once()
    mock_trainer_cls.assert_called_once()
    mock_trainer.train.assert_called_once()
    mock_trainer.save_model.assert_called_once()
