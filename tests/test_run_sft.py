import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Enforce project path import for scripts/utils
sys.path.append(str(Path(__file__).parent.parent.resolve()))

@patch("scripts.run_sft.AutoTokenizer")
@patch("scripts.run_sft.AutoModelForCausalLM")
@patch("scripts.run_sft.load_dataset")
@patch("scripts.run_sft.SFTTrainer")
@patch("scripts.run_sft.prepare_model_for_kbit_training")
def test_run_sft_main(mock_prep, mock_trainer_cls, mock_load_dataset, mock_causal_lm, mock_tokenizer):
    # Mock components
    mock_tok = MagicMock()
    mock_tok.pad_token = None
    mock_tokenizer.from_pretrained.return_value = mock_tok
    
    mock_model = MagicMock()
    mock_causal_lm.from_pretrained.return_value = mock_model
    mock_prep.return_value = mock_model
    
    mock_ds = MagicMock()
    mock_ds.__len__.return_value = 100
    mock_ds.select.return_value = mock_ds
    mock_load_dataset.return_value = mock_ds
    
    mock_trainer = MagicMock()
    mock_trainer.state.log_history = []
    mock_trainer.state.best_model_checkpoint = "mock-best"
    mock_trainer.state.best_metric = 0.5
    mock_trainer_cls.return_value = mock_trainer
    
    # Import and run main
    from scripts.run_sft import main
    
    # We pass a test config or mock CLI arg
    with patch("sys.argv", ["run_sft.py", "configs/sft_config.yaml"]):
        # Mock file write for sft_config.yaml to avoid modifying actual config saves during test
        with patch("builtins.open", MagicMock()):
            with patch("scripts.run_sft.Path.mkdir", MagicMock()):
                main()
        
    # Verify main calls
    mock_tokenizer.from_pretrained.assert_called_once()
    mock_causal_lm.from_pretrained.assert_called_once()
    mock_load_dataset.assert_called()
    mock_trainer_cls.assert_called_once()
    mock_trainer.train.assert_called_once()
    mock_trainer.save_model.assert_called_once()
