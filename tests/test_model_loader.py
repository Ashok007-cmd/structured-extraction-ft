import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

# Enforce project path import for scripts/utils
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from scripts.utils.model_loader import ModelLoader, get_dtype


def test_get_dtype():
    assert get_dtype("float16") == torch.float16
    assert get_dtype("bfloat16") == torch.bfloat16
    assert get_dtype("float32") == torch.float32
    assert get_dtype("invalid") == torch.bfloat16  # default

@patch("scripts.utils.model_loader.AutoTokenizer")
@patch("scripts.utils.model_loader.AutoModelForCausalLM")
@patch("scripts.utils.model_loader.PeftModel")
@patch("scripts.utils.model_loader.prepare_model_for_kbit_training")
def test_load_quantized_model_and_tokenizer(mock_prep, mock_peft, mock_causal_lm, mock_tokenizer):
    # Mock return values
    mock_tok = MagicMock()
    mock_tok.pad_token = None
    mock_tokenizer.from_pretrained.return_value = mock_tok

    mock_model = MagicMock()
    mock_causal_lm.from_pretrained.return_value = mock_model
    mock_prep.return_value = mock_model

    mock_peft_model = MagicMock()
    mock_peft.from_pretrained.return_value = mock_peft_model

    # 1. Test basic base model load
    model, tokenizer = ModelLoader.load_quantized_model_and_tokenizer(
        model_name_or_path="mock-base-path",
        adapter_path=None,
        use_4bit=True,
        bnb_4bit_compute_dtype="float16",
        padding_side="left"
    )

    assert model == mock_model
    assert tokenizer == mock_tok
    assert tokenizer.pad_token == tokenizer.eos_token

    mock_tokenizer.from_pretrained.assert_called_with(
        "mock-base-path",
        trust_remote_code=False,
        padding_side="left"
    )

    mock_causal_lm.from_pretrained.assert_called_once()
    args, kwargs = mock_causal_lm.from_pretrained.call_args
    assert args[0] == "mock-base-path"
    assert kwargs["torch_dtype"] == torch.float16
    assert kwargs["attn_implementation"] == "sdpa"
    assert kwargs["quantization_config"] is not None
    assert kwargs["trust_remote_code"] is False

    # 2. Test model load with PEFT adapter (is_trainable=True)
    with patch("scripts.utils.model_loader.Path.exists", return_value=True):
        model, tokenizer = ModelLoader.load_quantized_model_and_tokenizer(
            model_name_or_path="mock-base-path",
            adapter_path="mock-adapter-path",
            use_4bit=True,
            is_trainable=True,
            padding_side="right"
        )

        assert model == mock_peft_model
        mock_prep.assert_called_once_with(mock_model)
        mock_peft.from_pretrained.assert_called_once_with(
            mock_model,
            "mock-adapter-path",
            is_trainable=True
        )
        mock_peft_model.train.assert_called_once()

