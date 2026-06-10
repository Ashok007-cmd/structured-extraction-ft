from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the inference API.

    All values can be overridden via environment variables prefixed with
    ``EXTRACT_`` (e.g. ``EXTRACT_MODEL_NAME_OR_PATH``,
    ``EXTRACT_ADAPTER_PATH``).
    """

    model_config = SettingsConfigDict(env_prefix="EXTRACT_", env_file=".env", extra="ignore")

    # Model
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path: Optional[str] = "./outputs/dpo/adapter"
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Generation
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.7
    max_input_tokens: int = 1024

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    max_request_chars: int = 8000

    # Concurrency / lifecycle
    max_concurrency: int = 1
    enable_warmup: bool = True


def get_settings() -> Settings:
    return Settings()
