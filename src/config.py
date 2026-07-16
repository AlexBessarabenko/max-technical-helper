from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    yandex_api_key: str = ""
    yandex_folder_id: str = ""
    yandex_llm_model: str = "deepseek-v4-flash"
    yandex_base_url: str = "https://ai.api.cloud.yandex.net/v1"

    max_bot_token: str = ""

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "http://localhost:3000"

    kb_dir: Path = Path("data/kb")
    ad_path: Path = Path("data/ad/employees.json")
    finetune_dataset_path: Path = Path("data/finetune/dataset.jsonl")
    goldens_path: Path = Path("data/eval/goldens.json")
    chroma_dir: Path = Path("chroma_db")
    chroma_collection: str = "kb"
    retrieval_top_k: int = 4
    retrieval_min_score: float = 0.25

    finetune_base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"


@lru_cache
def get_settings() -> Settings:
    return Settings()
