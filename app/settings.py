import os
from dataclasses import dataclass

@dataclass
class AppConfig:
    URL: str = os.environ.get("MESO_URL", "https://en.wikipedia.org/wiki/List_of_Mesopotamian_deities")
    LLM_ID: str = os.environ.get("LLM_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    EMB_ID: str = os.environ.get("EMB_ID", "intfloat/multilingual-e5-small")
    STORAGE_DIR: str = os.environ.get("STORAGE_DIR", "storage_meso")
    TOP_K: int = int(os.environ.get("TOP_K", "4"))
    BUILD_ON_START: bool = os.environ.get("BUILD_ON_START", "true").lower() != "false"

CONFIG = AppConfig()
