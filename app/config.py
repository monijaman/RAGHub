"""
config.py — All settings loaded from the .env file.

pydantic-settings reads .env automatically and validates types.
Access settings anywhere:  from app.config import settings
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str

    # Models
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Chunking
    chunk_size: int = 500       # max tokens per chunk
    chunk_overlap: int = 100    # tokens shared between adjacent chunks

    # Retrieval
    top_k: int = 5              # number of chunks returned per query

    # Storage paths
    upload_dir: str = "data/uploads"
    vector_db_dir: str = "data/vector_db"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
