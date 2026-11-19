
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    GROQ_API_KEY: str | None = None

    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

    GROQ_MODEL: str = "openai/gpt-oss-20b"

    APP_ENV: str = "development"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
