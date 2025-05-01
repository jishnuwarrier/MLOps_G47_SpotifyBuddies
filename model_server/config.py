from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DEBUG: bool = True
    USE_MODEL: bool = False
    # Redis Connection
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # ML Model Path
    MODEL_PATH: str = ""

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
