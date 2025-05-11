from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DEBUG: bool = True
    LOCAL: bool = True
    # Redis Connection
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    # Postgres Connection
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "1234"
    DB_NAME: str = "client"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
