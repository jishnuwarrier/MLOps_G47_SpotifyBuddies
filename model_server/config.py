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
    MODEL_CONFIG_PATH: str = ""
    # User Playlist Path
    USER_PLAYLIST_PATH: str = ""
    # RabbitMQ
    RABBITMQ_HOST: str = "rabbitmq"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "user"
    RABBITMQ_PWD: str = "1234"
    BETA: bool = True

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
