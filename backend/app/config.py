from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://user:pass@postgres:5432/nn_platform"
    executor_image: str = "nn-executor:latest"
    max_execution_time: int = 30
    max_memory_mb: int = 512
    max_output_size: int = 1024 * 1024  # 1MB
    rate_limit_per_minute: int = 10

    class Config:
        env_file = ".env"


settings = Settings()
