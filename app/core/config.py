import pydantic_settings

class Settings(pydantic_settings.BaseSettings):
    secret_access_key: str

    class Config:
        env_file = ".env"

settings = Settings()