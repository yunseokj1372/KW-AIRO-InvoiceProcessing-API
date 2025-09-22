import pydantic_settings
from dotenv import load_dotenv
import os

class Settings(pydantic_settings.BaseSettings):
    # AWS Configuration
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    
    # S3 Bucket Configuration
    input_bucket: str = "airo-invoice-processing"
    output_bucket: str = "airo-invoice-outputs"
    
    # API Security
    api_key: str = os.getenv("API_KEY") # Change this in .env file
    
    # Legacy field
    secret_access_key: str = ""

    class Config:
        env_file = ".env"

settings = Settings()