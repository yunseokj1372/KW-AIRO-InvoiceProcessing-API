from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from app.routers import output
from app.core.config import settings

# Initialize the app
app = FastAPI(
    title="AIRO Invoice Processing API",
    description="API for processing invoice ZIP files from S3",
    version="1.0.0"
)

# API Key authentication
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return api_key

# Add routers to the app
app.include_router(output.router) 

# Root endpoint with API key protection
@app.get("/")
def read_root(api_key: str = Depends(get_api_key)):
    return {"message": "Welcome to AIRO's Invoice Processing API"}
