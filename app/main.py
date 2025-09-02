from fastapi import FastAPI
from app.routers import output

# Initialize the app
app = FastAPI()

# Add routers to the app
app.include_router(output.router) 

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to AIRO's Invoice Processing API"}
