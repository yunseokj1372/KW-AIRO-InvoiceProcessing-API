from fastapi import FastAPI
from app.routers import redo

# Initialize the app
app = FastAPI()

# Add routers to the app
app.include_router(redo.router) 

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to AIRO's KPI app"}
