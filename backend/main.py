from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.detect import router as detect_router

app = FastAPI(title="Number Plate Recognition System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://192.168.137.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect_router)


@app.get("/")
def home():
    return {"message": "ANPR System Running"}
