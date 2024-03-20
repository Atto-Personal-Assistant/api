from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import neural_network

app = FastAPI(app_name="Atto")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(neural_network.router)
