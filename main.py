from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.routes import neural_network
from app.routes import discord_routes

app = FastAPI(app_name="Atto")

load_dotenv()

discord_routes.start_bot()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(neural_network.router)
app.include_router(discord_routes.router)

app.mount("/static", StaticFiles(directory="static"), name="static")
