import os

import discord
from fastapi import APIRouter

from app.interactors.discord_interactor import (
    MyBotInteractor
)

router = APIRouter(prefix="/discord")


@router.get("/webhook")
async def verify_webhook_neural_network():
    return print('called webhook!')

intents = discord.Intents.default()
intents.messages = True
client = MyBotInteractor(intents=intents)
client.run(os.environ['DISCORD_TOKEN_BOT'])
