import os
import asyncio
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
intents.message_content = True

bot = MyBotInteractor(command_prefix="!", intents=intents)


def start_bot():
    loop = asyncio.get_event_loop()
    loop.create_task(bot.start(os.environ['DISCORD_TOKEN_BOT']))
