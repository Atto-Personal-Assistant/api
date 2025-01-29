import discord


class MyBotInteractor(discord.Client):
    async def on_ready(self):
        print(f'Logado como {self.user}')

    async def on_message(self, message):
        if message.author == self.user:
            return  # Evita responder a si mesmo

        data = {"user": str(message.author), "message": message.content}
        # requests.post(API_URL, json=data)
        print(f'Message sended: {data}')
