import requests
import discord

history_messages = []


class MyBotInteractor(discord.Client):
    async def on_ready(self):
        print(f'Logged with {self.user}')

    async def on_message(self, message):
        if message.author == self.user:
            return

        current_message = {"user": str(message.author), "text": message.content}
        history_messages_reversed = list(reversed(history_messages))

        for history_message in history_messages_reversed:
            words = current_message.get('text').split(' ')

            for word in words:
                history_message_username = history_message.get('user').upper()
                current_word = word.upper()

                response = history_message_username.find(current_word)

                if response != -1:
                    input_message = history_message.get('text')
                    output_message = current_message.get('text')

                    if input_message == output_message:
                        return

                    print(f'Train input_message: {input_message} and output_message: {output_message}')

                    response = requests.post('https://api-b87s.onrender.com/neural-network/train', json={
                        'input': input_message,
                        'output': output_message,
                    })

                    if response.status_code == 200:
                        print('Train with success')
                    else:
                        print('Error in train:', response.status_code)

        history_messages.append(current_message)

        print(f'Message sent: {current_message}')
