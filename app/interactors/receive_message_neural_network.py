import os
import requests


class ReceiveMessageNeuralNetworkResponse:
    def __init__(self):
        pass

    def __call__(self):
        return {}


class ReceiveMessageNeuralNetworkRequestModel:
    def __init__(self, data):
        self.data = data


class ReceiveMessageNeuralNetworkInteractor:
    def __init__(self, request: ReceiveMessageNeuralNetworkRequestModel):
        self.request = request

    @staticmethod
    def send_whatsapp_message(to, message):
        url = f"https://graph.facebook.com/v17.0/{os.environ['PHONE_NUMBER_ID']}/messages"
        headers = {
            "Authorization": f"Bearer {os.environ['ACCESS_TOKEN']}",
            "Content-Type": "application/json"
        }
        data = {
            "messaging_product": "whatsapp",
            "to": to,
            "text": {"body": message}
        }
        response = requests.post(url, json=data, headers=headers)
        return response.json()

    @staticmethod
    def get_message(data):
        for event in data.get('entry'):
            for change in event.get('changes'):
                if change.get('field') == 'messages':
                    messages = change["value"]["messages"]

                    print('=======> messages', messages)

                    for message in messages:
                        phone = message["from"]
                        text = message["text"]["body"]

                        print(f"Message received from {phone}: {text}")
                        return {"status": "received"}

        return {"status": "don't received"}

    def run(self):
        self.get_message(self.request.data)
        response = ReceiveMessageNeuralNetworkResponse()
        return response
