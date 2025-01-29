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
        if "entry" in data:
            for entry in data["entry"]:
                for change in entry["changes"]:
                    if "messages" in change["value"]:
                        for message in change["value"]["messages"]:
                            phone = message["from"]  # Número do remetente
                            text = message["text"]["body"]  # Texto da mensagem

                            print(f"Mensagem recebida de {phone}: {text}")

                            return {"status": "received"}

        return {"status": "don't received"}

    def run(self):
        self.get_message(self.request.data)
        response = ReceiveMessageNeuralNetworkResponse()
        return response()
