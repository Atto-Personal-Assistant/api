import requests
from fastapi import APIRouter, Request, Query

from app.interactors.audio_interactor import (
    AudioInteractor,
)
from app.interactors.post_train_neural_network_interactor import (
    PostTrainNeuralNetworkRequestModel,
    PostTrainNeuralNetworkInteractor,
)
from app.interactors.post_use_neural_network_interactor import (
    PostUseNeuralNetworkRequestModel,
    PostUseNeuralNetworkInteractor,
)
from app.schemas.neural_network_schemas import RequestTrainNeuralNetwork, RequestUseNeuralNetwork

router = APIRouter(prefix="/neural-network")


@router.post("/use")
async def post_use_neural_network(request: RequestUseNeuralNetwork):
    request = PostUseNeuralNetworkRequestModel(input_message=request.input)
    interactor = PostUseNeuralNetworkInteractor(request)

    result = interactor.run()

    return result()


@router.post("/train")
async def post_train_neural_network(request: RequestTrainNeuralNetwork):
    request = PostTrainNeuralNetworkRequestModel(
        input_message=request.input,
        output_message=request.output,
        audio_interactor=AudioInteractor(),
    )
    interactor = PostTrainNeuralNetworkInteractor(request)

    result = interactor.run()

    return result()

VERIFY_TOKEN = "seu_token_de_verificacao"
ACCESS_TOKEN = "seu_token_de_acesso"
PHONE_NUMBER_ID = "21969998205"


# Endpoint para verificar o webhook
@router.get("/webhook")
async def verify_webhook(
        hub_mode: str = Query(None, alias="hub.mode"),
        hub_challenge: str = Query(None, alias="hub.challenge"),
        hub_verify_token: str = Query(None, alias="hub.verify_token"),Ï
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return int(hub_challenge)
    return {"error": "Verificação falhou"}


# Endpoint para receber mensagens do WhatsApp
@router.post("/webhook")
async def receive_message(request: Request):
    data = await request.json()

    # Verifica se a mensagem é válida
    if "entry" in data:
        for entry in data["entry"]:
            for change in entry["changes"]:
                if "messages" in change["value"]:
                    for message in change["value"]["messages"]:
                        phone = message["from"]  # Número do remetente
                        text = message["text"]["body"]  # Texto da mensagem

                        print(f"Mensagem recebida de {phone}: {text}")

                        # Gera uma resposta automática (pode substituir pela rede neural)
                        resposta = f"Recebi sua mensagem: {text}"

                        send_whatsapp_message(phone, resposta)

    return {"status": "received"}


# Função para enviar mensagem no WhatsApp
def send_whatsapp_message(to, message):
    url = f"https://graph.facebook.com/v17.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "messaging_product": "whatsapp",
        "to": to,
        "text": {"body": message}
    }
    response = requests.post(url, json=data, headers=headers)
    return response.json()
