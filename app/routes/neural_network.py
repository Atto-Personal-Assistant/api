from fastapi import APIRouter, Request, Query

from app.interactors.post_train_neural_network_interactor import (
    PostTrainNeuralNetworkRequestModel,
    PostTrainNeuralNetworkInteractor,
)
from app.interactors.post_use_neural_network_interactor import (
    PostUseNeuralNetworkRequestModel,
    PostUseNeuralNetworkInteractor,
)
from app.interactors.receive_message_neural_network import (
    ReceiveMessageNeuralNetworkRequestModel,
    ReceiveMessageNeuralNetworkInteractor
)
from app.interactors.verify_webhook_neural_network import (
    VerifyWebhookNeuralNetworkRequestModel,
    VerifyWebhookNeuralNetworkInteractor
)
from app.schemas.neural_network_schemas import (
    RequestTrainNeuralNetwork,
    RequestUseNeuralNetwork,
)

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
    )
    interactor = PostTrainNeuralNetworkInteractor(request)

    result = interactor.run()

    return result()


@router.get("/webhook")
async def verify_webhook_neural_network(
    mode: str = Query(None, alias="hub.mode"),
    challenge: str = Query(None, alias="hub.challenge"),
    verify_token: str = Query(None, alias="hub.verify_token"),
):
    request = VerifyWebhookNeuralNetworkRequestModel(
        mode=mode,
        challenge=challenge,
        verify_token=verify_token,
    )

    interactor = VerifyWebhookNeuralNetworkInteractor(request)

    result = interactor.run()

    return result()


@router.post("/webhook")
async def receive_message_neural_network(request: Request):
    data = await request.json()

    request = ReceiveMessageNeuralNetworkRequestModel(
        data=data
    )

    interactor = ReceiveMessageNeuralNetworkInteractor(request)

    result = interactor.run()

    return result()
