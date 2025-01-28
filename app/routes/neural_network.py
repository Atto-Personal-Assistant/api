from typing import Dict

from fastapi import APIRouter, UploadFile, File

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
from app.schemas.neural_network_schemas import RequestTrainNeuralNetwork

router = APIRouter(prefix="/neural-network")


@router.post("/use")
async def post_use_neural_network(audio: UploadFile = File(...)) -> Dict[str, str]:
    bytes_audio = await audio.read()
    request = PostUseNeuralNetworkRequestModel(
        audio_interactor=AudioInteractor(),
        bytes_audio=bytes_audio,
    )
    interactor = PostUseNeuralNetworkInteractor(request)

    result = interactor.run()

    return result()


@router.post("/train")
async def post_train_neural_network(request: RequestTrainNeuralNetwork):
    request = PostTrainNeuralNetworkRequestModel(
        input=request.input,
        output=request.output,
        audio_interactor=AudioInteractor(),
    )
    interactor = PostTrainNeuralNetworkInteractor(request)

    result = interactor.run()

    return result()
