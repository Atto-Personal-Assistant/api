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
