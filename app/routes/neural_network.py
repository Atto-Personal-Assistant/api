from typing import Dict

from fastapi import APIRouter, UploadFile, File

from app.interactors.neural_network_interactor import NeuralNetwork
from app.interactors.post_use_neural_network_interactor import (
    PostUseNeuralNetworkRequestModel,
    PostUseNeuralNetworkInteractor,
)


nn = NeuralNetwork(2, 3, 2)

router = APIRouter(prefix="/neural-network")


@router.post("/use")
async def post_use_neural_network(audio: UploadFile = File(...)) -> Dict[str, str]:
    bytes_audio = await audio.read()
    request = PostUseNeuralNetworkRequestModel(nn=nn, bytes_audio=bytes_audio)
    interactor = PostUseNeuralNetworkInteractor(request)

    result = interactor.run()

    return result()
