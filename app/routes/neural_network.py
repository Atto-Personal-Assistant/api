from typing import Dict

from fastapi import APIRouter, UploadFile, File

from app.interactors.neural_network_interactor import (
    NeuralNetwork,
)
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


neural_network_interactor = NeuralNetwork(2, 3, 2)

router = APIRouter(prefix="/neural-network")


@router.post("/use")
async def post_use_neural_network(audio: UploadFile = File(...)) -> Dict[str, str]:
    bytes_audio = await audio.read()
    request = PostUseNeuralNetworkRequestModel(
        neural_network_interactor=neural_network_interactor,
        audio_interactor=AudioInteractor(),
        bytes_audio=bytes_audio,
    )
    interactor = PostUseNeuralNetworkInteractor(request)

    result = interactor.run()

    return result()


@router.post("/train")
async def post_train_neural_network(
        input_audio: UploadFile = File(...),
        output_audio: UploadFile = File(...),):
    input_audio_bytes = await input_audio.read()
    output_audio_bytes = await output_audio.read()

    request = PostTrainNeuralNetworkRequestModel(
        input_audio_bytes=input_audio_bytes,
        output_audio_bytes=output_audio_bytes,
        audio_interactor=AudioInteractor(),
        neural_network_interactor=neural_network_interactor,
    )
    interactor = PostTrainNeuralNetworkInteractor(request)

    result = interactor.run()

    return result()
