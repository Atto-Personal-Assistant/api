from pydantic import BaseModel


class NeuralNetworkUseSchema(BaseModel):
    audio: bytes
