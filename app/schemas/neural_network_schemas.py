from pydantic import BaseModel


class RequestUseNeuralNetwork(BaseModel):
    input: str


class RequestTrainNeuralNetwork(BaseModel):
    input: str
    output: str
