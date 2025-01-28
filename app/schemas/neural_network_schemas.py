from pydantic import BaseModel


class RequestTrainNeuralNetwork(BaseModel):
    input: str
    output: str
