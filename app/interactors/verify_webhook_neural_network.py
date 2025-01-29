import os


class VerifyWebhookNeuralNetworkResponse:
    def __init__(self, challenge: str):
        self.challenge = challenge

    def __call__(self):
        return self.challenge


class VerifyWebhookNeuralNetworkRequestModel:
    def __init__(self,
                 mode: str,
                 challenge: str,
                 verify_token: str, ):
        self.mode = mode
        self.challenge = challenge
        self.verify_token = verify_token


class VerifyWebhookNeuralNetworkInteractor:
    def __init__(self, request: VerifyWebhookNeuralNetworkRequestModel):
        self.request = request

    def verify(self):
        if self.request.mode == "subscribe" and \
                self.request.verify_token == os.environ["VERIFY_TOKEN"]:
            return int(self.request.challenge)
        return {"error": "Verify failed"}

    def run(self):
        challenge = self.verify()
        response = VerifyWebhookNeuralNetworkResponse(challenge)
        return response
