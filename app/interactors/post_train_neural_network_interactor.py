import librosa
import numpy as np

from app.interactors.neural_network_interactor import NeuralNetwork
from app.interactors.audio_interactor import AudioInteractor


class PostTrainNeuralNetworkResponseModel:
    def __init__(self,  response: str):
        self.response = response

    def __call__(self):
        return {
            "response": self.response
        }


class PostTrainNeuralNetworkRequestModel:
    def __init__(
            self,
            audio_interactor: AudioInteractor,
            input: str,
            output: str,
    ):
        self.audio_interactor = audio_interactor
        self.input = input
        self.output = output


class PostTrainNeuralNetworkInteractor:
    def __init__(self, request: PostTrainNeuralNetworkRequestModel):
        self.request = request
        self.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.spacial_char = [" ", "!", "?", ",", "."]
        self.chars = self.letters + self.numbers + self.spacial_char

    def pre_process(self, text: str, expected_len: int = 0):
        range_char = len(self.chars) / 1000
        text_uppercase = text.upper()

        text_converted = []

        for word in text_uppercase:
            letters_converted = [(self.chars.index(letter) * range_char) for index, letter in enumerate(word)]
            text_converted += letters_converted

        if expected_len == 0 or len(text_converted) == expected_len:
            return text_converted

        text_converted_with_more_len = text_converted + [0 for _ in range(expected_len)]

        return text_converted_with_more_len

    def pos_process(self, text: list[float]):
        range_char = len(self.chars) / 1000
        list_range_char = [idx * range_char for idx, w in enumerate(self.chars)]

        text_converted = ""

        for num in text:
            current_letter_range = self.round_to_nearest(num, list_range_char)

            for idx, letter in enumerate(self.letters):
                letter_range = idx * range_char
                if letter_range == current_letter_range:
                    letter_found = letter
                    text_converted += letter_found

        return text_converted

    @staticmethod
    def round_to_nearest(value, numbers):
        if not numbers:
            return None

        nearest = numbers[0]

        for num in numbers:
            if abs(num - value) < abs(nearest - value):
                nearest = num

        return nearest

    @staticmethod
    def _train_neural_network(
            input_nodes: int,
            hidden_nodes: int,
            output_nodes: int,
            intput: list[float],
            output: list[float], ):
        neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)
        neural_network.train(intput, output)
        return neural_network

    def run(self):
        input_text = self.request.input
        output_text = self.request.output

        print("input_text", input_text)
        print("output_text", output_text)

        calc_fill_input = (len(output_text) - len(input_text))
        calc_fill_output = (len(input_text) - len(output_text))

        filled_input = 0 if len(input_text) > len(output_text) else calc_fill_input
        filled_output = 0 if len(output_text) > len(input_text) else calc_fill_output

        intput = self.pre_process(text=input_text, expected_len=filled_input)
        output = self.pre_process(text=output_text, expected_len=filled_output)

        neural_network = self._train_neural_network(
            input_nodes=len(output_text),
            hidden_nodes=len(output_text),
            output_nodes=len(output_text),
            intput=intput,
            output=output,
        )
        print("intput", intput)
        print("output", output)
        nn = neural_network.predict(intput)

        result = nn.matrix_to_array(nn)
        response_text = self.pos_process(result)

        print("result", result)
        print("response_text", response_text)

        neural_network.save(file_path='neural_network')

        response = PostTrainNeuralNetworkResponseModel(response_text)
        return response
