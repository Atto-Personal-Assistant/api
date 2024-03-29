import librosa
import numpy as np


from app.interactors.neural_network_interactor import NeuralNetwork
from app.interactors.audio_interactor import AudioInteractor


class PostTrainNeuralNetworkResponseModel:
    def __init__(self):
        pass

    def __call__(self):
        return {}


class PostTrainNeuralNetworkRequestModel:
    def __init__(
            self,
            audio_interactor: AudioInteractor,
            input_audio_bytes: bytes,
            output_audio_bytes: bytes,
    ):
        self.audio_interactor = audio_interactor
        self.input_audio_bytes = input_audio_bytes
        self.output_audio_bytes = output_audio_bytes

        self.input_file_name = "input_train_audio"
        self.input_file_type = "wav"
        self.input_file_fullname = f'{self.input_file_name}.{self.input_file_type}'

        self.output_file_name = "output_train_audio"
        self.output_file_type = "wav"
        self.output_file_fullname = f'{self.output_file_name}.{self.output_file_type}'


class PostTrainNeuralNetworkInteractor:
    def __init__(self, request: PostTrainNeuralNetworkRequestModel):
        self.request = request

    @staticmethod
    def _mount_path_to_voice_folder(url: str):
        return f'static/{url}'

    @staticmethod
    def _save_audio(
            file_path: str,
            bytes_audio: bytes,):
        with open(file_path, "wb") as file:
            file.write(bytes_audio)

    @staticmethod
    def pre_process(text: str, expected_len: int = 0):
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                   'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        spacial_char = [" ", "!", "?", ",", "."]

        chars = letters + numbers + spacial_char
        range_char = len(chars) / 1000
        text_uppercase = text.upper()

        text_converted = []

        for word in text_uppercase:
            letters_converted = [(chars.index(letter) * range_char) for index, letter in enumerate(word)]
            text_converted += letters_converted

        if expected_len == 0 or len(text_converted) == expected_len:
            return text_converted

        text_converted_with_more_len = text_converted + [0 for _ in range(expected_len)]

        return text_converted_with_more_len

    def pos_process(self, text: list[float]):
        print("text pos", text)

        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                   'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        spacial_char = [" ", "!", "?", ",", "."]

        chars = letters + numbers + spacial_char
        range_char = len(chars) / 1000
        list_range_char = [idx * range_char for idx, w in enumerate(chars)]

        text_converted = ""

        for num in text:
            current_letter_range = self.round_to_nearest(num, list_range_char)

            for idx, letter in enumerate(letters):
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
            output: list[float],):
        neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)
        neural_network.train(intput, output)
        return neural_network

    def run(self):
        self._save_audio(
            self._mount_path_to_voice_folder(
                self.request.input_file_fullname), self.request.input_audio_bytes,)
        self._save_audio(
            self._mount_path_to_voice_folder(
                self.request.output_file_fullname), self.request.output_audio_bytes,)

        input_text = self.request.audio_interactor.transcribe(
            self._mount_path_to_voice_folder(self.request.input_file_fullname),)

        output_text = self.request.audio_interactor.transcribe(
            self._mount_path_to_voice_folder(self.request.output_file_fullname),)

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
        nn = neural_network.predict(intput)

        result = nn.matrix_to_array(nn)
        response_text = self.pos_process(result)
        print("response_text", response_text)

        neural_network.save(file_path=self._mount_path_to_voice_folder('neural_network'))

        response = PostTrainNeuralNetworkResponseModel()
        return response
