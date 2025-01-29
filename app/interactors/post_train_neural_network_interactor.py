from app.interactors.matrix_interactor import Matrix
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
            input_message: str,
            output_message: str,
    ):
        self.input_message = input_message
        self.output_message = output_message


class PostTrainNeuralNetworkInteractor:
    def __init__(self, request: PostTrainNeuralNetworkRequestModel):
        self.request = request
        self.letters = ['A', 'Á', 'Ã', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                        'O', 'Ó', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.spacial_char = [" ", "!", "?", ",", "."]
        self.chars = self.letters + self.numbers + self.spacial_char
        self.range_char = 1 / len(self.chars)
        self.list_range_char = [idx * self.range_char for idx, w in enumerate(self.chars)]
        self.max_length_char = 300

    def pre_process(self, text: str, expected_len: int = 0):
        text_uppercase = text.upper()

        text_converted = []

        for word in text_uppercase:
            letters_converted = [(self.chars.index(letter) * self.range_char) for index, letter in enumerate(word)]
            text_converted += letters_converted

        if expected_len == 0 or len(text_converted) == expected_len:
            return text_converted

        text_converted_with_more_len = text_converted + [
            (self.chars.index(self.spacial_char[0])) * self.range_char
            for _ in range(expected_len)]

        return text_converted_with_more_len

    def pos_process(self, array_num: list[float]):
        text_converted = ""

        for num in array_num:
            approximate_num = self.round_to_nearest(num, self.list_range_char)

            for idx, letter in enumerate(self.chars):
                letter_range_num = idx * self.range_char
                if letter_range_num == approximate_num:
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
        loaded_neural_network = neural_network.load(file_path='neural_network')

        for _ in range(10):
            loaded_neural_network.train(intput, output)

        return loaded_neural_network

    def run(self):
        input_text = self.request.input_message
        output_text = self.request.output_message

        filled_input = (self.max_length_char - len(input_text))
        filled_output = (self.max_length_char - len(output_text))

        intput = self.pre_process(text=input_text, expected_len=filled_input)
        output = self.pre_process(text=output_text, expected_len=filled_output)

        neural_network = self._train_neural_network(
            input_nodes=self.max_length_char,
            hidden_nodes=self.max_length_char,
            output_nodes=self.max_length_char,
            intput=intput,
            output=output,
        )

        output_matrix = neural_network.predict(intput)

        result = Matrix.matrix_to_array(output_matrix)

        response_message = self.pos_process(result)

        neural_network.save(file_path='neural_network')

        response = PostTrainNeuralNetworkResponseModel(response_message)
        return response
