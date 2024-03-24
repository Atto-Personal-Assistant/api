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
    def _load_audio(file_path: str):
        try:
            signal, simple_rate = librosa.load(file_path)
            number_dimensions = signal.ndim
            return signal, simple_rate, number_dimensions

        except Exception as error:
            print("Erro ao carregar o arquivo de áudio:", error)

    @staticmethod
    def extract_mfcc(
            signal,
            simple_rate,
            n_mfcc: int = 13):
        mfccs = librosa.feature.mfcc(
            y=signal,
            n_mfcc=n_mfcc,
            sr=simple_rate,
        )
        return np.mean(mfccs.T, axis=0)

    @staticmethod
    def normalize_values(arr):
        arr = np.array(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        return (arr - min_val) / (max_val - min_val)

    @staticmethod
    def _train_neural_network(
            input_nodes: int,
            hidden_nodes: int,
            output_nodes: int,
            input_list_audio_normalized: list[int],
            output_list_audio_normalized: list[int],):
        neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)
        neural_network.train(input_list_audio_normalized, output_list_audio_normalized)
        return neural_network

    def run(self):
        self._save_audio(
            self._mount_path_to_voice_folder(
                self.request.input_file_fullname), self.request.input_audio_bytes,)
        self._save_audio(
            self._mount_path_to_voice_folder(
                self.request.output_file_fullname), self.request.output_audio_bytes,)

        input_signal, input_simple_rate, input_number_dimensions = self._load_audio(
            self._mount_path_to_voice_folder(
                self.request.input_file_fullname,))
        output_signal, output_simple_rate, output_number_dimensions = self._load_audio(
            self._mount_path_to_voice_folder(
                self.request.output_file_fullname,))

        input_list_audio = self.extract_mfcc(input_signal, input_simple_rate,)
        output_list_audio = self.extract_mfcc(output_signal, output_simple_rate,)

        input_list_audio_normalized = self.normalize_values(input_list_audio)
        output_list_audio_normalized = self.normalize_values(output_list_audio)

        neural_network = self._train_neural_network(
            input_nodes=input_number_dimensions,
            hidden_nodes=10,
            output_nodes=len(output_list_audio_normalized),
            input_list_audio_normalized=input_list_audio_normalized,
            output_list_audio_normalized=output_list_audio_normalized,
        )

        neural_network.save(file_path=self._mount_path_to_voice_folder('neural_network'))

        response = PostTrainNeuralNetworkResponseModel()
        return response
