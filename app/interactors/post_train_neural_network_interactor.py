import librosa
import numpy as np


from app.interactors.neural_network_interactor import NeuralNetwork


class PostTrainNeuralNetworkResponseModel:
    def __init__(self):
        pass

    def __call__(self):
        return {}


class PostTrainNeuralNetworkRequestModel:
    def __init__(self,
                 nn: NeuralNetwork,
                 input_audio_bytes: bytes,
                 output_audio_bytes: bytes,):
        self.nn = nn
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
    def _save_audio(
            file_fullname: str,
            bytes_audio: bytes,):
        with open(f'static/{file_fullname}', "wb") as file:
            file.write(bytes_audio)

    @staticmethod
    def _load_audio(file_fullname: str):
        try:
            audio_data, sr = librosa.load(f'static/{file_fullname}')
            return audio_data, sr

        except Exception as error:
            print("Erro ao carregar o arquivo de áudio:", error)

    @staticmethod
    def extract_features(
            audio_data,
            sr,
            n_mfcc: int = 13):
        mfccs = librosa.feature.mfcc(
            y=audio_data,
            n_mfcc=n_mfcc,
            sr=sr,
        )
        return np.mean(mfccs.T, axis=0)

    @staticmethod
    def normalize_values(arr):
        arr = np.array(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        return (arr - min_val) / (max_val - min_val)

    @staticmethod
    def _train_neural_network(nn: NeuralNetwork,
                              input_list_audio_normalized: list[int],
                              output_list_audio_normalized: list[int],):
        nn.train(input_list_audio_normalized, output_list_audio_normalized)

    def run(self):
        self._save_audio(
            self.request.input_file_fullname, self.request.input_audio_bytes,)
        self._save_audio(
            self.request.output_file_fullname, self.request.output_audio_bytes,)

        input_audio_data, input_sr = self._load_audio(
            self.request.input_file_fullname,)
        output_audio_data, output_sr = self._load_audio(
            self.request.output_file_fullname,)

        input_list_audio = self.extract_features(input_audio_data, input_sr)
        output_list_audio = self.extract_features(output_audio_data, output_sr)

        input_list_audio_normalized = self.normalize_values(input_list_audio)
        output_list_audio_normalized = self.normalize_values(output_list_audio)

        print("input_list_audio_normalized", input_list_audio_normalized)
        print("output_list_audio_normalized", output_list_audio_normalized)

        self._train_neural_network(
            nn=self.request.nn,
            input_list_audio_normalized=input_list_audio_normalized,
            output_list_audio_normalized=output_list_audio_normalized,
        )

        response = PostTrainNeuralNetworkResponseModel()
        return response
