import librosa
import numpy as np
import soundfile as sf

from fastapi.responses import FileResponse

from app.interactors.neural_network_interactor import NeuralNetwork


class PostUseNeuralNetworkResponseModel:
    def __init__(self, file_fullname: str):
        self.file_fullname = file_fullname

    def __call__(self):
        return {"file_path": f'static/{self.file_fullname}'}


class PostUseNeuralNetworkRequestModel:
    def __init__(self,
                 nn: NeuralNetwork,
                 bytes_audio: bytes,):
        self.nn = nn
        self.bytes_audio = bytes_audio
        self.file_name = "audio"
        self.file_type = "wav"
        self.file_fullname = f'{self.file_name}.{self.file_type}'
        self.sample_rate = 44100


class PostUseNeuralNetworkInteractor:
    def __init__(self, request: PostUseNeuralNetworkRequestModel):
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

    def _use_neural_network(self,
                            nn: NeuralNetwork,
                            sample_rate: int,
                            list_inputs: list[int],):
        predict = nn.predict(list_inputs)
        file_fullname = f'{self.request.file_name}_response.{self.request.file_type}'
        sf.write(file_fullname, predict, sample_rate)
        return file_fullname

    def run(self):
        self._save_audio(
            file_fullname=self.request.file_fullname,
            bytes_audio=self.request.bytes_audio,
        )
        audio_data, sr = self._load_audio(
            file_fullname=self.request.file_fullname,
        )
        list_audio = self.extract_features(
            audio_data=audio_data,
            sr=sr,
        )
        list_audio_normalized = self.normalize_values(list_audio)
        file_fullname_path = self._use_neural_network(
            nn=self.request.nn,
            list_inputs=list_audio_normalized,
            sample_rate=self.request.sample_rate,
        )
        response = PostUseNeuralNetworkResponseModel(file_fullname_path)
        return response
