import librosa
import numpy as np
import soundfile as sf


from app.interactors.neural_network_interactor import NeuralNetwork
from app.interactors.audio_interactor import AudioInteractor


class PostUseNeuralNetworkResponseModel:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def __call__(self):
        return {"file_path": self.file_path}


class PostUseNeuralNetworkRequestModel:
    def __init__(self,
                 neural_network_interactor: NeuralNetwork,
                 audio_interactor: AudioInteractor,
                 bytes_audio: bytes,):
        self.neural_network_interactor = neural_network_interactor
        self.audio_interactor = audio_interactor
        self.bytes_audio = bytes_audio
        self.file_name = "audio"
        self.file_type = "wav"
        self.file_fullname = f'{self.file_name}.{self.file_type}'
        self.sample_rate = 44100


class PostUseNeuralNetworkInteractor:
    def __init__(self, request: PostUseNeuralNetworkRequestModel):
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
    def _load_audio(file_fullname: str):
        try:
            audio_data, simple_rate = librosa.load(file_fullname)
            number_dimensions = audio_data.ndim
            return audio_data, simple_rate, number_dimensions

        except Exception as error:
            print("Erro ao carregar o arquivo de áudio:", error)

    @staticmethod
    def extract_mfcc(
            audio_data,
            simple_rate,
            n_mfcc: int = 13):
        mfccs = librosa.feature.mfcc(
            y=audio_data,
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
    def _scale_values(arr):
        return np.array(arr) * 2 - 1

    def _use_neural_network(self,
                            nn: NeuralNetwork,
                            sample_rate: int,
                            list_inputs: list[int],):
        outputs_predict = nn.predict(list_inputs)
        scaled_output = self._scale_values(outputs_predict)
        file_fullname = self._mount_path_to_voice_folder(
            f'{self.request.file_name}_response.{self.request.file_type}',
        )
        sf.write(
            file_fullname,
            scaled_output,
            sample_rate,
            format=self.request.file_type,
        )
        return file_fullname

    def run(self):
        self._save_audio(
            self._mount_path_to_voice_folder(
                self.request.file_fullname,
            ),
            self.request.bytes_audio,
        )
        audio_data, simple_rate, number_dimensions = self._load_audio(
            self._mount_path_to_voice_folder(
                self.request.file_fullname,
            ),
        )
        list_audio = self.extract_mfcc(
            audio_data,
            simple_rate,
        )
        list_audio_normalized = self.normalize_values(
            list_audio,
        )

        print("number_dimensions", number_dimensions)

        self.request.neural_network_interactor.input_nodes = number_dimensions
        # self.request.interactor_neural_network.hidden_nodes =
        # self.request.interactor_neural_network.output_nodes =

        file_fullname_path = self._use_neural_network(
            nn=self.request.neural_network_interactor,
            list_inputs=list_audio_normalized,
            sample_rate=self.request.sample_rate,
        )
        response = PostUseNeuralNetworkResponseModel(file_fullname_path)
        return response
