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
                 audio_interactor: AudioInteractor,
                 bytes_audio: bytes,):
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
            signal, simple_rate = librosa.load(file_fullname)
            number_dimensions = signal.ndim
            return signal, simple_rate, number_dimensions

        except Exception as error:
            print("Erro ao carregar o arquivo de áudio:", error)

    @staticmethod
    def extract_mfcc(
            signal,
            simple_rate,
            n_mfcc: int = 20):
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
    def _scale_values(arr):
        return np.array(arr) * 2 - 1

    def _use_neural_network(self,
                            sample_rate: int,
                            list_inputs: list[int],
                            n_mels: int = 20,
                            n_fft: int = 2048,
                            hop_length: int = 512,):
        neural_network = NeuralNetwork.load(file_path=self._mount_path_to_voice_folder('neural_network'))
        outputs_predict_mfccs = neural_network.predict(list_inputs)

        file_fullname = self._mount_path_to_voice_folder(
            f'{self.request.file_name}_response.{self.request.file_type}',
        )
        outputs_predict_mfccs_2d = np.array(outputs_predict_mfccs)
        outputs_predict_mfccs_2d = outputs_predict_mfccs_2d.T if (outputs_predict_mfccs_2d.shape[0] <
                                                                  outputs_predict_mfccs_2d.shape[
                                                                           1]) else outputs_predict_mfccs_2d

        print("outputs_predict_mfccs_2d", outputs_predict_mfccs_2d)


        # Reverter a transformação DCT
        mfcc_dct = librosa.feature.inverse.mfcc_to_mel(outputs_predict_mfccs_2d, n_mels=n_mels)

        print("mfcc_dct", mfcc_dct)

        # Exponenciar os resultados para reverter o logaritmo
        mfcc_dct_exp = np.exp(mfcc_dct)

        print("mfcc_dct_exp", mfcc_dct_exp)

        # Aplicar a inversa da escala de frequência de Mel
        mel_basis = librosa.filters.mel(sample_rate, n_fft=n_fft, n_mels=n_mels)
        inv_mel_basis = np.linalg.pinv(mel_basis)
        mel_space = np.dot(inv_mel_basis, mfcc_dct_exp)

        print("mel_space", mel_space)

        # Aplicar a transformada de Fourier inversa (IFT) para obter o sinal de áudio no domínio do tempo
        audio_signal = librosa.feature.inverse.mel_to_audio(
            M=mel_space,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        print("audio_signal", audio_signal)

        # Escrever o sinal de áudio reconstruído em um arquivo
        sf.write(file_fullname, audio_signal, sample_rate, format=self.request.file_type)

        return file_fullname

    def run(self):
        self._save_audio(
            self._mount_path_to_voice_folder(
                self.request.file_fullname,
            ),
            self.request.bytes_audio,
        )

        signal, simple_rate, number_dimensions = self._load_audio(
            self._mount_path_to_voice_folder(
                self.request.file_fullname,
            ),
        )
        list_audio = self.extract_mfcc(
            signal,
            simple_rate,
        )
        list_audio_normalized = self.normalize_values(
            list_audio,
        )

        file_fullname_path = self._use_neural_network(
            list_inputs=list_audio_normalized,
            sample_rate=self.request.sample_rate,
        )
        response = PostUseNeuralNetworkResponseModel(file_fullname_path)
        return response
