import speech_recognition as speech_lib


class AudioInteractor:
    def __init__(self,
                 language: str = 'pt-BR',):
        self.empty_text = ""
        self.language = language
        self.instance_recognizer = speech_lib.Recognizer()

    def encoder(self):
        pass

    def decoder(self):
        pass

    def transcribe(self,
                   path_audio: str,):

        with speech_lib.AudioFile(path_audio) as source:
            audio_data = self.instance_recognizer.record(source)

        try:
            words = self.instance_recognizer.recognize_google(
                audio_data=audio_data,
                language=self.language,
            )
            return words

        except speech_lib.UnknownValueError:
            print("Audio don't recognize")
            return self.empty_text

        except speech_lib.RequestError as e:
            print(f"Error in access API: {e}")
            return self.empty_text
