from base_agent import BaseAgent
from google.cloud import speech, texttospeech
import io
import os
import tempfile
from pydub import AudioSegment


class VoiceAgent(BaseAgent):
    def __init__(self):
        # Initialize Google Cloud clients
        self.stt_client = speech.SpeechClient()
        self.tts_client = texttospeech.TextToSpeechClient()


    def speech_to_text(self, audio_file_path):
    # Preprocess: convert to mono and 16kHz
        print("[INFO] Preprocessing audio...")
        sound = AudioSegment.from_file(audio_file_path)
        sound = sound.set_channels(1).set_frame_rate(16000)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            temp_path = tmp_file.name
            sound.export(temp_path, format="wav")
    
        print(f"[INFO] Processed audio saved to: {temp_path}")

    # Load processed audio for Google STT
        client = speech.SpeechClient()
        with open(temp_path, "rb") as f:
            audio_content = f.read()
    
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )

        response = client.recognize(config=config, audio=audio)

    # Clean up temp file
        os.remove(temp_path)

        if response.results:
            transcript = " ".join([result.alternatives[0].transcript for result in response.results])
            return transcript
        else:
            return "No transcription available."


    def text_to_speech(self, text: str, output_file_path: str = "output.mp3") -> str:
        """
        Converts text to speech audio using Google Cloud Text-to-Speech.
        Saves audio to output_file_path and returns the path.
        """
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
        )

        response = self.tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )

        with open(output_file_path, "wb") as out:
            out.write(response.audio_content)

        return output_file_path

    def run(self, input_data: str, mode: str = "text-to-speech") -> str:
        """
        Main run method to either convert text to speech or speech to text.
        mode: 'text-to-speech' or 'speech-to-text'

        If mode is 'speech-to-text', input_data should be path to audio file.
        If mode is 'text-to-speech', input_data should be text to convert.

        Returns transcript text (for speech-to-text) or output audio path (for text-to-speech).
        """
        if mode == "speech-to-text":
            return self.speech_to_text(input_data)
        elif mode == "text-to-speech":
            return self.text_to_speech(input_data)
        else:
            raise ValueError("Invalid mode. Use 'text-to-speech' or 'speech-to-text'.")






"""

if __name__ == "__main__":
    agent = VoiceAgent()

    # Example 1: Convert speech audio file to text
    original_audio = "/Users/rakeshcavala/agents-assemble/backend/agents/sample_audio.wav"  # Input audio file

    try:
        # Preprocess audio for Google STT

        # Run speech-to-text
        transcript = agent.run(original_audio, mode="speech-to-text")
        print(f"Transcript from '{original_audio}':\n{transcript}")
    except Exception as e:
        print(f"Error in speech-to-text: {e}")

    # Example 2: Convert text to speech audio file
    text_input = "Hello, this is a test of the Google Cloud Text-to-Speech API."
    try:
        output_audio = agent.run(text_input, mode="text-to-speech")
        print(f"Audio content saved to: {output_audio}")
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

    """

