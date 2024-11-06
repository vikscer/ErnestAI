import os
import sounddevice as sd
import soundfile as sf
from elevenlabs import play, Voice, VoiceSettings, stream
from elevenlabs.client import ElevenLabs
from src.utils.config_loader import load_config

# Load configuration
config = load_config()
client = ElevenLabs(
  api_key=config['api_keys']['elevenlabs'], # Defaults to ELEVEN_API_KEY
)


def text_to_speech(text_for_speech):
    """Convert text to speech using ElevenLabs, stream it live, and save to a temp file after streaming."""
    try:
        # Generate the audio stream from text using ElevenLabs
        audio_generator = client.generate(
            text=text_for_speech,
            voice=Voice(
                voice_id=config["elevenlabs"]["voice_id"],
                settings=VoiceSettings(
                    stability=config["elevenlabs"]["stability"],
                    similarity_boost=config["elevenlabs"]["similarity_boost"],
                    style=config["elevenlabs"]["style"],
                    use_speaker_boost=True
                )
            ),
            model="eleven_multilingual_v2"
        )

        # Play the audio stream live
        stream(audio_generator)

        # After streaming, save the complete audio data to a file
        temp_audio_path = "temp/tts_response.wav"
        with open(temp_audio_path, "wb") as f:
            for chunk in audio_generator:
                f.write(chunk)

        print(f"Audio saved to {temp_audio_path}")

    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")
