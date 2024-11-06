import os
import sounddevice as sd
import soundfile as sf
from elevenlabs import play, Voice, VoiceSettings
from elevenlabs.client import ElevenLabs
from src.utils.config_loader import load_config

# Load configuration
config = load_config()
client = ElevenLabs(
  api_key=config['api_keys']['elevenlabs'], # Defaults to ELEVEN_API_KEY
)

def text_to_speech(text_for_speech):
    """Convert text to speech using ElevenLabs and play the audio."""
    try:
        # Generate the audio from text using ElevenLabs
        audio = client.generate(
            text=text_for_speech,
            voice=Voice(
                voice_id='HKAraoM4XkoVRdCc6Iq9',
                settings=VoiceSettings(stability=0.75, similarity_boost=1, style=0.45, use_speaker_boost=True))
            ,
            model="eleven_multilingual_v2"
        )

        # Play the audio
        play(audio)
        print("Playing response audio...")
    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")
