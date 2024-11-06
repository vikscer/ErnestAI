import librosa
import numpy as np
import soundfile as sf
import os

def augment_wake_word(file_path, output_dir, sample_rate=16000):
    """Create augmented versions of a wake word sample."""
    y, sr = librosa.load(file_path, sr=sample_rate)

    # Handle pitch shift based on librosa version requirements
    try:
        y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
    except TypeError:
        # Alternative syntax if the above fails
        y_pitch_up = librosa.effects.pitch_shift(y, sr, 2)
        y_pitch_down = librosa.effects.pitch_shift(y, sr, -2)

    # Time stretching workaround by resampling
    def time_stretch_audio(y, rate):
        return librosa.resample(y, orig_sr=sr, target_sr=int(sr * rate))

    # Time stretch by resampling the audio
    y_stretch_fast = time_stretch_audio(y, 1.1)  # 10% faster
    y_stretch_slow = time_stretch_audio(y, 0.9)  # 10% slower

    # Add slight noise
    noise = np.random.normal(0, 0.005, y.shape)
    y_noisy = y + noise

    # Volume adjustments
    y_louder = y * 1.2  # Increase volume by 20%
    y_quieter = y * 0.8  # Decrease volume by 20%

    # Save augmented samples
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    sf.write(os.path.join(output_dir, f"{base_name}_pitch_up.wav"), y_pitch_up, sr)
    sf.write(os.path.join(output_dir, f"{base_name}_pitch_down.wav"), y_pitch_down, sr)
    sf.write(os.path.join(output_dir, f"{base_name}_stretch_fast.wav"), y_stretch_fast, sr)
    sf.write(os.path.join(output_dir, f"{base_name}_stretch_slow.wav"), y_stretch_slow, sr)
    sf.write(os.path.join(output_dir, f"{base_name}_noisy.wav"), y_noisy, sr)
    sf.write(os.path.join(output_dir, f"{base_name}_louder.wav"), y_louder, sr)
    sf.write(os.path.join(output_dir, f"{base_name}_quieter.wav"), y_quieter, sr)

def process_directory(input_dir, output_dir, sample_rate=16000):
    """Apply augmentation to all .wav files in a directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(input_dir, filename)
            augment_wake_word(file_path, output_dir, sample_rate)
            print(f"Processed and augmented {filename}")

# Example usage
input_directory = "../wake_word_data/wake_word"
output_directory = "../wake_word_data/wake_word"
process_directory(input_directory, output_directory)
