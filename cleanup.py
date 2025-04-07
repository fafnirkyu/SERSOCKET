import kagglehub
import ffmpeg
import os

# Download latest version
path = kagglehub.dataset_download("dmitrybabko/speech-emotion-recognition-en")

# Path to the directory containing audio files
input_folder = r"data\rawdataset\Crema"
output_folder = r"data\dataset"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    if filename.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
        output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.wav")

        try:
            # Use ffmpeg to convert audio to WAV, 16kHz sample rate, mono
            ffmpeg.input(file_path).output(output_file_path, ar='16000', ac=1).run()

            print(f"Converted {filename} to {output_file_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")