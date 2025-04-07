import sounddevice as sd
import librosa
import numpy as np
from nnetwork import model
from trainer import scaler
from organizer import label_encoder
from collections import deque
import time


def continuous_listener(model, scaler, sample_rate=16000, chunk_duration=0.5, analysis_duration=3):
    """
    Continuously listens to microphone and predicts emotions in real-time
    Args:
        model: Loaded TensorFlow/Keras model
        scaler: Pre-fitted StandardScaler
        sample_rate: Audio sample rate (Hz)
        chunk_duration: Duration of each audio chunk (seconds)
        analysis_duration: Duration of audio used for each prediction (seconds)
    """
    # Circular buffer to store audio chunks
    buffer_size = int(analysis_duration / chunk_duration)
    audio_buffer = deque(maxlen=buffer_size)
    
    # Pre-allocate array for analysis window
    analysis_window = np.zeros(int(analysis_duration * sample_rate))
    
    def audio_callback(indata, frames, time_info, status):
        """Called for each audio chunk from microphone"""
        audio_buffer.append(indata.copy())
        
        # When buffer has enough data
        if len(audio_buffer) == buffer_size:
            # Combine chunks into analysis window
            for i, chunk in enumerate(audio_buffer):
                start_idx = i * len(chunk)
                end_idx = start_idx + len(chunk)
                analysis_window[start_idx:end_idx] = chunk.flatten()
            
            # Extract features
            mfccs = librosa.feature.mfcc(
                y=analysis_window,
                sr=sample_rate,
                n_mfcc=13
            )
            mfccs = np.mean(mfccs, axis=1).reshape(1, -1)
            
            # Preprocess and predict
            mfccs = scaler.transform(mfccs)
            mfccs = np.expand_dims(mfccs, axis=-1)
            
            pred = model.predict(mfccs, verbose=0)
            emotion_idx = np.argmax(pred)
            emotion = label_encoder.inverse_transform([emotion_idx])[0]
            
            print(f"\rPredicted emotion: {emotion:<10} Confidence: {np.max(pred):.2f}", end='', flush=True)

    # Start streaming
    with sd.InputStream(
        channels=1,
        samplerate=sample_rate,
        blocksize=int(chunk_duration * sample_rate),
        callback=audio_callback
    ):
        print("Listening continuously... Press Ctrl+C to stop")
        while True:
            time.sleep(0.1)

# Usage
try:
    continuous_listener(model, scaler)
except KeyboardInterrupt:
    print("\nStopped listening")