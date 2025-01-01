import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Title
st.title("Siliqx NMP (No MORE Praat!)")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None:
    # Load audio file
    y, sr = librosa.load(uploaded_file, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # Display audio details
    st.write(f"Audio duration: {duration:.2f} seconds, Sample rate: {sr} Hz")

    # User inputs for start and end times
    start_time = st.number_input("Start Time (seconds)", min_value=0.0, max_value=duration, value=0.0)
    end_time = st.number_input("End Time (seconds)", min_value=0.0, max_value=duration, value=duration)

    # Clip the audio
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    y_clip = y[start_sample:end_sample]

    # Play the clipped audio
    st.audio(y_clip, format="audio/wav", sample_rate=sr)

    # Frequency range controls
    st.subheader("Visualization Controls")
    lowest_hz = st.number_input("Lowest Frequency (Hz)", min_value=0, max_value=sr // 2, value=0, step=50)
    highest_hz = st.number_input("Highest Frequency (Hz)", min_value=0, max_value=sr // 2, value=sr // 2, step=50)
    magnitude_threshold = st.slider("Magnitude Threshold for Pitch", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

    # Spectrogram visualization
    st.subheader("Spectrogram")
    fig, ax = plt.subplots(figsize=(12, 6))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y_clip)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=ax)
    ax.set_ylim([lowest_hz, highest_hz])
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    st.pyplot(fig)

    # Pitch visualization
    st.subheader("Pitch Contour")
    pitches, magnitudes = librosa.piptrack(y=y_clip, sr=sr)
    times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=sr)

    # Extract pitch values with thresholding
    indices = magnitudes.argmax(axis=0)
    pitch_values = pitches[indices, np.arange(pitches.shape[1])]
    pitch_values[magnitudes[indices, np.arange(pitches.shape[1])] < magnitude_threshold] = np.nan

    # Calculate harmonics
    harmonics = {f"H{i}": [] for i in range(2, 6)}  # Harmonics H2–H5
    for pitch in pitch_values:
        for i in range(2, 6):
            harmonic = pitch * i if pitch and pitch * i <= highest_hz else np.nan
            harmonics[f"H{i}"].append(harmonic)

    # Plot pitch contour and harmonics
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(times, pitch_values, label="Main Pitch (F0)", color="blue", s=10)
    colors = ["orange", "green", "purple", "brown"]
    for i, color in enumerate(colors, start=2):
        ax.scatter(times, harmonics[f"H{i}"], label=f"H{i}", color=color, s=10)

    ax.set_ylim([lowest_hz, highest_hz])
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    ax.set_title("Pitch Contour with Harmonics", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    st.pyplot(fig)

    # Transcription input
    st.subheader("Transcription")
    transcription = st.text_area("Enter your transcription here (Markdown supported):")
    if transcription:
        st.markdown(transcription)