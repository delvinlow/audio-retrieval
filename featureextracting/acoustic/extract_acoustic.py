# __author__ = "xiangwang1223@gmail.com"
# The simple implementation of extracting multiple types of traditional acoustic features,
# consisting of Mel-Frequency Cepstral Coefficient (MFCC), Zero-Crossing Rate, Melspectrogram,
#  and Root-Mean-Square features.

# Input: an original audio clip.
# Output: multiple types of acoustic features.
#   Please note that, 1. you need to select suitable and reasonable feature vector(s) to represent the video.
#                     2. if you select mfcc features, you need to decide how to change the feature matrix to vector.

# More details: http://librosa.github.io/librosa/tutorial.html#more-examples.

from __future__ import print_function
import moviepy.editor as mp
import librosa
import numpy as np


def getAcousticFeatures(audio_reading_path):
    # 1. Load the audio clip;
    y, sr = librosa.load(audio_reading_path)

    # 2. Separate harmonics and percussives into two waveforms.
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # 3. Beat track on the percussive signal.
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

    # 4. Compute MFCC features from the raw signal.
    feature_mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
    print("MFCC Feature Done:", np.shape(feature_mfcc))

    # 5. Compute Melspectrogram features from the raw signal.
    feature_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=80000)
    print("Melspectrogram Feature Done:", np.shape(feature_spect))

    # 6. Compute Zero-Crossing features from the raw signal.
    feature_zerocrossing = librosa.feature.zero_crossing_rate(y=y)
    print("Zero-Crossing Rate:", np.shape(feature_zerocrossing))

    # 7. Compute Root-Mean-Square (RMS) Energy for each frame.
    feature_energy = librosa.feature.rmse(y=y)
    print("Energy Feature:", np.shape(feature_energy))

    return feature_mfcc, feature_spect, feature_zerocrossing, feature_energy


if __name__ == '__main__':
    # 1. Set the access path to the audio clip.
    audio_reading_path = "../../data/audio/1.wav"

    # 2. Fetch the corresponding features of the audio, consisting of mfcc, melspect, zero-crossing rate, and energy.
    feature_mfcc, feature_spect, feature_zerocrossing, feature_energy = getAcousticFeatures(
        audio_reading_path=audio_reading_path)

    # 3. Select the type of feature of interest, and set the storing path.
    acoustic_storing_path = "../../feature/acoustic/1.csv"

    # 4. Store the extracted acoustic feature(s) to .csv form.
    np.savetxt(acoustic_storing_path, feature_energy, delimiter=",")