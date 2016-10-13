__author__ = "xiangwang1223@gmail.com"
# The simple implementation of obtaining the audio clip of a original video.

import moviepy.editor as mp


def getAudioClip(video_reading_path, audio_storing_path):
    clip = mp.VideoFileClip(video_reading_path)
    clip.audio.write_audiofile(audio_storing_path)


if __name__ == '__main__':
    # 1. Set the access path to the original file.
    video_reading_path = "../data/video/1.mp4"

    # 2. Set the path to store the extracted audio clip.
    audio_storing_path = "../data/audio/1.wav"

    # 3. Fetch and store the corresponding audio clip.
    getAudioClip(video_reading_path=video_reading_path, audio_storing_path=audio_storing_path)