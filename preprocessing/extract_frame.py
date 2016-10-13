__author__ = "xiangwang1223@gmail.com"
# The simple implentation of obtaining the key frames of an original video clip.

# Input: an original video file.
# Output: multiple key frames (images).
#   Please note that, 1. different videos can have different key frames (images).
#                     2. you need to select a suitable and reasonable way to
#                        deal with the multiple output (frames, images) to represent the video.

import cv2
import numpy as np


# The implementation of fetching the key frames.
def getKeyFrames(vidcap, store_frame_path):
    count = 0
    lastHist = None
    sumDiff = []
    n_frames = int(vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    while vidcap.isOpened():
        success, frame = vidcap.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        if count >0:
            diff = np.abs(hist - lastHist)
            s = np.sum(diff)
            sumDiff.append(s)
        lastHist = hist
        count += 1

    m = np.mean(sumDiff)
    std = np.std(sumDiff)

    candidates = []
    candidates_value = []
    for i in range(len(sumDiff)):
        if sumDiff[i] > m + std*3:
            candidates.append(i+1)
            candidates_value.append(sumDiff[i])

    if len(candidates) > 20:
        top10list = sorted(range(len(candidates_value)), key=lambda i: candidates_value[i])[-9:]
        res = []
        for i in top10list:
            res.append(candidates[i])
        candidates = sorted(res)

    candidates = [0] + candidates

    keyframes = []
    lastframe = -2
    for frame in candidates:
        if not frame == lastframe + 1:
            keyframes.append(frame)
        lastframe = frame

    count = 0
    for frame in keyframes:
        vidcap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame)
        success, image = vidcap.read()
        cv2.imwrite(store_frame_path+"frame%d.jpg" % count, image)
        count += 1
    return keyframes


if __name__ == '__main__':
    # 1. Set the access path to the original video clip.
    video_file = "../data/video/4.mp4"

    # 2. Open the video clip.
    vidcap = cv2.VideoCapture(video_file)

    # 3. Get and store the resulting frames via the specific path.
    keyframes = getKeyFrames(vidcap=vidcap, store_frame_path="../data/frame/4-")

    # 4. Close the video clip.
    vidcap.release()