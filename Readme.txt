General Notes:
Three of the microclips have some audio issues. They are listed in
errors.txt. They throw an exception with getAudioClip().

Run extract_raw_audio.py and extract_raw_frames only if necessary. These
are very long operations that can take hours. 

Run extract_features_script.py when you have the audio and videos to
generate the acoustic features.

------------------------------------------------------------------------------------------------
For Developers:
1. The 'data' directory stores the several video samples in '/video/', and the corresponding extracted key frames, and audio clips in '/frame/' and '/audio/' paths, respectively.

2. The 'feature' directory stores the corresponding extracted features from the key frames, audio clips, and associated texts, consisting of the acoustic, visual, and textual features.

3. The 'preprocessing' python package is to pre-process the original video clips and result in the associated key frames and audio clips.

4. The 'featureextracting' python package is to extract visual, acoustic, and textual features from the key frames, audio clips, and texts.

5. The 'UI' python package provides the basic UI to present the key frames of the selected videos, and their estimated venue categories, which is need to be done by yourselves.

To run this demo, please install tensorflow and other python libraries in the source codes.


When running the code via pycharm, please remember to change the Project Interpreter.

What you need to:
    1. Select suitable and reasonable methods to extract necessary visual, acoustic, and textual features based on the given training and validation video dataset.

    2. Select effective and efficient ways to combine different features (e.g., early-fusion, or late-fusion).

    3. Construct feasible classifier (e.g., softmax, svm, linear regression) to estimate the venue categories of the validation video dataset, and implement the evaluation measurements (e.g., precession, recall, and f1 socre).

    4. For your online testing and presentation, you will be given a new testing video dataset, and you need to run your code, provide the final results over the given evaluation measurements.