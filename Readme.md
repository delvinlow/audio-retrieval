##Introduction:
This is a simple audio-based venue classifier program that can be used to predict the types of locations where microclips are taken at.

![Venue Classifier](https://github.com/bingoyahoo/cs2108assignment2/blob/master/Latest%20Screenshot.png)

We have extracted and generated some acoustic, visual and textual features based on the given training and validation video dataset to perform the classification. kNN and late-fusion are used to combine results from the different features.

---
##General Notes:

1. To run, install dependencies including opencv, moviepy, pyqt etc.
a. To install PyQt4: 
		i. For Mac users, use brew and just run "brew install pyqt". 
		ii. For Windows users, go to the PyQt Download page: https://www.riverbankcomputing.com/software/pyqt/download , choose the version for Py2.7, Qt4.8 and finally 32/64 bit depending on your system.

2. Simply run "python UI.py" from terminal/command prompt! Enjoy!

---
###For Developers:

1. The `data` directory stores the several video samples in `/video/`, and the corresponding extracted key frames, and audio clips in `/frame/` and `/audio/` paths, respectively.

2. The `feature` directory stores the corresponding extracted features from the key frames, audio clips, and associated texts, consisting of the acoustic, visual, and textual features.

3. The `preprocessing` python package is to pre-process the original video clips and result in the associated key frames and audio clips.

4. The `featureextracting` python package is to extract visual, acoustic, and textual features from the key frames, audio clips, and texts.

5. The `UI` python package provides the basic UI to present the key frames of the selected videos, and their estimated venue categories, which is need to be done by yourselves.

---
##Notes
Three of the microclips have some audio issues. They are listed in
errors.txt. They throw an exception with getAudioClip().

Run extract_raw_audio.py and extract_raw_frames only if necessary. These
are very long operations that can take hours. 

Run extract_features_script.py when you have the audio and videos to
generate the acoustic features.

Training files are not included. Download them [here](https://drive.google.com/file/d/0BzCduZQhBlNybURhWlBsRGcwUjA/view?usp=sh aring).




