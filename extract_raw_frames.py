#!/usr/bin/env python

"""
This script is used to extract the audio and frames of the 3000 given microclips in the dataset.
The resultant .wav files and .jpg images are stored in the specified audio_storing_dir and frames_storing_dir.
"""

from preprocessing.extract_audio import getAudioClip
from preprocessing.extract_frame import getKeyFrames
import argparse
import glob
import cv2
import os 
import cPickle as pickle

if __name__ == '__main__':
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required = False, default="dataset_vine/vine/training/video/",
		help = "Path to the directory that contains the videos clips to be indexed")
	ap.add_argument("-a", "--audio_storing_dir", required = False, default="dataset_vine/vine/training/audio/",
		help = "Path to where the extracted audio will be stored")
	ap.add_argument("-f", "--frame_storing_dir", required = False, default="dataset_vine/vine/training/frame/",
		help = "Path to where the extracted frames will be stored")
	args = vars(ap.parse_args())
	error_file = open("errors.txt", "a")

	count = 0
	# Create folder
	try:
		os.makedirs(args["audio_storing_dir"])
	except OSError:
		if not os.path.isdir(args["audio_storing_dir"]):
			raise
	try:
		os.makedirs(args["frame_storing_dir"])
	except OSError:
		if not os.path.isdir(args["frame_storing_dir"]):
			raise


	# Preprocessing: use glob to grab the videos paths and loop over them
	for video_path in glob.glob(args["dataset"] + "*.mp4"):
		count += 1
		print count

		video_id_with_ext = video_path[video_path.rfind("/") + 1:] #eg 1001.mp4
		video_id = video_id_with_ext.replace(".mp4", "") # e.g 1001
		frame_storing_path = args["frame_storing_dir"] + video_id + "-"  #e.g 1001-

		# 1. Extract Audio Clips
		# try:
		# 	getAudioClip(video_path, audio_storing_path)
		# except:
		# 	error_file.write(video_path) + "\n"

		# 2. Extract Key Video Frames
		vid_cap = cv2.VideoCapture(video_path) # Open the video file	
		key_frames = getKeyFrames(vid_cap, frame_storing_path)
		vid_cap.release()
		

	# close the Errors file
	error_file.close()

