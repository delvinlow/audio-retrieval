#!/usr/bin/env python
"""
This script is used to extract acoustic features from the audio clips 
in the audio_storing_dir and write them into the feature/acoustic folder.
"""
from featureextracting.acoustic import extract_acoustic
import numpy as np
import argparse
import glob

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


	file_venueid_venuename = open("dataset_vine/venue-name.txt", "r") # eg: 1	City
	file_videoid_venueid = open("dataset_vine/vine-venue-training.txt", "r") # eg: 1000110881755082752	1
	# Build a dictionary for id: name
	dict_venueid_name = {} 
	for line in file_venueid_venuename:
		venue_id, venue_name = line.split("\t", 1)
		dict_venueid_name[venue_id] = venue_name.strip()

	# Change from given 1000110881755082752	1 to 1000110881755082752 City
	dict_videoid_name = {}
	for line in file_videoid_venueid:
		video_id, venue_index = line.split("\t", 1)
		dict_videoid_name[video_id] = dict_venueid_name[venue_index.strip()]

	# Actual extracting of acoustic features
	count = 0
	for audio_path in glob.glob(args["audio_storing_dir"] + "*.wav"):
		count +=1
		audio_id_with_ext = audio_path[audio_path.rfind("/") + 1:] #eg 1001.wav
		audio_id = audio_id_with_ext.rstrip(".wav") # e.g 1001
		print audio_path
		print count
		header = " ".join([audio_id, dict_videoid_name[audio_id] ])
		# Fetch the corresponding features of the audio, consisting of mfcc, melspect, zero-crossing rate, and energy.
		feature_mfcc, feature_spect, feature_zerocrossing, feature_energy = extract_acoustic.getAcousticFeatures(audio_path)

		file_feature_mfcc = open("feature/acoustic/" + audio_id + "_feature_mfcc.csv", "wb")
		file_feature_spect = open("feature/acoustic/" + audio_id + "_feature_spect.csv", "wb")
		file_feature_zerocrossing = open("feature/acoustic/" + audio_id + "_feature_zerocrossing.csv", "wb")
		file_feature_energy = open("feature/acoustic/" + audio_id + "_feature_energy.csv", "wb")

		np.savetxt(file_feature_mfcc, feature_mfcc, delimiter=",", header=header)
		np.savetxt(file_feature_spect, feature_spect, delimiter=",", header=header)
		np.savetxt(file_feature_zerocrossing, feature_zerocrossing, delimiter=",", header=header)
		np.savetxt(file_feature_energy, feature_energy, delimiter=",", header=header)

		file_feature_mfcc.close()
		file_feature_spect.close()
		file_feature_zerocrossing.close()
		file_feature_energy.close()

