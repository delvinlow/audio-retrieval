#! /usr/bin/env
# -*- coding: latin2 -*- 

import sys
import os
import design
import cv2
import glob
import heapq
import index.index as acoustic_searcher
import moviepy.editor as mp
from preprocessing.extract_frame import getKeyFrames
from featureextracting.acoustic import extract_acoustic
from multiprocessing.pool import ThreadPool

from colorhist.colordescriptor import ColorDescriptor
from colorhist.searcher import Searcher

from textsearch.index_text import index_tags_normal
from textsearch.search_text import search_text_index

from preprocessing.extract_frame import getKeyFrames

# from time import sleep

class TestScript():
	def __init__(self, search_path, frame_storing_path, validation_videos_path):
		self.frames = []
		self.pool = ThreadPool(processes=8)

		self.init_audio_index()
		self.build_venues_index()
		self.build_tags_index()
		self.search_path = search_path
		self.frame_storing_path = frame_storing_path
		self.limit = 100
		
		self.cd = ColorDescriptor((8, 12, 3))
		
		self.statesConfiguration = {"colorHist": True, "visualConcept": True, "text": True, "energy": True, "zeroCrossing": True, "spect": True, "mfcc" : True}
		self.weights = {"colorHistWeight": 3, 
		 "textWeight": 3} 
		print self.weights

		self.weights_acoustic = {"energyWeight": 2, 
		 "zeroCrossingWeight": 4,
		 "spectWeight": 1, "mfccWeight": 1} #total = 10
		print self.weights_acoustic

		for video in glob.glob(validation_videos_path):
			self.show_venue_category(video)
		print glob.glob(validation_videos_path)



	def build_venues_index(self):
		"""Builds two dictionaries for venue_id, video_id and category name mapping"""
		file_venueid_venuename = open("dataset_vine/venue-name.txt", "r") # eg: 1	City
		file_videoid_venueid = open("dataset_vine/vine-venue-validation.txt", "r") # eg: 1000110881755082752	1
		file_videoid_venueid2 = open("dataset_vine/vine-venue-training.txt", "r") # eg: 1000110881755082752	1
		# Build a dictionary for id: name
		self.dict_venueid_name = {} 
		for line in file_venueid_venuename:
			venue_id, venue_name = line.split("\t", 1)
			self.dict_venueid_name[venue_id] = venue_name.strip()

		# Change from given 1000110881755082752	1 to 1000110881755082752 City
		self.dict_videoid_name = {}
		for line in file_videoid_venueid:
			video_id, venue_index = line.split("\t", 1)
			self.dict_videoid_name[video_id] = self.dict_venueid_name[venue_index.strip()]
			self.dict_videoid_name[video_id] = self.dict_venueid_name[venue_index.strip()]
		for line in file_videoid_venueid2:
			video_id, venue_index = line.split("\t", 1)
			self.dict_videoid_name[video_id] = self.dict_venueid_name[venue_index.strip()]
			self.dict_videoid_name[video_id] = self.dict_venueid_name[venue_index.strip()]
		file_venueid_venuename.close()
		file_videoid_venueid.close()
		file_videoid_venueid2.close()


	def init_audio_index(self):
		"""Read in the 3000 acoustic features files in the 4 folders into memory during __init__ """
		self.async_result_energy = self.pool.apply_async(acoustic_searcher.build_index, ("feature/acoustic/feature_energy/",))
		self.async_result_mfcc = self.pool.apply_async(acoustic_searcher.build_index, ("feature/acoustic/feature_mfcc/",))
		self.async_result_zero_crossing = self.pool.apply_async(acoustic_searcher.build_index, ("feature/acoustic/feature_zero_crossing/",))
		self.async_result_spect = self.pool.apply_async(acoustic_searcher.build_index, ("feature/acoustic/feature_spect/",))


	def search_color_hist_in_background(self, frames):
		"""Compare using color histogram""" 
		# Try using middle frame first
		middleLen = len(frames)/2
		middleFrame = frames[middleLen]
		scores_color_hist = []
		query = cv2.imread(middleFrame)
		if query != None:
			queryfeatures = ColorDescriptor((8, 12, 3)).describe(query)
			scores_color_hist = scores_color_hist + Searcher("colorhist/index_color_hist_small.csv").search(queryfeatures, self.limit)
			scores_color_hist = scores_color_hist + Searcher("colorhist/index_color_hist_small_normal.csv").search(queryfeatures, self.limit)
		else:
			for frame in reversed(self.frames):
				query = cv2.imread(frame)
				if query != None:
					queryfeatures = ColorDescriptor((8, 12, 3)).describe(query)
					scores_color_hist = scores_color_hist + Searcher("colorhist/index_color_hist_small.csv").search(queryfeatures, self.limit)
					scores_color_hist = scores_color_hist + Searcher("colorhist/index_color_hist_small_normal.csv").search(queryfeatures, self.limit)
					break
		return scores_color_hist


	def get_top_scorers(self, scores, limit=16):
		"""Get top 16 results from the list scores"""
		heap = []
		for video_id in scores:
			if scores[video_id] != sys.float_info.max:
				heapq.heappush(heap, (scores[video_id], video_id))

		largest = heapq.nsmallest(limit, heap) # Filter to Top K results based on score
		return largest

	def add_to_final_scores(self, scores_feature, final_scores_cat, weight):
		"""Add the weighted feature score of top 16 video to the final scores"""
		remaining_points = 16
		TOTAL_POINTS = 136.0 # 1+2+...+16
		for (score, video_id) in self.get_top_scorers(scores_feature):
			venue_name = self.dict_videoid_name[video_id]
			current_score = final_scores_cat.setdefault(venue_name, 0)
			final_scores_cat[venue_name] = current_score + remaining_points/TOTAL_POINTS * weight
			remaining_points -= 1
		return final_scores_cat

	def add_text_to_final_scores(self, scores_feature, final_scores_cat, weight):
		"""Add the weighted feature score of top 16 video to the final scores"""
		remaining_points = len(scores_feature)
		TOTAL_POINTS = 136.0 # 1+2+...+16
		for (score, video_id) in scores_feature:
			venue_name = self.dict_videoid_name[video_id]
			current_score = final_scores_cat.setdefault(venue_name, 0)
			final_scores_cat[venue_name] = current_score + remaining_points/TOTAL_POINTS * weight
			remaining_points -= 1
		return final_scores_cat

	def add_hist_to_final_scores(self, scores_feature, final_scores_cat, weight):
		"""Add the weighted feature score of top 16 video to the final scores"""
		remaining_points = len(scores_feature)
		TOTAL_POINTS = 136.0 # 1+2+...+16
		for (score, video_id) in scores_feature:
			video_id = video_id.split("-")[0]
			venue_name = self.dict_videoid_name[video_id]
			current_score = final_scores_cat.setdefault(venue_name, 0)
			final_scores_cat[venue_name] = current_score + remaining_points/TOTAL_POINTS * weight
			remaining_points -= 1
		return final_scores_cat

	def fuse_scores(self, final_score_energy, final_score_zero_crossing, final_score_spect, final_score_mfcc, final_scores_text, final_scores_colorhist):
		"""Fuse the scores by summing the five dictionaries"""
		WEIGHT_ENERGY = self.weights_acoustic["energyWeight"]
		WEIGHT_ZERO_CROSSING= self.weights_acoustic["zeroCrossingWeight"]
		WEIGHT_SPECT = self.weights_acoustic["spectWeight"]
		WEIGHT_MFCC = self.weights_acoustic["mfccWeight"]
		if len(str(self.tags_search.text().toLatin1())) > 0:
			WEIGHT_TEXT = self.weights["textWeight"]
		else: 
			WEIGHT_TEXT = 0
		WEIGHT_COLOR_HIST = self.weights["colorHistWeight"]
		SUM_WEIGHTS = WEIGHT_ENERGY + WEIGHT_ZERO_CROSSING + WEIGHT_SPECT + WEIGHT_MFCC + WEIGHT_TEXT +WEIGHT_COLOR_HIST

		sum_of_scores = { k: (WEIGHT_ENERGY * final_score_energy.get(k, 0) + WEIGHT_ZERO_CROSSING * final_score_zero_crossing.get(k, 0) + WEIGHT_SPECT * final_score_spect.get(k, 0) + WEIGHT_MFCC * final_score_mfcc.get(k, 0) + WEIGHT_TEXT * final_scores_text.get(k, 0) + WEIGHT_COLOR_HIST * final_scores_colorhist.get(k, 0))/SUM_WEIGHTS for k in set(final_score_energy) | set(final_score_zero_crossing) | set(final_score_spect) | set(final_score_mfcc) | set(final_scores_text) | set(final_scores_colorhist)}
		NUM_OF_FEATURES = 7 # +1 to avoid dividing by 0
		for k in sum_of_scores:
			count = 0
			if k in final_score_energy:
				count += 1
			if k in final_score_zero_crossing:
				count += 1
			if k in final_score_spect:
				count += 1
			if k in final_score_mfcc:
				count += 1
			if k in final_scores_text:
				count += 1
			sum_of_scores[k] = sum_of_scores[k]/(NUM_OF_FEATURES-count) # combo boost
		return self.normalize_score(sum_of_scores)


	def normalize_score(self, scores_feature):
		"""Normalise the dictionary scores_features to that sum of values = 1.0"""
		if len(scores_feature) == 0:
			return scores_feature
		factor = 1.0/sum(scores_feature.itervalues())
		for k in scores_feature:
			scores_feature[k] = scores_feature[k] * factor
		return scores_feature

	def show_venue_category(self, FILENAME):
		allframes = os.listdir("dataset_vine/vine/validation/frame/")
		self.filename = str(FILENAME)
		self.videoname = self.filename.strip().split("/")[-1].replace(".mp4","")
		# If tags exist, load them into the searchbar
		if self.videoname in self.tags_index:
			tags = " ".join(self.tags_index[self.videoname])

		self.async_result_extract_frame = self.pool.apply_async(self.extract_frame_async, ())

		print "videoname", self.videoname
		error_file = open("errors_query.txt", "a")

		audio_storing_path = "dataset_vine/vine/validation/audio/" + self.videoname + ".wav"
		try:
			# Extract Audio
			clip = mp.VideoFileClip(self.filename)
			clip.audio.write_audiofile(audio_storing_path)
		except:
			error_file.write(self.filename + "\n")
		
		# Extract audio features
		self.query_feature_mfcc, self.query_feature_spect, self.query_feature_zerocrossing, self.query_feature_energy = extract_acoustic.getAcousticFeatures(audio_storing_path)

		self.frames = self.async_result_extract_frame.get()
		self.async_result_color_hist = self.pool.apply_async(self.search_color_hist_in_background, (self.frames, ) ) # tuple of args for foo

		# Perform Text Search
		queryTags = str(" ".join(self.tags_index[self.videoname]))
		self.weights["textWeight"] = 3
		scores_text = []
		if len(queryTags) > 0:
			scores_text = search_text_index(queryTags, self.limit) # Will return a min heap (smaller is better)
	
		self.labels, self.features_energy = self.async_result_energy.get()
		self.labels, self.features_zero_crossing = self.async_result_zero_crossing.get()
		self.labels, self.features_spect = self.async_result_spect.get()
		self.labels, self.features_mfcc = self.async_result_mfcc.get()

		scores_energy = {}
		for i in range(0, len(self.features_energy)):
			score_energy = acoustic_searcher.array_sum(self.features_energy[i], self.query_feature_energy)
			video_id = self.labels[i]
			scores_energy[video_id] = score_energy

		scores_zero_crossing = {}
		for i in range(0, len(self.features_zero_crossing)):
			score_zero_crossing = acoustic_searcher.array_sum(self.features_zero_crossing[i], self.query_feature_zerocrossing)
			video_id = self.labels[i]
			scores_zero_crossing[video_id] = score_zero_crossing		

		scores_spect = {}
		for i in range(0, len(self.features_spect)):
			score_spect = acoustic_searcher.matrix_sum(self.features_spect[i], self.query_feature_spect)
			video_id = self.labels[i]
			scores_spect[video_id] = score_spect

		scores_mfcc = {}
		for i in range(0, len(self.features_mfcc)):
			score_mfcc = acoustic_searcher.matrix_sum(self.features_mfcc[i], self.query_feature_mfcc)
			video_id = self.labels[i]
			scores_mfcc[video_id] = score_mfcc

		WEIGHT_ENERGY = self.weights_acoustic["energyWeight"]
		WEIGHT_ZERO_CROSSING= self.weights_acoustic["zeroCrossingWeight"]
		WEIGHT_SPECT = self.weights_acoustic["spectWeight"]
		WEIGHT_MFCC = self.weights_acoustic["mfccWeight"]
		WEIGHT_COLOR_HIST = self.weights["colorHistWeight"]
		if len(queryTags) > 0:
			WEIGHT_TEXT = self.weights["textWeight"]
		else:
			WEIGHT_TEXT = 0

		SUM_WEIGHTS = WEIGHT_ENERGY + WEIGHT_ZERO_CROSSING + WEIGHT_SPECT + WEIGHT_MFCC + WEIGHT_TEXT + WEIGHT_COLOR_HIST

		final_scores_cat = {}
		final_score_energy = {}
		if self.statesConfiguration["energy"] == True:
			final_scores_cat = self.add_to_final_scores(scores_energy, final_scores_cat, WEIGHT_ENERGY/SUM_WEIGHTS)
			final_score_energy = self.normalize_score(final_scores_cat)
			print "Energy: ", final_score_energy
			print ""

		final_scores_cat = {}
		final_score_zero_crossing = {}
		if self.statesConfiguration["zeroCrossing"] == True:
			final_scores_cat = self.add_to_final_scores(scores_zero_crossing, final_scores_cat, WEIGHT_ZERO_CROSSING/SUM_WEIGHTS)
			final_score_zero_crossing = self.normalize_score(final_scores_cat)
			print "Zero Crossing: ", final_score_zero_crossing 
			print ""

		final_scores_cat = {}
		final_score_spect = {}
		if self.statesConfiguration["spect"] == True:
			final_scores_cat = self.add_to_final_scores(scores_spect, final_scores_cat, WEIGHT_SPECT/SUM_WEIGHTS)
			final_score_spect = self.normalize_score(final_scores_cat)
			print "SPECT: ", final_score_spect
			print ""

		final_scores_cat = {}
		final_score_mfcc = {}
		if self.statesConfiguration["mfcc"] == True:
			final_scores_cat = self.add_to_final_scores(scores_mfcc, final_scores_cat, WEIGHT_MFCC/SUM_WEIGHTS)
			final_score_mfcc = self.normalize_score(final_scores_cat)
			print "MFCC: ", final_score_mfcc
			print ""

		final_scores_cat = {}
		final_scores_colorhist = {}
		if self.statesConfiguration["colorHist"] == True:
			print "in here"
			scores_color_hist = self.async_result_color_hist.get()
			print "Frames are " ,self.frames
			final_scores_cat = self.add_hist_to_final_scores(scores_color_hist, final_scores_cat, WEIGHT_COLOR_HIST/SUM_WEIGHTS)
			final_scores_colorhist = self.normalize_score(final_scores_cat)
			print "Color Hist: ", final_scores_colorhist
			print ""

		final_scores_cat = {}
		final_scores_text = {}
		if len(queryTags) > 0:
			final_scores_cat = self.add_text_to_final_scores(scores_text, final_scores_cat, WEIGHT_TEXT/SUM_WEIGHTS)
			final_scores_text = self.normalize_score(final_scores_cat)
			print "Text: ", final_scores_text
			print ""

		fused_scores = self.fuse_scores(final_score_energy, final_score_zero_crossing, final_score_spect, final_score_mfcc, final_scores_text, final_scores_colorhist)
		print "Final: ", fused_scores
		print ""

		if len(fused_scores) != 0:
			venue_texts = heapq.nlargest(3, fused_scores, key=fused_scores.get)
		else:
			venue_texts.append("?")

		count = 0
		if self.dict_videoid_name(self.videoname) in venue_texts:
			count += 1
		print count
			

	def extract_frame_async(self):
		"""Extract frames"""
		existing_frames = glob.glob("dataset_vine/vine/validation/frame/"  + self.videoname + "-" + "*")
		if len(existing_frames) > 0:
			return existing_frames
		frame_storing_path = "dataset_vine/vine/validation/frame/"  + self.videoname + "-"
		vid_cap = cv2.VideoCapture(self.filename) # Open the video file	
		getKeyFrames(vid_cap, frame_storing_path)
		vid_cap.release()
		return glob.glob("dataset_vine/vine/validation/frame/"  + self.videoname + "-" + "*")
		


	def build_tags_index(self):
		# Read in query tags
		test_tags = os.path.join(os.path.dirname(__file__), "dataset_vine", "vine-desc-validation.txt")
		try:
			file_train_tags = open(test_tags, "r")
		except IOError:
			print "Cannot open vine-desc-validation.txt"
		else:
			self.tags_index = index_tags_normal(file_train_tags)
			file_train_tags.close()
		print self.tags_index

	# def compare(self, final_results):
	# 	"""For testing F1"""
	# 	category_name = "cat" # Put category here
	#  	path = os.path.join(os.path.dirname(__file__), "ImageData", "train", "data", category_name)
	#  	img_dir = glob.glob(path + "/*.jpg")
	#  	count = 0

	#  	for imagePath in img_dir:
	# 		imageID = imagePath[imagePath.rfind("/") + 1:]

	# 		for score, img in final_results:
	# 			if img == imageID:
	# 				count += 1
	# 				break
	# 	print count

def main():
	test = TestScript(search_path='../dataset_vine/vine/validation/video/', frame_storing_path='../dataset_vine/vine/validation/frame/', validation_videos_path='dataset_vine/vine/validation/video/*.mp4')
	sys.exit()

main()