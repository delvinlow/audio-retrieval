#! /usr/bin/env
# -*- coding: latin2 -*- 

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import *

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

# from textsearch.index_text import build_normal_index
from textsearch.index_text import index_tags_normal
from textsearch.search_text import search_text_index
# from SIFT.search_sift import SIFTandBOW
# from fuse_scores import fuse_scores
# from deeplearning.classify_image import run_inference_on_image
# from deeplearning.classify_image import run_inference_on_query_image
# from deeplearning.classify_image import create_session
# from deeplearning.classify_image import create_graph
# from deeplearning.search_deep_learning import DeepLearningSearcher
# from imageconcept.search_concept import search_concept
# from time import sleep

from preprocessing.extract_frame import getKeyFrames


class Window(QtGui.QMainWindow, design.Ui_MainWindow):
	def __init__(self, search_path, frame_storing_path):
		self.pool = ThreadPool(processes=4)
		self.pool_extract = ThreadPool(processes=4)

		self.init_audio_index()
		self.build_venues_index()
		self.search_path = search_path
		self.frame_storing_path = frame_storing_path
		self.limit = 100
		# self.searcher = Searcher("colorhist/index_color_hist.txt")
		super(Window, self).__init__()
		self.setupUi(self)
		
		self.home()
		# self.sab = SIFTandBOW(True)
		# self.build_index()
		self.build_tags_index()
		print self.tags_index
		self.statesConfiguration = {"colorHist": True, "visualConcept": True, "text": True, "energy": True, "zeroCrossing": True, "spect": True, "mfcc" : True}
		self.weights = {"colorHistWeight": self.doubleSpinBoxColorHist.value(), 
		 "vkWeight": self.doubleSpinBoxVisualKeyword.value(),
		 "textWeight": self.doubleSpinBoxText.value()} 
		print self.weights

		self.weights_acoustic = {"energyWeight": self.doubleSpinBoxEnergy.value(), 
		 "zeroCrossingWeight": self.doubleSpinBoxZeroCrossing.value(),
		 "spectWeight": self.doubleSpinBoxSpect.value(), "mfccWeight": self.doubleSpinBoxMFCC.value()} #total = 10
		print self.weights_acoustic
		

	def home(self):
		"""Specific to page. Connect the buttons to functions"""
		self.btn_picker.clicked.connect(self.choose_video)
		self.btn_search.clicked.connect(self.show_venue_category)
		self.btn_quit.clicked.connect(self.close_application)
		self.btn_reset.clicked.connect(self.clear_results)

		self.checkBoxColorHist.stateChanged.connect(self.state_changed)
		self.checkBoxVisualKeyword.stateChanged.connect(self.state_changed)

		self.checkBox_energy.stateChanged.connect(self.state_changed)
		self.checkBox_zerocrossing.stateChanged.connect(self.state_changed)
		self.checkBox_spect.stateChanged.connect(self.state_changed)
		self.checkBox_mfcc.stateChanged.connect(self.state_changed)

		self.doubleSpinBoxColorHist.valueChanged.connect(self.value_changed)
		self.doubleSpinBoxVisualKeyword.valueChanged.connect(self.value_changed)

		self.doubleSpinBoxEnergy.valueChanged.connect(self.value_changed)
		self.doubleSpinBoxZeroCrossing.valueChanged.connect(self.value_changed)
		self.doubleSpinBoxSpect.valueChanged.connect(self.value_changed)
		self.doubleSpinBoxMFCC.valueChanged.connect(self.value_changed)

		self.show()

		# self.cd = ColorDescriptor((8, 12, 3))

	def build_venues_index(self):
		"""Builds two dictionaries for venue_id, video_id and category name mapping"""
		file_venueid_venuename = open("dataset_vine/venue-name.txt", "r") # eg: 1	City
		file_videoid_venueid = open("dataset_vine/vine-venue-training.txt", "r") # eg: 1000110881755082752	1
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
		file_venueid_venuename.close()
		file_videoid_venueid.close()

	def init_audio_index(self):
		"""Read in the 3000 acoustic features files in the 4 folders into memory during __init__ """
		self.async_result_energy = self.pool.apply_async(acoustic_searcher.build_index, ("feature/acoustic/feature_energy/",))
		self.async_result_mfcc = self.pool.apply_async(acoustic_searcher.build_index, ("feature/acoustic/feature_mfcc/",))
		self.async_result_zero_crossing = self.pool.apply_async(acoustic_searcher.build_index, ("feature/acoustic/feature_zero_crossing/",))
		self.async_result_spect = self.pool.apply_async(acoustic_searcher.build_index, ("feature/acoustic/feature_spect/",))


	def value_changed(self):
		"""Modify weights if values changed in any of the spinBoxes"""
		self.weights = {"colorHistWeight": self.doubleSpinBoxColorHist.value(), 
		 "vkWeight": self.doubleSpinBoxVisualKeyword.value(),
		 "textWeight": self.doubleSpinBoxText.value()} #total = 10
		print self.weights

		self.weights_acoustic = {"energyWeight": self.doubleSpinBoxEnergy.value(), 
		 "zeroCrossingWeight": self.doubleSpinBoxZeroCrossing.value(),
		 "spectWeight": self.doubleSpinBoxSpect.value(), "mfccWeight": self.doubleSpinBoxMFCC.value()} #total = 10
		print self.weights_acoustic

	def state_changed(self):
		"""Update configuration when checkboxes are clicked"""
		if self.checkBoxColorHist.isChecked():
			self.statesConfiguration["colorHist"] = True
			self.doubleSpinBoxColorHist.setEnabled(True)
		else:
			self.statesConfiguration["colorHist"] = False
			self.doubleSpinBoxColorHist.setEnabled(False)

		if self.checkBoxVisualKeyword.isChecked():
			self.statesConfiguration["visualKeyword"] = True
			self.doubleSpinBoxVisualKeyword.setEnabled(True)
		else:
			self.statesConfiguration["visualKeyword"] = False
			self.doubleSpinBoxVisualKeyword.setEnabled(False)

		# Acoustic
		if self.checkBox_energy.isChecked():
			self.statesConfiguration["energy"] = True
			self.doubleSpinBoxEnergy.setEnabled(True)
		else:
			self.statesConfiguration["energy"] = False
			self.doubleSpinBoxEnergy.setEnabled(False)

		if self.checkBox_zerocrossing.isChecked():
			self.statesConfiguration["zeroCrossing"] = True
			self.doubleSpinBoxZeroCrossing.setEnabled(True)
		else:
			self.statesConfiguration["zeroCrossing"] = False
			self.doubleSpinBoxZeroCrossing.setEnabled(False)

		if self.checkBox_spect.isChecked():
			self.statesConfiguration["spect"] = True
			self.doubleSpinBoxSpect.setEnabled(True)
		else:
			self.statesConfiguration["spect"] = False
			self.doubleSpinBoxSpect.setEnabled(False)

		if self.checkBox_mfcc.isChecked():
			self.statesConfiguration["mfcc"] = True
			self.doubleSpinBoxMFCC.setEnabled(True)
		else:
			self.statesConfiguration["mfcc"] = False
			self.doubleSpinBoxMFCC.setEnabled(False)

		print self.statesConfiguration

	def closeEvent(self, event):
		"""Handle closing of app"""
		event.ignore()
		self.close_application()

	def close_application(self):
		choice = QtGui.QMessageBox.question(self, "Quit?", 
			"Are you sure to quit?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
		if choice == QtGui.QMessageBox.Yes:
			sys.exit()

	# def search_color_hist_in_background(self):
	# 	query = cv2.imread(self.filename)
	# 	# load the query image and describe it
	# 	self.queryfeatures = self.cd.describe(query)
	# 	return self.searcher.search(self.queryfeatures, self.limit)

	# def search_visual_concept_in_background(self):
	# 	path = os.path.join(os.path.curdir, "ImageData", "train", "data")
	# 	return search_concept(self.filename, path, self.limit)

	# def search_deep_learn_in_background(self):
	# 	self.queryProbability = run_inference_on_query_image(self.deepLearningSession, self.filename, self.softmax_tensor)
	# 	return self.deep_learner_searcher.search_deep_learn(self.queryProbability, self.limit)

	# def search_sift_in_background(self):
	# 	query = cv2.imread(self.filename)
	# 	self.hist_sift_query = self.sab.histogramBow(query)
	# 	return self.sab.search(self.hist_sift_query, self.limit)


	def pad_rows_with_dummy_images(self):
		count = self.listWidgetResults.count()
		print "count ", count
		MAX_COLUMNS = 8
		if count < MAX_COLUMNS:
			remainder = MAX_COLUMNS - count 
		else:
			remainder = MAX_COLUMNS - (count % MAX_COLUMNS) # -1 for classification

		for frame in range(0, remainder):
			img_widget_icon = QListWidgetItem(QIcon("transparent.png"), "")
			self.listWidgetResults.addItem(img_widget_icon)

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
		for (score, video_id) in self.get_top_scorers(scores_feature):
			venue_name = self.dict_videoid_name[video_id]
			current_score = final_scores_cat.setdefault(venue_name, 0)
			final_scores_cat[venue_name] = current_score + remaining_points * weight
			remaining_points -= 1
		return final_scores_cat

	def add_text_to_final_scores(self, scores_feature, final_scores_cat, weight):
		"""Add the weighted feature score of top 16 video to the final scores"""
		remaining_points = self.limit
		for (score, video_id) in scores_feature:
			venue_name = self.dict_videoid_name[video_id]
			current_score = final_scores_cat.setdefault(venue_name, 0)
			final_scores_cat[venue_name] = -current_score + remaining_points * weight
			remaining_points -= 1
		return final_scores_cat

	def show_venue_category(self):
		# Perform Text Search
		queryTags = str(self.tags_search.text().toLatin1())
		self.weights["textWeight"] = self.doubleSpinBoxText.value()
		scores_text = []
		if len(queryTags) > 0:
			scores_text = search_text_index(queryTags, self.limit) # Will return a min heap (smaller is better)
			print scores_text

		self.labels, self.features_energy = self.async_result_energy.get()
		self.labels, self.features_zero_crossing = self.async_result_zero_crossing.get()
		self.labels, self.features_spect = self.async_result_spect.get()
		self.labels, self.features_mfcc = self.async_result_mfcc.get()

		if self.columns == 0:
			print("Please extract the key frames for the selected video first!!!")
		else:
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

        	WEIGHT_TEXT = self.weights["textWeight"]
        	SUM_WEIGHTS = WEIGHT_ENERGY + WEIGHT_ZERO_CROSSING + WEIGHT_SPECT + WEIGHT_MFCC + WEIGHT_TEXT

        	final_scores_cat = {}
        	final_scores_cat = self.add_to_final_scores(scores_energy, final_scores_cat, WEIGHT_ENERGY/SUM_WEIGHTS)
        	print "Energy: ", final_scores_cat
        	print ""
        	final_scores_cat = {}
        	final_scores_cat = self.add_to_final_scores(scores_zero_crossing, final_scores_cat, WEIGHT_ZERO_CROSSING/SUM_WEIGHTS)
        	print "Zero Crossing: ", final_scores_cat
        	print ""
        	final_scores_cat = {}
        	final_scores_cat = self.add_to_final_scores(scores_spect, final_scores_cat, WEIGHT_SPECT/SUM_WEIGHTS)
        	print "SPECT: ", final_scores_cat
        	print ""
        	final_scores_cat = {}
        	final_scores_cat = self.add_to_final_scores(scores_mfcc, final_scores_cat, WEIGHT_MFCC/SUM_WEIGHTS)
        	print "MFCC: ", final_scores_cat
        	print ""

        	final_scores_cat = {}
        	final_scores_cat = self.add_text_to_final_scores(scores_text, final_scores_cat, WEIGHT_TEXT/SUM_WEIGHTS)
        	print "Text: ", final_scores_cat

		if len(final_scores_cat) != 0:
			venue_text = max(final_scores_cat, key=lambda k: final_scores_cat[k])
		else:
			venue_text = "?"

		pixmap = QPixmap("venue_background.jpg")
		# pixmap.fill(Qt.white)
		font = QFont("Arial", 30)
		painter = QPainter()
		painter.begin(pixmap)
		painter.setFont(font)
		painter.drawText(32, 75, venue_text)
		painter.end()

		venue_img_icon = QListWidgetItem(QIcon(pixmap), "")
		self.listWidgetResults.addItem(venue_img_icon)
		self.pad_rows_with_dummy_images()

	def extract_frame_async(self):
		"""Extract frames"""
		frame_storing_path = "dataset_vine/vine/validation/frame/"  + self.videoname + "-"
		vid_cap = cv2.VideoCapture(self.filename) # Open the video file	
		self.frames = getKeyFrames(vid_cap, frame_storing_path)
		vid_cap.release()
		return glob.glob("dataset_vine/vine/validation/frame/"  + self.videoname + "-" + "*")
		

	def choose_video(self):
		"""Set a video from File Dialog"""
		self.tags_search.setText("")
		self.filename = QtGui.QFileDialog.getOpenFileName(self, "Open Video", os.path.dirname(__file__),"Videos (*.mp4)")

		allframes = os.listdir("dataset_vine/vine/validation/frame/")
		self.filename = str(self.filename)
		self.videoname = self.filename.strip().split("/")[-1].replace(".mp4","")
		self.frames = []

		self.async_result_extract_frame = self.pool_extract.apply_async(self.extract_frame_async, ())

		print "videoname", self.videoname
		error_file = open("errors_query.txt", "a")

		#1. Extract Audio Clips
		audio_storing_path = "dataset_vine/vine/validation/audio/" + self.videoname + ".wav"
		print audio_storing_path
		print self.filename
		try:
			# Extract Audio
			clip = mp.VideoFileClip(self.filename)
			clip.audio.write_audiofile(audio_storing_path)
		except:
			error_file.write(self.filename + "\n")
		

		self.query_feature_mfcc, self.query_feature_spect, self.query_feature_zerocrossing, self.query_feature_energy = extract_acoustic.getAcousticFeatures(audio_storing_path)

		self.frames = self.async_result_extract_frame.get()
		self.columns = len(self.frames)
		image_count = 0
		MAX_COLUMNS = 8
		row = self.listWidgetResults.currentRow()
		row += self.columns/MAX_COLUMNS + 1
		if self.columns == 0:
			self.frames.append("none.png")
			print("Please extract the key frames for the selected video first!!!")
			self.columns = 1

		print self.frames
		for frame in self.frames:

			r, c = divmod(image_count, self.columns)
			try:
				img_widget_icon = QListWidgetItem(QIcon(frame), "")
				self.listWidgetResults.addItem(img_widget_icon)

				image_count += 1
			except Exception, e:
				continue

		# If tags exist, load them into the searchbar
		if self.videoname in self.tags_index:
			tags = " ".join(self.tags_index[self.videoname])
			self.tags_search.setText(tags)

		# # Deep Learning
		# self.async_result_deep_learn = self.pool.apply_async(self.search_deep_learn_in_background, ())
		# # Visual Concept
		# self.async_result_visual_concept = self.pool.apply_async(self.search_visual_concept_in_background, ())
		# # Color Histogram -process query image to feature vector
		# self.filename = str(self.filename)
		# self.async_result_color_hist = self.pool.apply_async(self.search_color_hist_in_background, ()) # tuple of args for foo
		# # SIFT
		# self.async_result_sift = self.pool.apply_async(self.search_sift_in_background, ()) # tuple of args for foo
		# sleep(1.6)
		# self.label_query_img.setPixmap(QPixmap(self.filename).scaledToWidth(100) )
		# self.label_query_img.setToolTip(base_img_id)

		# # If tags exist, load them into the searchbar
		# if base_img_id in self.tags_index:
		# 	tags = " ".join(self.tags_index[base_img_id])
		# 	self.tags_search.setText(tags)
		self.raise_()
		self.activateWindow()
		self.btn_search.setFocus()
	

	# def normalize_score(self, score, results):
	# 	"""Normalises score to 0(best) -> 1(worst)"""
	# 	if len(results) == 0:
	# 		return 0.5
	# 	normalized_score = score
	# 	maxScore = results[len(results)-1][0]
	# 	minScore = results[0][0]
	# 	if maxScore == minScore: 
	# 	    normalized_score = 0.5
	# 	else:
	# 	    if len(results) > 1:
	# 	        normalized_score = (normalized_score - minScore) / (maxScore - minScore)
	# 	return normalized_score

	# def get_max(self, results):
	# 	"""Return the largest score from the results"""
	# 	if len(results) == 0:
	# 		return 1
	# 	return results[len(results)-1][0]


	# def search_image(self):
	# 	final_results = []
	

	# 	# Perform search on SIFT
	# 	results_sift = []
	# 	if self.statesConfiguration["visualKeyword"] == True:
	# 		results_sift = self.async_result_sift.get()

	# 	# Search Visual Concept
	# 	results_visual_concept = []
	# 	if self.statesConfiguration["visualConcept"] == True:
	# 		results_visual_concept = self.async_result_visual_concept.get()

	# 	# Perform the search on Color Histogram
	# 	results_color_hist = []
	# 	if self.statesConfiguration["colorHist"] == True:
	# 		results_color_hist = self.async_result_color_hist.get()

	# 	results_deep_learn = []
	# 	if self.statesConfiguration["deepLearning"] == True:
	# 		results_deep_learn = self.async_result_deep_learn.get()
		
	# 	final_results, all_candidates = fuse_scores(self.statesConfiguration, self.weights, results_color_hist, results_sift, results_text, results_deep_learn, results_visual_concept)

	# 	for (score, img_id) in final_results:
	# 		fullpath = glob.glob(os.path.join(os.path.curdir, "ImageData", "train", "data", "*", img_id) )[0]
	# 		img_widget_icon = QListWidgetItem(QIcon(fullpath), img_id)

	# 		tooltip = str(img_id) + "\n" + "Final Score: " + ('%.3f' % score) + "\n"
	# 		dict_scores = all_candidates[img_id]
			
	# 		colorHistScore = dict_scores.get("colorHist", self.get_max(results_color_hist))
	# 		siftScore = dict_scores.get("sift", self.get_max(results_sift))
	# 		visualConceptScore = dict_scores.get("visualConcept", self.get_max(results_visual_concept))
	# 		deepLearningScore = dict_scores.get("deepLearn", self.get_max(results_deep_learn))
	# 		textScore = dict_scores.get("text", self.get_max(results_text))

	# 		colorHistScore = self.normalize_score(colorHistScore, results_color_hist)
	# 		siftScore = self.normalize_score(siftScore, results_sift)
	# 		visualConceptScore = self.normalize_score(visualConceptScore, results_visual_concept)
	# 		deepLearningScore = self.normalize_score(deepLearningScore, results_deep_learn)
	# 		textScore = self.normalize_score(textScore, results_text)

	# 		tooltip += "Color Hist: " + ('%.3f' % colorHistScore) + "\n"
	# 		tooltip += "Visual Keyword: " + ('%.3f' % siftScore) + "\n"
	# 		tooltip += "Visual Concept: " + ('%.3f' % visualConceptScore) + "\n"
	# 		tooltip += "Deep Learning: " + ('%.3f' % deepLearningScore) + "\n"
	# 		if len(results_text) == 0:
	# 			tooltip += "Text: N/A"  
	# 		else:
	# 			tooltip += "Text: " + ('%.3f' % textScore)
	# 		img_widget_icon.setToolTip(tooltip)
	# 		self.listWidgetResults.addItem(img_widget_icon)
	# 	self.compare(final_results)


	def clear_results(self):
		self.listWidgetResults.clear()

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
	 		# print self.tags_index

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
	app = QtGui.QApplication(sys.argv)
	GUI = Window(search_path='../dataset_vine/vine/training/video/', frame_storing_path='../dataset_vine/vine/training/frame/')
	sys.exit(app.exec_())

main()