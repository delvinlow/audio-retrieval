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

from textsearch.index_text import index_tags_normal
from textsearch.search_text import search_text_index

from preprocessing.extract_frame import getKeyFrames

# from time import sleep

class Window(QtGui.QMainWindow, design.Ui_MainWindow):
	def __init__(self, search_path, frame_storing_path):
		self.frames = []
		self.pool = ThreadPool(processes=4)
		self.pool_extract = ThreadPool(processes=4)

		self.init_audio_index()
		self.build_venues_index()
		self.search_path = search_path
		self.frame_storing_path = frame_storing_path
		self.limit = 100
		
		self.colorhist_searcher = Searcher("colorhist/index_color_hist.csv")
		self.cd = ColorDescriptor((8, 12, 3))

		super(Window, self).__init__()
		self.setupUi(self)
		
		self.home()
		self.build_tags_index()
		self.statesConfiguration = {"colorHist": True, "visualConcept": True, "text": True, "energy": True, "zeroCrossing": True, "spect": True, "mfcc" : True}
		self.weights = {"colorHistWeight": self.doubleSpinBoxColorHist.value(), 
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

		self.checkBox_energy.stateChanged.connect(self.state_changed)
		self.checkBox_zerocrossing.stateChanged.connect(self.state_changed)
		self.checkBox_spect.stateChanged.connect(self.state_changed)
		self.checkBox_mfcc.stateChanged.connect(self.state_changed)

		self.doubleSpinBoxColorHist.valueChanged.connect(self.value_changed)

		self.doubleSpinBoxEnergy.valueChanged.connect(self.value_changed)
		self.doubleSpinBoxZeroCrossing.valueChanged.connect(self.value_changed)
		self.doubleSpinBoxSpect.valueChanged.connect(self.value_changed)
		self.doubleSpinBoxMFCC.valueChanged.connect(self.value_changed)

		self.show()


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
			self.colorhist_searcher.f.close()
			sys.exit()

	def search_color_hist_in_background(self):
		print "THAT ", self.frames
		middleLen = len(self.frames) / 2
		query = cv2.imread(self.frames[middleLen])
		# load the query image and describe it
		self.queryfeatures = self.cd.describe(query)
		results = self.colorhist_searcher.search(self.queryfeatures, self.limit)
		return results


	def pad_rows_with_dummy_images(self):
		count = self.listWidgetResults.count()
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
		WEIGHT_TEXT = self.weights["textWeight"]
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
		factor=1.0/sum(scores_feature.itervalues())
		for k in scores_feature:
			scores_feature[k] = scores_feature[k] * factor
		return scores_feature

	def show_venue_category(self):
		# Perform Text Search
		queryTags = str(self.tags_search.text().toLatin1())
		self.weights["textWeight"] = self.doubleSpinBoxText.value()
		scores_text = []
		if len(queryTags) > 0:
			scores_text = search_text_index(queryTags, self.limit) # Will return a min heap (smaller is better)
	
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
        	WEIGHT_COLOR_HIST = self.weights["colorHistWeight"]
        	WEIGHT_TEXT = self.weights["textWeight"]

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

			self.frames = self.async_result_extract_frame.get()
			final_scores_cat = {}
			final_scores_colorhist = {}

			if self.statesConfiguration["colorHist"] == True:
				scores_color_hist = self.async_result_color_hist.get()
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


		for venue_text in venue_texts:
			tooltip = venue_text + "\n" + "Probability: " + ('%.1f%%' % (float(fused_scores[venue_text])*100))
			pixmap = QPixmap("venue_background.jpg")
			# pixmap.fill(Qt.white)
			font = QFont("Arial", 30)
			painter = QPainter()
			painter.begin(pixmap)
			painter.setFont(font)
			painter.drawText(32, 75, venue_text)
			painter.end()

			venue_img_icon = QListWidgetItem(QIcon(pixmap), "")
			venue_img_icon.setToolTip(tooltip)
			self.listWidgetResults.addItem(venue_img_icon)
		self.pad_rows_with_dummy_images()
	# 	self.compare(final_results) # for testing

	def extract_frame_async(self):
		"""Extract frames"""
		frame_storing_path = "dataset_vine/vine/validation/frame/"  + self.videoname + "-"
		vid_cap = cv2.VideoCapture(self.filename) # Open the video file	
		getKeyFrames(vid_cap, frame_storing_path)
		vid_cap.release()
		return glob.glob("dataset_vine/vine/validation/frame/"  + self.videoname + "-" + "*")
		

	def choose_video(self):
		"""Set a video from File Dialog"""
		self.tags_search.setText("")
		self.filename = QtGui.QFileDialog.getOpenFileName(self, "Open Video", os.path.dirname(__file__),"Videos (*.mp4)")

		allframes = os.listdir("dataset_vine/vine/validation/frame/")
		self.filename = str(self.filename)
		self.videoname = self.filename.strip().split("/")[-1].replace(".mp4","")

		self.async_result_extract_frame = self.pool_extract.apply_async(self.extract_frame_async, ())

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
		print "THIS ", self.frames
		# Color Histogram -process query image to feature vector
		self.async_result_color_hist = self.pool_extract.apply_async(self.search_color_hist_in_background, () ) # tuple of args for foo
		print self.search_color_hist_in_background()

		self.columns = len(self.frames)
		image_count = 0
		MAX_COLUMNS = 8
		if self.columns == 0:
			self.frames.append("none.png")
			print("Please extract the key frames for the selected video first!!!")
			self.columns = 1

		# print self.frames
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

		self.raise_()
		self.activateWindow()
		self.btn_search.setFocus()



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