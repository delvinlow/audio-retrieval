#! /usr/bin/env
# -*- coding: latin2 -*- 
import cv2
# from colorhist.colordescriptor import ColorDescriptor
# from colorhist.searcher import Searcher
# from textsearch.index_text import build_normal_index
# from textsearch.index_text import index_tags_normal
# from textsearch.search_text import search_text_index
# from SIFT.search_sift import SIFTandBOW
# from fuse_scores import fuse_scores

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import *

import sys
import os
import design
# import glob
# from multiprocessing.pool import ThreadPool
# from deeplearning.classify_image import run_inference_on_image
# from deeplearning.classify_image import run_inference_on_query_image
# from deeplearning.classify_image import create_session
# from deeplearning.classify_image import create_graph
# from deeplearning.search_deep_learning import DeepLearningSearcher
# from imageconcept.search_concept import search_concept
# from time import sleep


class Window(QtGui.QMainWindow, design.Ui_MainWindow):
	def __init__(self, search_path, frame_storing_path):
		self.search_path = search_path
		self.frame_storing_path = frame_storing_path
		# self.limit = 100
		# self.searcher = Searcher("colorhist/index_color_hist.txt")
		super(Window, self).__init__()
		self.setupUi(self)
		
		self.home()
		# self.sab = SIFTandBOW(True)
		# self.build_index()
		# self.statesConfiguration = {"colorHist": True, "visualConcept": True, "visualKeyword": True, "deepLearning": True}
		# self.weights = {"colorHistWeight": 3.0, "vkWeight": 1.0, "vcWeight": 2.0, "textWeight": 1.0, "dpLearnWeight": 3.0} #total = 5.0
		# self.deep_learner_searcher = DeepLearningSearcher("deeplearning/output_probabilities.txt")
		# # self.deepLearningGraph = create_graph()
		# self.deepLearningSession, self.softmax_tensor = create_session()
		# self.pool = ThreadPool(processes=4)
		# self.cd = ColorDescriptor((8, 12, 3))

	def home(self):
		"""Specific to page. Connect the buttons to functions"""
		self.btn_picker.clicked.connect(self.choose_video)
		self.btn_search.clicked.connect(self.show_venue_category)
		self.btn_quit.clicked.connect(self.close_application)
		self.btn_reset.clicked.connect(self.clear_results)

		# self.checkBoxColorHist.stateChanged.connect(self.state_changed)
		# self.checkBoxVisualConcept.stateChanged.connect(self.state_changed)
		# self.checkBoxVisualKeyword.stateChanged.connect(self.state_changed)
		# self.checkBoxDeepLearning.stateChanged.connect(self.state_changed)

		# self.doubleSpinBoxColorHist.valueChanged.connect(self.value_changed)
		# self.doubleSpinBoxVisualConcept.valueChanged.connect(self.value_changed)
		# self.doubleSpinBoxVisualKeyword.valueChanged.connect(self.value_changed)
		# self.doubleSpinBoxDeepLearning.valueChanged.connect(self.value_changed)

		self.show()

	# def value_changed(self):
	# 	self.weights = {"colorHistWeight": self.doubleSpinBoxColorHist.value(), 
	# 	 "vkWeight": self.doubleSpinBoxVisualKeyword.value(),
	# 	 "vcWeight": self.doubleSpinBoxVisualConcept.value(), 
	# 	 "textWeight": self.doubleSpinBoxText.value(), 
	# 	 "dpLearnWeight": self.doubleSpinBoxDeepLearning.value()} #total = 10
	# 	print self.weights

	# def state_changed(self):
	# 	if self.checkBoxColorHist.isChecked():
	# 		self.statesConfiguration["colorHist"] = True
	# 		self.doubleSpinBoxColorHist.setEnabled(True)
	# 	else:
	# 		self.statesConfiguration["colorHist"] = False
	# 		self.doubleSpinBoxColorHist.setEnabled(False)

	# 	if self.checkBoxVisualConcept.isChecked():
	# 		self.statesConfiguration["visualConcept"] = True
	# 		self.doubleSpinBoxVisualConcept.setEnabled(True)
	# 	else:
	# 		self.statesConfiguration["visualConcept"] = False
	# 		self.doubleSpinBoxVisualConcept.setEnabled(False)
	# 	if self.checkBoxVisualKeyword.isChecked():
	# 		self.statesConfiguration["visualKeyword"] = True
	# 		self.doubleSpinBoxVisualKeyword.setEnabled(True)
	# 	else:
	# 		self.statesConfiguration["visualKeyword"] = False
	# 		self.doubleSpinBoxVisualKeyword.setEnabled(False)

	# 	if self.checkBoxDeepLearning.isChecked():
	# 		self.statesConfiguration["deepLearning"] = True
	# 		self.doubleSpinBoxDeepLearning.setEnabled(True)
	# 	else:
	# 		self.statesConfiguration["deepLearning"] = False
	# 		self.doubleSpinBoxDeepLearning.setEnabled(False)

	# 	print self.statesConfiguration

	def closeEvent(self, event):
		event.ignore()
		self.close_application()

	def close_application(self):
		choice = QtGui.QMessageBox.question(self, "Quit?", 
			"Are you sure to quit?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)

		if choice == QtGui.QMessageBox.Yes:
			sys.exit()
		else:
			pass

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

	def show_venue_category(self):
		print "do whatever here"

		if self.columns == 0:
			print("Please extract the key frames for the selected video first!!!")
		else:
			# Please note that, you need to write your own classifier to estimate the venue category to show blow.
			if self.videoname == '1':
			   venue_text = "Home"
			elif self.videoname == '2':
			    venue_text = 'Bridge'
			elif self.videoname == '4':
			    venue_text = 'Park'




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

		# venue_img = Image.open("venue_background.jpg")
		# draw = ImageDraw.Draw(venue_img)

		# font = ImageFont.truetype("/Library/Fonts/Arial.ttf",size=66)

		self.pad_rows_with_dummy_images()
		pass
		

	def choose_video(self):
		self.tags_search.setText("")
		self.filename = QtGui.QFileDialog.getOpenFileName(self, "Open Video", os.path.dirname(__file__),"Videos (*.mp4)")
		# base_img_id = os.path.splitext(os.path.basename(str(self.filename)))
		# base_img_id = "".join(base_img_id)


		allframes = os.listdir(self.frame_storing_path)
		self.filename = str(self.filename)
		self.videoname = self.filename.strip().split("/")[-1].replace(".mp4","")

		self.frames = []
		for frame in allframes:
			if self.videoname +"-frame" in frame:
				self.frames.append(self.frame_storing_path + frame)

		self.columns = len(self.frames)
		image_count = 0

		MAX_COLUMNS = 8
		row = self.listWidgetResults.currentRow()

		row += self.columns/MAX_COLUMNS + 1

		if self.columns == 0:
			self.frames.append("none.png")
			print("Please extract the key frames for the selected video first!!!")
			self.columns = 1

		for frame in self.frames:

			r, c = divmod(image_count, self.columns)
			try:
				# im = Image.open(frame)
				# resized = im.resize((100, 100), Image.ANTIALIAS)

				# fullpath = glob.glob(os.path.join(os.path.curdir, "ImageData", "train", "data", "*", img_id) )[0]
				img_widget_icon = QListWidgetItem(QIcon(frame), "")
				self.listWidgetResults.addItem(img_widget_icon)

        #         tkimage = ImageTk.PhotoImage(resized)

        #         myvar = Label(self.query_img_frame, image=tkimage)
        #         myvar.image = tkimage
        #         myvar.grid(row=r, column=c)

				image_count += 1
        #         self.lastR = r
        #         self.lastC = c
			except Exception, e:
				continue

        # self.query_img_frame.mainloop()



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
	
	# 	# Perform Text Search
	# 	queryTags = str(self.tags_search.text().toLatin1())
	# 	self.weights["textWeight"] = self.doubleSpinBoxText.value()
	# 	results_text = []
	# 	if len(queryTags) > 0:
	# 		results_text = search_text_index(queryTags, self.limit) # Will return a min heap (smaller is better)

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

	# def build_index(self):
	# 	# Read in query tags
	# 	test_tags = os.path.join(os.path.dirname(__file__), "ImageData", "test", "test_text_tags.txt")
	# 	try:
	# 	 	file_train_tags = open(test_tags, "r")
	#  	except IOError:
	#  		print "Cannot open test_text_tags.txt"
	#  	else:
	#  		self.tags_index = index_tags_normal(file_train_tags)
	#  		file_train_tags.close()
	#  		# print self.tags_index

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
	GUI = Window(search_path='../data/video/', frame_storing_path='../data/frame/')
	sys.exit(app.exec_())

main()