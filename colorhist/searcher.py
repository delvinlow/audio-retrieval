# import the necessary packages
import numpy as np
import cv2
import cPickle as pickle

class Searcher:
	def __init__(self, indexPath):
		# store our index path
		self.indexPath = indexPath
		f = open(self.indexPath, "r")
		self.dict_features =  pickle.load(f)
		f.close()

	def search(self, queryFeatures, limit = 10):
		# initialize our dictionary of results
		results = {}
		queryFeatures = np.array(queryFeatures)
		# loop over the rows in the index
		for img_id in self.dict_features:
			# parse out the image ID and features, then compute the
			# chi-squared distance between the features in our index
			# and our query features
			features = [float(x) for x in self.dict_features[img_id]]
			d = self.chi2_distance(np.array(features), queryFeatures)

			# now that we have the distance between the two feature
			# vectors, we can udpate the results dictionary -- the
			# key is the current image ID in the index and the
			# value is the distance we just computed, representing
			# how 'similar' the image in the index is to our query
			results[img_id] = d

		# close the reader

		# sort our results, so that the smaller distances (i.e. the
		# more relevant images are at the front of the list)
		results = sorted([(v, k) for (k, v) in results.items()])

		# return our (limited) results
		return results[:limit]

	def chi2_distance(self, histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])

		# return the chi-squared distance
		return d