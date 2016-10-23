#! /usr/bin/env
# -*- coding: latin2 -*- 
# Usage: Just run "python index_text.py" provided that 
# image tags are located at /../ImageData/train/train_text_tags.txt 

import os
import nltk
import codecs
import json
# from nltk import PorterStemmer
import collections
import cPickle as pickle

FILE_TRAIN_INDEX = "train_text_tags.txt" # Contains training tags i.e. img_id -> tag1, tag2, ...
FILE_TAGS_DICT = "text_dict_tags.txt" # Contains tag1 -> ptr_start, ptr_end, doc_freq, ...
FILE_TAGS_POSTINGS = "text_postings_tags.txt" # Contains actual postings


def build_inverted_index(train_tags_file, isTest=False):
	"""Build an inverted index from the given text file"""
	global FILE_TAGS_POSTINGS, FILE_TAGS_DICT
	if isTest:
		FILE_TAGS_DICT = "query_" + FILE_TAGS_DICT
		FILE_TAGS_POSTINGS = "query_" + FILE_TAGS_POSTINGS

	file_train_tags = open(train_tags_file, "r")
	dict_tags = index_tags_inverted(file_train_tags)
	file_train_tags.close()

	dict_pointers = {}
	file_posting = open(FILE_TAGS_POSTINGS, "wb")

	for term, dict_imageid_tf in dict_tags.iteritems():
		doc_freq = len(dict_imageid_tf)

		ptr_begin = file_posting.tell()
		pickle.dump(dict_imageid_tf, file_posting) # (image id: term freq)
		ptr_end = file_posting.tell()

	 	dict_pointers[term] = [ptr_begin, ptr_end, doc_freq]
	 	file_posting.seek(1, os.SEEK_CUR) # jump one byte from current file position to save next entry

	file_posting.close()

	sorted_list = collections.OrderedDict(sorted(dict_pointers.items()))

	with open(FILE_TAGS_DICT,"wb") as handle:
		json.dump(sorted_list, handle, ensure_ascii=False)
	
	return (sorted_list, FILE_TAGS_DICT, FILE_TAGS_POSTINGS)


def build_normal_index(train_tags_file):
	"""Build a normal index i.e. img_id -> tag1, tag2, ... """
	file_train_tags = open(train_tags_file, "r")
	dict_tags = index_tags_normal(file_train_tags)
	file_train_tags.close()
	return dict_tags


def index_tags_normal(file_train_tags):
	dict_tags = {}
	if file_train_tags:
		for line in file_train_tags:
			tokens = line.split()
			image_id = tokens[0]
			len_doc = len(tokens) # for normalization

			dict_tags[image_id] = tokens[1:]
	return dict_tags


def index_tags_inverted(file_train_tags):
	dict_tags = {}
	if file_train_tags:
		for line in file_train_tags:
			tokens = line.split()
			image_id = tokens[0]
			len_doc = len(tokens) # for normalization

			for term in tokens[1:]:
				term = term.lower()

				# {term: {image_id: count} }
				if term in dict_tags:
					if image_id not in dict_tags[term]:
						dict_tags[term][image_id] = 1
					else:
						dict_tags[term][image_id] += 1
				else:
					dict_tags[term] = {image_id: 1}
	return dict_tags


if __name__ == '__main__':
	# Go to /../ImageData/train/train_text_tags.txt
	train_tags_path = os.path.join(os.path.dirname(__file__), "..", "ImageData", "train", FILE_TRAIN_INDEX)
	build_inverted_index(train_tags_path)