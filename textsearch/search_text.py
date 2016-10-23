#! /usr/bin/env
# -*- coding: latin2 -*- 
# Usage: python filename.py -q example_query
import os
import getopt
import sys
import math
import cPickle as pickle
import json
# import nltk
# from nltk.stem import PorterStemmer
import operator
import heapq

# ALL_DOC_ID_KEYS = "ALL_DOC_IDS"
# FILE_DOC_LENGTHS = "lengths.txt"
HEAP_CAPACITY = 320
THRESHOLD_LOW_IDF = 0.35
K = 16 # Number of results

# dict_all_doc_ids = []
N = 0
INPUT_FILE_TAGS_DICT = "text_dict_tags.txt" # Contains tag1 -> ptr_start, ptr_end, doc_freq, ...
INPUT_FILE_TAGS_POSTINGS = "text_postings_tags.txt" # Contains actual postings
OUTPUT_FILE_RANKING = "output_rankings.txt"


def search_text_index(input_query, limit):
	""" Load the dictionary and execute the queries one by one"""
	dict_pointers = {}
	dict_doc_lengths = {}
	posting_file = None
	
	global N
	N = 100

    #If the contents of fp are encoded with an ASCII based encoding other than UTF-8 (e.g. latin-1),
    # then an appropriate encoding name must be specified. 
	dictionary_file = open(os.path.join(os.path.dirname(__file__), INPUT_FILE_TAGS_DICT), "rb")
	dict_pointers = json.load(dictionary_file, encoding="latin2")
	
	posting_file = open(os.path.join(os.path.dirname(__file__), INPUT_FILE_TAGS_POSTINGS), "rb")

	results = execute_query(input_query, dict_pointers, limit)

	dictionary_file.close()
	posting_file.close()
	return results
	# doc_lengths_file.close()
	# output_file.close()


def execute_query(line, dict_pointers, limit):
	'''Execute query and return top K=160 entries of (score, img_id)'''
	scores = {}

	query = line.split()
	query = [x.lower() for x in query]

	weighted_term_freq_for_query = 0.0
	l2_norm_query = 0.0

	for term in query:
		# Process high idf terms only unless single term query
		if compute_idf(term, dict_pointers) >= THRESHOLD_LOW_IDF or len(query) == 1:
			weighted_term_freq_for_query = 1 + math.log10(query.count(term)) #tf
			idf = compute_idf(term, dict_pointers) #idf
			weighted_term_freq_for_query *= idf #tf-idf

			# l2_norm_query += math.pow(weighted_term_freq_for_query, 2) # For normalisation later

			postings = retrieve_postings(term, dict_pointers) # Retrieve image_ids to compute train images similarity scores for this term
			for doc_id, term_freq in postings.iteritems():
				cur_score = scores.setdefault(doc_id, 0)
				weighted_term_freq_for_doc = compute_log_tf(term, doc_id, dict_pointers)
				scores[doc_id] = cur_score + (weighted_term_freq_for_query * weighted_term_freq_for_doc) #existing score + (similarity for this term)

	heap = []
	for doc in scores:
		scores[doc] = scores[doc]#/math.sqrt(l2_norm_query) #/math.sqrt(dict_doc_lengths[doc])
		heapq.heappush(heap, (-scores[doc], doc)) # ALL TIMES -1 so that top of heap is best score
		if len(heap) >= HEAP_CAPACITY:
			del heap[HEAP_CAPACITY:] # Pruning non-contenders roughly

	largest = heapq.nsmallest(limit, heap) # Because python is min heap so 
	return largest 


def compute_idf(term, dict_pointers):
	'''Compute the inverse document frequency of a term'''
	if term in dict_pointers:
		df = dict_pointers[term][2] # Dict[term] contains [ptr_begin, ptr_end, doc_freq]
	else:
		df = 0
	return math.log10(float(N)/df) if df != 0 else sys.float_info.epsilon


def compute_log_tf(term, doc_id, dict_pointers):
	'''Compute the log10 of term frequency of a term in a document'''
	dict_doc_ids_tf = retrieve_postings(term, dict_pointers)
	term_freq = dict_doc_ids_tf.get(doc_id, 0)
	if term_freq == 0:
		return 0
	else:
		return 1.0 + math.log10(term_freq)


def retrieve_postings(term, dict_pointers):
	'''Retrive all postings with term_freq from the posting file'''
	posting_file = open(os.path.join(os.path.dirname(__file__), INPUT_FILE_TAGS_POSTINGS), "rb")

	if term in dict_pointers:
		ptr_begin, ptr_end, doc_freq = dict_pointers[term]

		posting_file.seek(ptr_begin)
		posting_list = pickle.loads(posting_file.read(ptr_end - ptr_begin))

		return posting_list
	posting_file.close()
	return {}


def usage():
	print "usage: " + sys.argv[0] + " -q example_query"


# input_file_d = input_file_p = input_file_q = output_file_o = None
try:
	opts, args = getopt.getopt(sys.argv[1:], 'q:')
except getopt.GetoptError, err:
	# usage()
	sys.exit(2)


def main():
	input_query = None
	for o, a in opts:
		if o == '-q':
			input_query = a

	if input_query == None:
		usage()
		sys.exit(2)

	search_text_index(input_query)

if __name__ == '__main__':
	main()
