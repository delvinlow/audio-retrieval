#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
The simple implementation of Sentence2Vec using Word2Vec.
"""

import logging
import sys
import os
from word2vec import Word2Vec, Sent2Vec, LineSentence

def getTextualFeature(text_reading_path):
    # Train and save the Word2Vec model for the text file.
    # Please note that, you can change the dimension of the resulting feature vector by modifying the value of 'size'.
    model = Word2Vec(LineSentence(text_reading_path), size=500, window=5, sg=0, min_count=5, workers=8)
    model.save(text_reading_path + '.model')

    # Train and save the Sentence2Vec model for the sentence file.
    model = Sent2Vec(LineSentence(text_reading_path), model_file=text_reading_path + '.model')
    model.save_sent2vec_format(text_reading_path + '.vec')

    program = os.path.basename(sys.argv[0])


if __name__ == '__main__':
    # 1. Set the access path to read the text file.
    #    Where each line is one sentence, in your case, one sentence is one description of video.
    text_reading_path = 'test.txt'

    # 2. Utilize the well-refined Word2Vec and Sentence2Vec model to extract textual features.
    #    There is the output file, named 'xxx.vec', where each line is the corresponding feature vector of sentence.
    getTextualFeature(text_reading_path=text_reading_path)