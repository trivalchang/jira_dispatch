from __future__ import print_function

import os
import sys
import random
from collections import OrderedDict
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import argparse

sys.path.append('./utility')

from fileOp.h5_dataset import h5_dump_dataset, h5_load_dataset

assigneeList = dict()
def main():

	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--feature", required=True,  help="feature file")
	ap.add_argument("-n", "--number", required=True,  type=int, help="number of training sample")
	args = vars(ap.parse_args())

	issue_text = []
	issue_label = []
	ifile = open(args['feature'], "r")
	for line in ifile:
		label, text = line.split(',')
		issue_text.append(text.replace('\n', ''))	
		issue_label.append(int(label))
	ifile.close()

	positiveNum = np.count_nonzero(issue_label)
	positiveLabel = issue_label[:positiveNum]
	positiveText = issue_text[:positiveNum]
	negativeLabel = issue_label[positiveNum:]
	negativeText = issue_text[positiveNum:]

	train_text = positiveText[:args['number']] + negativeText[:args['number']]
	train_label = positiveLabel[:args['number']] + negativeLabel[:args['number']]

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(train_text)
	issue_matrix = tokenizer.texts_to_matrix(train_text)
	issue_matrix = pad_sequences(issue_matrix, maxlen=100)


main()