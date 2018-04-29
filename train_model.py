from __future__ import print_function

import os
import sys
import random
from collections import OrderedDict
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np
import argparse

sys.path.append('./utility')


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

	evaluate_start = args['number']
	evaluate_end = args['number'] + 10
	evaluate_text = positiveText[evaluate_start:evaluate_end] + negativeText[evaluate_start:evaluate_end]
	evaluate_label = positiveLabel[evaluate_start:evaluate_end] + negativeLabel[evaluate_start:evaluate_end]

	test_start = evaluate_end
	test_end = test_start + 20
	test_text = positiveText[test_start:test_end] + negativeText[test_start:test_end]
	test_label = positiveLabel[test_start:test_end] + negativeLabel[test_start:test_end]

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(train_text)
	issue_matrix = tokenizer.texts_to_matrix(train_text)
	issue_matrix = pad_sequences(issue_matrix, maxlen=100)

	model = Sequential()
	model.add(Dense(256, input_dim=100, activation='sigmoid'))
	model.add(Dense(128, input_dim=100, activation='sigmoid'))
	model.add(Dense(1, activation='sigmoid'))
	#model.add(Dense(128))
	#model.add(Activation('sigmoid'))
	model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
	input_data = np.asarray(issue_matrix)
	input_label = np.asarray(train_label)
	model.summary()
	model.fit(input_data, input_label, validation_split=0.1, epochs=100, batch_size=10)

	issue_matrix = tokenizer.texts_to_matrix(evaluate_text)
	issue_matrix = pad_sequences(issue_matrix, maxlen=100)
	input_data = np.asarray(issue_matrix)
	input_label = np.asarray(evaluate_label)
	score = model.evaluate(input_data, input_label)
	print('score = ', score)

	issue_matrix = tokenizer.texts_to_matrix(test_text)
	issue_matrix = pad_sequences(issue_matrix, maxlen=100)
	input_data = np.asarray(issue_matrix)
	input_label = np.asarray(test_label)

	predict = model.predict(input_data).flatten()
	predict = [1 if p > 0.7 else 0 for p in predict]
	predict = np.asarray(predict)
	print('predict = ', predict)
	print('real = ', input_label)
	diff =  predict - input_label
	diff_cnt = np.count_nonzero(diff)
	print('incorrect = {}/{}'.format(diff_cnt, predict.shape[0]))
main()