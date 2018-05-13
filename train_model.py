from __future__ import print_function

import os
import sys
import random
from collections import OrderedDict
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
from keras import initializers
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.append('./utility')


assigneeList = dict()

def readAssignee(fname):
	_assignee = dict()
	_rev_assignee = dict()
	f = open(fname, mode='r', encoding="utf-8")
	for line in f:
		w = line.replace('\n', '').split()
		_assignee[w[0]] = int(w[1])
		_rev_assignee[int(w[1])] = w[0]
	f.close()
	return _assignee, _rev_assignee	

def main():

	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--feature", required=True,  help="feature file")
	ap.add_argument("-n", "--number", required=True,  type=float, help="number of training sample")
	args = vars(ap.parse_args())

	myAssignee, _myRevAssignee = readAssignee("assignee.txt")
	
	issue_text = []
	issue_label = []
	issue_key = []
	ifile = open(args['feature'], "r")
	for line in ifile:
		key, label, text = line.split(',')
		issue_text.append(text.replace('\n', ''))	
		issue_label.append(int(label))
		issue_key.append(key)
	ifile.close()

	label_list = sorted(myAssignee.values())
	issue_label = np.asarray(issue_label)
	issue_text = np.asarray(issue_text)

	train_text = []
	train_label = []
	train_key = []
	test_text = []
	test_label = []
	test_key = []
	
	max_sample_per_class = 32768
	for label in label_list:
		_where = np.where(issue_label == label)[0]
		print('class {} has {} samples'.format(label, len(_where)))
		if len(_where) < max_sample_per_class:
			max_sample_per_class = len(_where)

	for label in label_list:
		_where = np.where(issue_label == label)[0]
		#if len(_where) > args['number']:
		#	_test_where = 
		#	_where = _where[:args['number']]
		#	_test_where = 
		if len(_where) == 0:
			continue
		n = int(args['number'] * max_sample_per_class)
		train_samples = np.random.choice(_where, n, replace=False)
		test_samples = np.asarray([x for x in _where if x not in train_samples])
		train_text = train_text + np.take(issue_text, train_samples, 0).flatten().tolist()
		train_label = train_label + np.take(issue_label, train_samples, 0).flatten().tolist()
		train_key = train_key + np.take(issue_key, train_samples, 0).flatten().tolist()
		test_text = test_text + np.take(issue_text, _where[n:], 0).flatten().tolist()
		test_label = test_label + np.take(issue_label, _where[n:], 0).flatten().tolist()
		test_key = test_key + np.take(issue_key, _where[n:], 0).flatten().tolist()
		print('class {} has {} training samples'.format(label, n))

	ofile = open("train_record.txt", "w")
	for record in zip(train_key, train_label, train_text):
		ofile.write('{}\n'.format(record))
	ofile.close()
	ofile = open("test_record.txt", "w")
	for record in zip(test_key, test_label, test_text):
		ofile.write('{}\n'.format(record))
	ofile.close()		
		
		
	tokenizer = Tokenizer()
	# find the words by tokenizer  
	tokenizer.fit_on_texts(train_text)
	# transform the text to tf-idf feature vector
	issue_matrix = tokenizer.texts_to_matrix(train_text+test_text, mode='tfidf')
	# normalize the feature vector
	scaler = MinMaxScaler(feature_range=(-1, 1))
	issue_matrix = scaler.fit_transform(issue_matrix)
	# split training data and test data
	train_matrix = issue_matrix[:len(train_text)]
	test_matrix = issue_matrix[len(train_text):]
	
	#kernel_init=initializers.random_normal(stddev=0.01)
	kernel_init=initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
	
	# establish model
	model = Sequential()
	model.add(Dense(512, 
					kernel_initializer=kernel_init,
					input_dim=issue_matrix.shape[1], 
					activation='sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(128, 
					kernel_initializer=kernel_init,
					activation='sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(len(label_list), activation='softmax'))
	model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
	input_data = np.asarray(train_matrix)
	input_label = to_categorical(np.asarray(train_label))
	model.summary()
	#history = model.fit(input_data, input_label, validation_split=0.2, epochs=500, batch_size=10)
	history = model.fit(input_data, input_label, epochs=500, batch_size=20)
	
	# plot the result
	if False:
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		# summarize history for loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()	
	
	# transform the test data
	input_data = np.asarray(test_matrix)
	input_label = to_categorical(np.asarray(test_label))
	score = model.evaluate(input_data, input_label)
	
	# classify the test data
	print('evaluate score = ', score)
	predicts = model.predict_classes(input_data)
	
	print('predict = ', predicts)
	# print the result by pandas
	df = pd.DataFrame({'label':test_label, 'predict':predicts, 'key':test_key})
	print('=======predict error {}/{}========='.format(len(df[df['label']!=df['predict']]), len(test_label)))
	print(df[df['label']!=df['predict']])

	# print the probability of each class
	pred_prob = model.predict_proba(input_data)
	ofile = open("pred_prob.txt", "w")
	for (key, real, pred, prob) in zip(test_key, test_label, predicts, pred_prob):
		if real == pred:
			continue
		ofile.write('{}\t{}({})\t{}({})\t\t'.format(key.ljust(16), _myRevAssignee[real], real, _myRevAssignee[pred], pred))
		for pp in prob.tolist():
			ofile.write(' {:.3f}'.format(pp))
		ofile.write('\n')
	
	# print the summary
	ofile.write('\n\nSummary\n')
	ofile.write('total wrong {}/{}\n'.format(len(df[df['label']!=df['predict']]), len(test_label)))
	ofile.write('evaluate score = {}\n'.format(score))
	test_label_a = np.asarray(test_label)
	for label in label_list:
		_where = np.where(test_label_a == label)[0]
		wrong_pred = np.take(predicts, _where, 0).flatten()
		wrong_num = np.count_nonzero(wrong_pred != label)
		ofile.write('{}: wrong {}/{}\n'.format(_myRevAssignee[label], wrong_num, len(_where)))
	ofile.close()
main()