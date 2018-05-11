from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from keras import optimizers

featureLen = 2

def xorfunc(data):
	if (data[0] % 2 == 1) and (data[1] % 2 == 0):
		return 1
	if (data[1] % 2 == 1) and (data[0] % 2 == 0):
		return 1
	return 0

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tokenizer = Tokenizer(char_level=False)
test_str = [
'here computer test',
'name computer OK',
'vision am here'
]

tokenizer.fit_on_texts(test_str)

print('word_counts =', tokenizer.word_counts)
matrix = tokenizer.texts_to_matrix(['computer vision'], mode='tfidf')
print('matrix = ', matrix)