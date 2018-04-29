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
'In a moment rich with symbolism and pomp, South Korean leader Moon Jae-in and Mr Kim shook hands at the border.Mr Kim said it was the starting point for peace, after crossing the military line that divides the peninsula.It comes just months after warlike rhetoric from North Korea.Much of what the summit will focus on has been agreed in advance, but many analysts remain sceptical about the North apparent enthusiasm for engagement.',
'The meeting - the first between Korean leaders in more than a decade - is seen as a step toward possible peace on the peninsula and marks the first summit of its kind for Mr Kim.',
'The leaders were met by an honour guard in traditional costume on the South Korean side. The pair then walked to the Peace House in Panmunjom, a military compound in the demilitarised zone (DMZ) between the two countries.'
]

tokenizer.fit_on_texts(test_str)

print(tokenizer.word_counts)
tokenizer.texts_to_sequences(['military comes'])
tokenizer.texts_to_matrix(['military comes'])