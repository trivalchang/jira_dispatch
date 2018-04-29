from __future__ import print_function

import os
import sys
import io
import re
import numpy as np
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import OrderedDict

from hanziconv import HanziConv

sys.path.append('./utility')
from fileOp.h5_dataset import h5_dump_dataset, h5_load_dataset

class jira_xml_reader:
	root = None
	def __init__(self, fname):
		tree = ElementTree.parse(fname)
		self.root = tree.getroot()
		print('self.root = {}'.format(self.root))
		
	def basicInfo(self):
		try:
			if (self.root != None):
				print('xml open OK')
		except:
			print('xml open error!')

	def parseDesc(self, desc):
		lines = ''
		if desc.text == None:
			return lines
		text = '<rss>'+desc.text.replace("<br/>", "").replace('&nbsp;', '')+'</rss>'
		text = text.encode('utf-8')
		try:
			desc_root = ElementTree.fromstring(text)
		except:
			print('=========error========')
			print(text)
		if desc_root == None:
			print('can not decode description')
			return lines
		lines = lines.join(desc_root.itertext())
		#for element in desc_root.iter():
			#if element.text != None:
				#lines.append(element.text)			
		
		return lines
		
	def getIssueInfo(self):
		channel = self.root.find('channel')
		for issue in channel.findall('item'):
			title = issue.find('title').text
			summary = issue.find('summary').text
			key = issue.find('key').text
			desc = self.parseDesc(issue.find('description'))
			assignee = issue.find('assignee').text
			#desc = issue.find('description').text

			
			yield(title, summary, key, desc, assignee)
			

def prurify_text(str, _re):
	newstr = _re.sub(' ', str)
	return newstr

def translate_text(str, _dict):
	for (key, val) in _dict.items():
		str = str.replace(key, val)
	return str

def readChinesetoEnglishDict(fname):
	_dict = dict()
	f = io.open(fname, mode='r', encoding="utf-8")
	for line in f:
		vocab = line.replace('\n', '').split()
		_dict[vocab[0]] = vocab[1]

	f.close()
	return _dict

def readVocabulary(fname):
	_vocab = []
	f = io.open(fname, mode='r', encoding="utf-8")
	_vocab = [line.replace('\n', '') for line in f]
	f.close()
	return _vocab

#fileList = ['jira_all_1.xml', 'jira_all.xml', 'FREEWVIEW.xml']
#fileList = ['AND_Andy.xml', 'FREEWVIEW.xml']
fileList = ['AND_Andy.xml', 'jira_all_1.xml']
#fileList = ['ML3RTANOM-461.xml']
#fileList = ['img80.xml']			
def main():
	global featureWordList

	myDict = readChinesetoEnglishDict("translate.txt")
	myVocab = readVocabulary("vocabulary.txt")
	tokenizer = Tokenizer()

	list_patterns = [r'<p>', r'</p>', r'<br/>', r'\n', r'[\[\]]', r'[\u4E00-\u9FA5]', r'[\\,:_=]']		
	combined_pat = r'|'.join(list_patterns)
	toRemoveRe = re.compile(combined_pat)


	issue_cnt = 0
	issue_text = []
	issue_label = []
	for f in fileList:
		issue_parser = jira_xml_reader(f)
		issue_parser.basicInfo()

		for (title, summary, key, desc, assignee) in issue_parser.getIssueInfo():
			words = []
			issue_cnt = issue_cnt + 1

			text = title + summary + desc

			text = translate_text(text, myDict)
			text = prurify_text(text, toRemoveRe)

			_text = " ".join([x for x in text.split() if x in myVocab])

			issue_text.append(_text)
			classIdx = 1 if assignee == 'Andy Chang' else 0
			issue_label.append(classIdx)

	#tokenizer.fit_on_texts(issue_text)
	print('\n\nTotal Issue         {}'.format(issue_cnt))
	#issue_matrix = tokenizer.texts_to_matrix(issue_text)
	#issue_matrix = pad_sequences(issue_matrix, maxlen=100)

	#h5_dump_dataset(issue_matrix, 
	#				issue_label, 
	#				'issueFeature.hdf5', 
	#				'issue',
	#				'w')
	ofile = open("issue_Output.txt", "w")
	for label, text in zip(issue_label, issue_text):
		if (label == 1):
			ofile.write('{},{}\n'.format(label, text))
	for label, text in zip(issue_label, issue_text):
		if (label == 0):
			ofile.write('{},{}\n'.format(label, text))
	ofile.close()
main()
