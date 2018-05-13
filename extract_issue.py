from __future__ import print_function

import os
import sys
import io
import re
import numpy as np
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from collections import OrderedDict
import nltk
from hanziconv import HanziConv

# download nltk package
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag

lmtzr = WordNetLemmatizer()

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
		
		return lines
		
	def getIssueInfo(self):
		channel = self.root.find('channel')
		for issue in channel.findall('item'):
			title = issue.find('title').text
			summary = issue.find('summary').text
			key = issue.find('key').text
			desc = self.parseDesc(issue.find('description'))
			assignee = issue.find('assignee').get('username')
			#desc = issue.find('description').text

			
			yield(title, summary, key, desc, assignee)
			

def get_wordnet_pos(word_tag):
	if word_tag.startswith('J'):
		return wordnet.ADJ
	elif word_tag.startswith('V'):
		return wordnet.VERB
	elif word_tag.startswith('N'):
		return wordnet.NOUN
	elif word_tag.startswith('R'):
		return wordnet.ADV
	else:
		return None

def prurify_text(str, _re, vocab):
	str = _re.sub(' ', str).lower()
	words = ''
	for word, pos in pos_tag(word_tokenize(str)):
		if word in vocab:
			words = words + ' ' + word
		else:
			word_pos = get_wordnet_pos(pos) or wordnet.NOUN
			root_word = lmtzr.lemmatize(word, pos=word_pos)
			if root_word in vocab:
				words = words + ' ' + root_word
	return words

def translate_text(str, _dict):
	for (key, val) in _dict.items():
		str = str.replace(key, val+' ')
	return str

def readChinesetoEnglishDict(fname):
	_dict = dict()
	f = io.open(fname, mode='r', encoding="utf-8")
	for line in f:
		vocab = line.replace('\n', '').split()
		_dict[vocab[0]] = vocab[1]
		text_simplified = HanziConv.toSimplified(vocab[0])
		_dict[text_simplified] = vocab[1]

	f.close()
	return _dict
	
def readAssignee(fname):
	_assignee = dict()
	f = io.open(fname, mode='r', encoding="utf-8")
	for line in f:
		w = line.replace('\n', '').split()
		_assignee[w[0]] = int(w[1])

	f.close()
	return _assignee	

def readVocabulary(fname):
	_vocab = []
	f = io.open(fname, mode='r', encoding="utf-8")
	_vocab = [line.replace('\n', '').lower() for line in f]
	f.close()
	return _vocab

def main():
	
	myDict = readChinesetoEnglishDict("translate1.txt")
	myVocab = readVocabulary("vocabulary.txt")
	myAssignee = readAssignee("assignee.txt")

	# prepare regulation expression to remove symobls, characters, or unuseful chinses words
	list_patterns = [r'<p>', r'</p>', r'<br/>', r'\n', r'[\[\]]', r'[\u4E00-\u9FA5]', r'[\\,:_=-]']		
	combined_pat = r'|'.join(list_patterns)
	toRemoveRe = re.compile(combined_pat)


	issue_cnt = 0
	issue_text = []
	issue_label = []
	issue_key = []
	issue_assignee = []

	fileList = []
	for f in os.listdir('xml'):
		if f.endswith(".xml") == False:
			continue
		fileList.append('xml/'+f)

	for f in fileList:
		# prepare for xml parser
		issue_parser = jira_xml_reader(f)
		issue_parser.basicInfo()

		for (title, summary, key, desc, assignee) in issue_parser.getIssueInfo():
			# read and extract the information of issues
			words = []
			# combine the values of different tags to form a string
			text = title + ' ' + summary + ' ' + desc
			# translate chinese to english by proprietary dictionary
			text = translate_text(text, myDict)
			# remove symbols, tags and unuseful word
			_text = prurify_text(text, toRemoveRe, myVocab)

			if key in issue_key:
				continue
			issue_text.append(_text)
			# find the corresponding class of the assignee of the issue
			classIdx = myAssignee[assignee] if assignee in myAssignee else len(myAssignee)
			issue_label.append(classIdx)
			issue_key.append(key)
			issue_assignee.append(assignee)
			issue_cnt = issue_cnt + 1
			
	assignees, counts = np.unique(issue_assignee, return_counts=True)
	# show the issue number of all assignees, information only
	for (assignee, count) in zip(assignees, counts):
		#if count >= 80 and assignee in myAssignee:
		if count >= 0:
			print('{} :: {}'.format(assignee, count))
		
	print('\n\nTotal Issue         {}'.format(issue_cnt))

	# output the result to a file for training/test
	ofile = open("issue_Output.txt", "w")
	issue_label = np.asarray(issue_label)
	issue_text = np.asarray(issue_text)
	label_list = sorted(myAssignee.values())
	for label in label_list:
		_where = np.where(issue_label == label)
		alltext = np.take(issue_text, _where, 0).flatten()
		allkey = np.take(issue_key, _where, 0).flatten()
		for key, text in zip(allkey, alltext): 
			ofile.write('{},{},{}\n'.format(key, label, text))
		
	ofile.close()
	
main()
