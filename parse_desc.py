from __future__ import print_function

#from bs4 import BeautifulSoup as Soup
#from opencc import OpenCC 
from hanziconv import HanziConv
import numpy as np
import re
#from xml.etree import ElementTree
#from xml.etree.ElementTree import Element, SubElement
#from lxml import etree
#import jieba

#def parse_description(desc):

def prune_text(str):
	str = str.replace("<p>", "")
	str = str.replace("</p>", "")
	str = str.replace("<br/>", "")
	str = str.replace("[", " ")
	str = str.replace("]", " ")
	str = str.replace(",", " ")
	str = str.replace(":", " ")
	str = str.replace("\\", " ")
	str = str.replace("/", " ")
	str = str.replace("\n", " ")
	str = str.replace("\r", " ")
	return str
			
def main():
	global featureWordList
	
	list_patterns = [r'<p>', r'</p>', r'<br/>', r'[\[\]]', r'[\u4E00-\u9FA5]', r'[,:]']
	#combined_pat = r'|'.join((r'<p>', r'</p>'))
	combined_pat = r'|'.join(list_patterns)
	symbolRe = re.compile(combined_pat)
	exTest = symbolRe.sub('', 'it 聲音 is a test,:: <p></p><br/>[asd] :, ')
	print(exTest)

	vocab_patterns = ['test', 'it', 'is']
	combined_pat = r'|'.join(vocab_patterns)
	textRe = re.compile(combined_pat)
	exTest = textRe.match('it 聲音 is a test,:: <p></p><br/>[asd] :, ')
	print('text = ', exTest)

	to_convert = '聲音 asd 开放中文转换 asd'	
	print('original ', to_convert)
	converted = HanziConv.toTraditional(to_convert)
	print('converted ', converted)
main()
