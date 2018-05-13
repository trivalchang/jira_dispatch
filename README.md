# jira_dispatch

This is to extract issues from Jira, a software tracking system, and use deep learning to find the possibile owner of the issue.

There are 2 python programs for this project.

**extract_issue.py** is to extract the issue from Jira.
**train_model.py** is to train and classify the issues.

## extract_issue.py

Users must provide xml files from Jira server. A valid xml file for training/testing must include 4 tags of an issue, *title*, *summary*, *key*, *description* and *assignee*. Since I want to predict the issue owner where the issue is created, other Jira field is not used.

### python modules required

**libxml** - XML parser<br />
**nltk** - lemmatize English word to find the base form of the word. For example, "words" will be transformed to "word". After this process, an issue with "words" has the same feature with a issue with 'word'.<br />
**hanziconv** - to translate traditional Chinese words to Simplified Chinese words. My jira issues are mixed by English, simplified Chinese and traditional Chinese. So I have to translate some Chinese words to English.

### Several text files are required<br />
**translate.txt** - a mapping table for traditional Chinese to English. Chinese words not in this file will be discarded.<br />
**vocabulary.txt** - English words that will be keeped for the feature vector<br />
**assigneee.txt** - a mapping table for the name of the real assignee to a interger class label

### the output
extract_issue.py parses the xml files and outputs a csv file issue_Output.txt.<br />
issue_Output.txt contains all issues belonging to assginees in  **assigneee.txt**.
> TESTPRO01,1, program crash key press

the first value is the key of the issue. The second is the label of the issue assignee and the 3rd is a string extracted from issue's title, summary. 

## train_model.py

> usage: train_model.py [-h] -f FEATURE -n NUMBER

> optional arguments:<br />
  > -h, --help                      show this help message and exit<br />
  > -f FEATURE, --feature FEATURE   feature file<br />
  > -n NUMBER, --number NUMBER      percentage of training sample in feature file <br />
  
> example: python train_model.py -f issue_Output.txt -n 0.8  

### python modules required

**keras** - preprocessing of text, neuron network components<br />
**sklearn.preprocessing.MinMaxScaler** - to normalize the feature text<br />
**pandas** - to analyze the prediciton result<br />

train_model.py starts with reading the output file of extract_issue.py and split the issues into training data and test data. Then, keras Tokenizer is used to gernerate the feature vectors in tf-idf mode and normalize these feature vectors by MinMaxScaler.

After the above process, I use keras Sequential model to train and predict. <br />
With a 500 issues, 10 assignees of training data, the model can achieves 65% accuracy on test data. After analyzing the output, I think the accuracy could be further improved if the better descriotion of the issues are given.



