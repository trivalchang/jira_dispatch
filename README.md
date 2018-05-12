# jira_dispatch

This is to extract issues from Jira, a software tracking system, and use deep learning to find the possibile owner of the issue.

There are 2 python programs for this project.

**extract_issue.py** is to extract the issue from Jira.
**train_model.py** is to train and classify the issues.

## extract_issue.py

Users must provide xml files from Jira server. A valid xml file for training/testing must include 4 tags of a issue, *title*, *summary*, *key*, *description* and *assignee*. Since I want to predict the issue owner where the issue is created, other Jira field is not used.

python modules required

**libxml** - XML parser<br />
**nltk** - lemmatize English word to find the base form of the word. For example, "words" will be transformed to "word". After this process, an issue with "words" has the same feature with a issue with 'word'.<br />
**hanziconv** - to translate traditional Chinese words to Simplified Chinese words. My jira issues are mixed by English, simplified Chinese and traditional Chinese. So I have to translate some Chinese words.

Several text files are required

