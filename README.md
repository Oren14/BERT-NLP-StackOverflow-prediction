# Predict Closed Questions on StackOverflow

## Contents

## 1. Project overview

This article will focus on BERT NLP model, and how it can be used to solve a common issue in Q&A sites, using data collected from StackOverflow. Sites like StackOverflow deal with a repetitive issue - bad questions!
Questions that don't meet the StackOverflow standard because they are irrelevant, poorly written, etc make about 6% of the questions posted. 

It can be very time consuming to deal with this issue so a ML model that can help predict a question classification could be very valuable.

## 2 .Libraries used:

### data manipulation
`import numpy as np`  <br>
`import pandas as pd` <br>
`import matplotlib.pyplot as plt` <br>
`import re` <br>
`import os` <br>
`from pathlib import Path` <br>
`import datetime` <br>
`import seaborn as sn` <br>

### import nlp Tokenizer/tools
`import nltk` <br>
`from nltk.corpus import stopwords` <br>
`from nltk.tokenize import word_tokenize` <br>
`from nltk.corpus import words` <br>
`from nltk.stem import WordNetLemmatizer` <br>
`nltk.download('omw-1.4')` <br>
`from transformers import BertTokenizer` <br>
`import torch` <br>
`import torch.nn as nn` <br>
`import torch.optim as optim` <br>
`from torch.optim import Adam` <br>
`from tqdm import tqdm` <br>
`from transformers import BertModel,DistilBertModel` <br>
`from transformers import AdamW` <br>


### Evaluation and data prep
`from sklearn.metrics import accuracy_score, classification_report, confusion_matrix` <br>
`import seaborn as sns` <br>
`from sklearn.model_selection import train_test_split` <br>

### setup
`pd.set_option('display.max_colwidth', None)`

## 3 Data been imported:
`df_stock_over_raw` - StackOverflow raw questions data <br>

## 4 files:
`StackOverflow.ipynb` - Jupyter notebook that analyze and predict StackOverflow questions <br>
`README.md` - readme file

https://www.kaggle.com/competitions/predict-closed-questions-on-stack-overflow/data -  training data

## 5 Conclusions:
The final results showed an accuracy rate of 57% in predicting the different classes. <br>
Good recall for “open” questions (93%), I worry about the 0% recall rate for type 1 questions and I think for type 2/3/4 further training is needed. <br>
I will consider adding more features into the model.

This is the post I :
https://medium.com/@orengutman10/stackoverflow-text-classification-with-bert-38af99c287cb

## 6 Acknowledgements:  <br>
Data was provided from Kaggle:  <br>
https://www.kaggle.com/competitions/predict-closed-questions-on-stack-overflow/data

towardsdatascience articles used:  <br>
https://towardsdatascience.com/a-sub-50ms-neural-search-with-distilbert-and-weaviate-4857ae390154
https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b
https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f
