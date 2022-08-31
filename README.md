# Predict Closed Questions on StackOverflow

## Contents

## 1. Project overview

This article will focus on BERT NLP model, and how it can be used to solve a common issue in Q&A sites, using data collected from StackOverflow. Sites like StackOverflow deal with a repetitive issue - bad questions!
Questions that don't meet the StackOverflow standard because they are irrelevant, poorly written, etc make about 6% of the questions posted. 

It can be very time consuming to deal with this issue so a ML model that can help predict a question classification could be very valuable.

## 2 .Libraries used:

# libraries

### data manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from pathlib import Path  
import datetime
import seaborn as sn

### import nlp Tokenizer/tools
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
from transformers import BertTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertModel,DistilBertModel
from transformers import AdamW

### Evaluation and data prep
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split

### setup
pd.set_option('display.max_colwidth', None) # want to see all info in a cell

## 3 Data been imported:

`df_stock_over_raw` - StackOverflow raw questions data <br>

## 4 files:
StackOverflow.ipynb - Jupyter notebook that analyze and predict StackOverflow questions <br>
README.md - readme file

https://www.kaggle.com/competitions/predict-closed-questions-on-stack-overflow/data -  training data

## 5 Conclusions:


This is the post I :


## 6 Acknowledgements:

Data was provided from Kaggle:
https://www.kaggle.com/competitions/predict-closed-questions-on-stack-overflow/data

towardsdatascience articles used:
https://towardsdatascience.com/a-sub-50ms-neural-search-with-distilbert-and-weaviate-4857ae390154
https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b
https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f
