# -*- coding: utf-8 -*-
"""
Sentiment Analysis on Amazon reviews
"""

import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression

from bs4 import BeautifulSoup

wordnet_lemmatizer = WordNetLemmatizer()

# from http://www.lextek.com/manuals/onix/stopwords1.html
stopwords = set(words.rstrip() for words in open('stopwords.txt'))

# from http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
#The Sentiment Dataset contains product reviews taken from Amazon.com for Electronics category
positive_rev = BeautifulSoup(open('positive.review').read())
positive_rev = positive_rev.findAll('review_text')

negative_rev = BeautifulSoup(open('negative.review').read())
negative_rev = negative_rev.findAll('review_text')

np.random.shuffle(positive_rev)

positive_rev = positive_rev[:len(negative_rev)]

def custom_tokenizer(s):
    s = s.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    return tokens
    
vocab = {}
idx = 0

pos_tokenized=[]
neg_tokenized =[]

for review in positive_rev:
    tokens = custom_tokenizer(review.text)
    pos_tokenized.append(tokens)
    for w in tokens:
        if(w not in vocab):
            vocab[w] = idx
            idx+=1
    
for review in negative_rev:
    tokens = custom_tokenizer(review.text)
    neg_tokenized.append(tokens)
    for w in tokens:
        if(w not in vocab):
            vocab[w] = idx
            idx+=1

def wordtovector(tokens,label):
    y = np.zeros(len(vocab) + 1)
    for t in tokens:
        i = vocab[t]
        y[i]+=1
    y = y/y.sum()
    y[-1] = label
    return y
    
N = len(pos_tokenized) + len(neg_tokenized)
data = np.zeros((N,len(vocab) + 1))
i = 0

for tokens in pos_tokenized:
    data[i:] = wordtovector(tokens,1)
    i+=1
for tokens in neg_tokenized:
    data[i:] = wordtovector(tokens,0)
    i+=1
    
np.random.shuffle(data)
X = data[:,:-1]
y = data[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25)
model = LogisticRegression()
model.fit(X_train,y_train)
print('CLassification : ', model.score(X_test,y_test))
    