# -*- coding: utf-8 -*-
"""

A prototye for spam detection using basic ML classifiers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud


data = pd.read_csv('spam.csv',encoding = "ISO-8859-1")
X = data.iloc[1:,1]
y = data.iloc[1:,0]
X_word = X

#Encoding Spam
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

count_vectorizer = CountVectorizer(decode_error = "ignore")
X = count_vectorizer.fit_transform(X)

#Split train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)

model = MultinomialNB()
model.fit(X_train,y_train)
print("train score: ",model.score(X_train,y_train))
print("test score: ",model.score(X_test,y_test))

#y_predict = model.predict(X_test)

#wordcloud for spam messages

words = ""
for msg in data[data['v1'] == 'spam']['v2']:
    msg = msg.lower()
    words+=msg + ' '
wordcloud  = WordCloud(width = 400, height = 600).generate(words)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
    


