# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 16:28:28 2021

@author: tapan
"""


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

messages = pd.read_csv(r'C:\Users\tapan\Documents\Tapan\machine_learning_notes\Tapan_ML_Projects\smsspamcollection\SMSSpamCollection',
                       sep = '\t', names = ['label','message'])

# print(messages.head())
#print(messages.describe())

ps = PorterStemmer()
wordnet = WordNetLemmatizer()

corpus = []

for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
cv = TfidfVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values

X_train,X_test,y_train,y_test = train_test_split(X, y,test_size = 0.20, random_state =0)

spam_detect = MultinomialNB().fit(X_train,y_train)
y_pred = spam_detect.predict(X_test)

confusion_mat = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)



    

    
    

