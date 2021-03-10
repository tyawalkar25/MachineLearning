# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 10:55:23 2021

@author: tapan
"""


import pandas as pd

stock_data = pd.read_csv('Stock_Dataa.csv/Stock_Data.csv', encoding = "ISO-8859-1")


train = stock_data[stock_data['Date'] < '20150101']
test = stock_data[stock_data['Date'] > '20141231']

data = train.iloc[:,2:27]

data.replace("[^A-Za-z]"," ",regex = True,inplace = True)

l = [i for i in range(25)]
new_headers = [str(i) for i in l]
data.columns = new_headers

for i in new_headers:
    data[i] = data[i].str.lower()
    
    
headlines = []

for i in range(len(data)):
    headlines.append(' '.join(str(x) for x in data.iloc[i,0:25]))
    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

cv = CountVectorizer()

trainDataset = cv.fit_transform(headlines)

randomClassifier = RandomForestClassifier()

randomClassifier.fit(trainDataset,train['Label'])

test_headlines = []
for i in range(len(test)):
    test_headlines.append(' '.join(str(x) for x in test.iloc[i,2:27]))
    
testDataset = cv.transform(test_headlines)
predictions = randomClassifier.predict(testDataset)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

conf_mat = confusion_matrix(test['Label'], predictions)

score = accuracy_score(test['Label'], predictions)

report = classification_report(test['Label'], predictions)









    