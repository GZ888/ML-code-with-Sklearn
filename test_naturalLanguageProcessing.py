# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:04:47 2020

@author: gzhan
"""
# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# use tab as delimiter between columns, ignore double quote
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter= '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')  #download review related keywords from nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()  # convert all characters to lower case
    review = review.split()  # split review into different words
    ps = PorterStemmer()  # stemming is to retrieve the root of each word
    #python has algorithm to go through words in a set faster than in a list
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)  # reconstruct review from cleaned text words
    corpus.append(review)    
    
# Creating the Bag of Words model
# sparse matrix with each word in the review as one column
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #filter out the most relevant word by deleting words that only show once or twice
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Apply Naive Bayes to classify y into good and bad review
# based on bag of word model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting Classification
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


#Predicting the Test set results
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
