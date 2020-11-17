# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 18:54:24 2020

@author: hp
"""

import pandas as pd
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import re
import string
from sklearn.metrics import accuracy_score, f1_score
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


df=pd.read_csv('trainer.csv')

df.duplicated().value_counts()
# Duplicates detected and dropped
df.drop_duplicates(subset='tweet', keep='first', inplace=True)

#Upsampling
df_minority = df[df.label==1]
df_majority = df[df.label==0]
 
df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=5892,
                                 random_state=123)
 
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

'''df_upsampled=df'''

#Cleaning the data using regular expression and more
def clean(text):
    num='[0-9]'
    text = re.sub('\@.*?\:', '', text)
    text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)
    text = re.sub('@', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(num, '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\r', '', text)
    text = re.sub(r'[^\x00-\x7f]',r'', text)
    text = text.lower()
    return text
round1= lambda x: clean(x)
df_upsampled['tweet']=pd.DataFrame(df_upsampled.tweet.apply(round1))


'''Document Term Matrix'''
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
X = cv.fit_transform(df_upsampled['tweet'].values)
Y = df_upsampled['label']

X_train, X_test, Y_train, Y_test=tts(X, Y, test_size=0.2, random_state=42)

classifier=MultinomialNB(alpha=1)
classifier.fit(X_train, Y_train)
y_pred=classifier.predict(X_test)
f1_score(Y_test, y_pred)
classifier=MultinomialNB()

tuned = [{'alpha': [10**-4, 10**-2, 10**0, 10**2, 10**4]}]

model = GridSearchCV(MultinomialNB(), tuned, scoring = 'roc_auc', cv=5)
model.fit(X_train, Y_train)
print(model.best_estimator_)
print(model.score(X_test, Y_test))
y_pred = model.predict(X_test)
f1_score(Y_test, y_pred)

classifier=MultinomialNB(alpha=0.01)
classifier.fit(X_train, Y_train)
y_pred=classifier.predict(X_test)
f1_score(Y_test, y_pred)


'''TF-IDF'''

tf_idf_vect=TfidfVectorizer(ngram_range=(1,2), strip_accents='ascii', lowercase=True, stop_words='english')
final_tf=tf_idf_vect.fit_transform(df_upsampled['tweet'].values)

X = final_tf
Y = df_upsampled['label']

X_train, X_test, Y_train, Y_test=tts(X, Y, test_size=0.2, random_state=42)

classifier=MultinomialNB(alpha=1)
classifier.fit(X_train, Y_train)
y_pred=classifier.predict(X_test)
f1_score(Y_test, y_pred)
classifier=MultinomialNB()

classifier=MultinomialNB(alpha=0.01)
classifier.fit(X_train, Y_train)
y_pred=classifier.predict(X_test)
f1_score(Y_test, y_pred)





























param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['rbf'],
              'class_weight':['balanced']}
  
grid = GridSearchCV(SVC(), param_grid, refit = True) 
grid.fit(X_train, Y_train)
grid_predictions = grid.predict(X_test)
print(grid.best_params_)
print(grid.best_estimator_)






























''' Populating the Test data feature: label using the fitted model'''

df1=pd.read_csv('tester.csv')

df1.drop_duplicates(subset='tweet', keep='first', inplace=True)
df1.reset_index(drop=True, inplace=True)

df1['tweet']=pd.DataFrame(df1.tweet.apply(round1))

X1_test = tf_idf_vect.transform(df1['tweet'].values)
X1_test = cv.transform(df1['tweet'].values)

y_pred=classifier.predict(X1_test)

df_ans=pd.DataFrame(y_pred)

df_ans.columns=['label']
df_ans['id']=df1['id']
df_ans.set_index('id', inplace=True)
df_ans.index.name='id'
df_ans.to_csv('Predicted Labels Stacking.csv')






















