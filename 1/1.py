import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.utils import shuffle
'''
    数据应分别放置在test与train文件夹下，分别命名test.tsv与train.tsv
'''
data_loc = r'train/train.tsv'
def process_data():
    all_data = pd.read_csv(data_loc,sep=r'\t')
    x_data = all_data["Phrase"]
    y_data = all_data["Sentiment"]
    
    x_train,x_test,y_train,y_test = train_test_split(x_data,y_data, test_size=0.2)
    # x_train,x_valid,y_train,y_valid = train_test_split(x_data,y_data, test_size=0.2)
    return  x_train,x_test,y_train,y_test

def _count(x_train,x_test):
    cv = CountVectorizer()
    x_train_feature = cv.fit_transform(x_train)
    x_test_feature = cv.transform(x_test)
    return x_train_feature,x_test_feature
def _tf(x_train,x_test):
    tf = TfidfVectorizer(max_features=20000)
    x_train_feature = tf.fit_transform(x_train)
    x_test_feature = tf.transform(x_test)
    return x_train_feature,x_test_feature
def _ngram(x_train,x_test):
    tf = TfidfVectorizer(ngram_range=(2,2),max_features=30000)
    x_train_feature = tf.fit_transform(x_train)
    x_test_feature = tf.transform(x_test)
    return x_train_feature,x_test_feature

def _lr():
    clf = LogisticRegression(random_state=0, solver='sag', multi_class='multinomial')
    return clf

def _sgd():
    clf = SGDClassifier(alpha=0.001,early_stopping=True,validation_fraction=0.2,shuffle=True)
    return clf

if __name__ == "__main__":
    x_train,x_test,y_train,y_test = process_data()
    for feature in (_count,_tf,_ngram):
        for model in (_lr,_sgd):
            x_train_feature,x_test_feature = feature(x_train,x_test)
            shuffle(x_train_feature,y_train)
            clf = model()
            clf.fit(x_train_feature,y_train)
            predict = clf.predict(x_test_feature)
            print(np.mean(predict == y_test))

"""
0.6483403819043957
0.550397283096245
0.6340510060233243
0.5144495706779444
0.5950916314238114
0.5062796360374215
"""