#coding=utf-8

from base import raw_trainingset, proced_trainingset
from preprocess import part_trainingset
from feature import feature
from corpus import gen_training_corpus

from sklearn import svm
from sklearn.svm import LinearSVC, SVC, NuSVR, SVR
from sklearn import multiclass
from sklearn import cross_validation
from sklearn.externals import joblib
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import *


class jc_model:

    features = None
    targets  = None
    clf = None

    def __init__(self):
        # self.clf = svm.NuSVR(kernel='sigmoid')
        # self.clf = multiclass.OneVsOneClassifier(SVC(kernel='sigmoid', probability=True)) # 里面的参数可以随意设置
        # self.clf = multiclass.OneVsRestClassifier(SVR())
        self.clf = svm.SVC(kernel='sigmoid', probability=True)

    def load_model(self):
        # self.clf = joblib.load('./pkl/nusvr.pkl')
        # self.clf = joblib.load('./pkl/ovoclf.pkl')
        # self.clf = joblib.load('./pkl/ovrclf.pkl')
        self.clf = joblib.load('./pkl/svc.pkl')

    def save_model(self):
        # joblib.dump(self.clf, './pkl/linearsvc.pkl', compress=3)
        # joblib.dump(self.clf, './pkl/nusvr.pkl', compress=3)
        # joblib.dump(self.clf, './pkl/ovoclf.pkl', compress=3)
        # joblib.dump(self.clf, './pkl/ovrclf.pkl', compress=3)
        joblib.dump(self.clf, './pkl/svc.pkl', compress=3)

    def load_data(self):
        features = []
        targets = []
        # f_train = open(raw_trainingset, 'r')
        f_train = open(part_trainingset, 'r') # 使用80%训练集
        for line in f_train:
            corpus = gen_training_corpus(line)
            feat = feature(corpus.title, corpus.person1, corpus.person2)
            features.append(np.array(feat))
            targets.append(corpus.label)
        self.features = np.array(features)
        self.targets = np.array(targets)
        f_train.close()

    def train(self):
        self.clf.fit(self.features, self.targets)

    def test_model(self): # k-fold cross validation
        scores = cross_validation.cross_val_score(self.clf, self.features, self.targets, cv=10)
        print scores

    def predict(self, feature):
        return self.clf.predict(feature)

    def predict_proba(self, feature):
        return self.clf.predict_proba(feature)

class baseline_model:
    features = None
    targets  = None
    clf = None

    def __init__(self):
        # self.clf = BernoulliNB()
        # self.clf = AdaBoostClassifier() # 不好使，通通分到几类
        self.clf = RandomForestClassifier()

    def load_model(self):
        # self.clf = joblib.load('./pkl/bernoullinb.pkl')
        # self.clf = joblib.load('./pkl/adaboost.pkl')
        self.clf = joblib.load('./pkl/randomforest.pkl')

    def save_model(self):
        # joblib.dump(self.clf, './pkl/bernoullinb.pkl', compress=3)
        # joblib.dump(self.clf, './pkl/adaboost.pkl', compress=3)
        joblib.dump(self.clf, './pkl/randomforest.pkl', compress=3)

    def load_data(self):
        features = []
        targets = []
        f_train = open(proced_trainingset, 'r')
        # f_train = open(part_trainingset, 'r')
        for line in f_train:
            corpus = gen_training_corpus(line)
            feat = feature(corpus.title, corpus.person1, corpus.person2)
            features.append(np.array(feat))
            targets.append(corpus.label)
        self.features = np.array(features)
        self.targets = np.array(targets)
        f_train.close()

    def train(self):
        self.clf.fit(self.features, self.targets)

    def test_model(self): # k-fold cross validation
        scores = cross_validation.cross_val_score(self.clf, self.features, self.targets, cv=10)
        print scores

    def predict(self, feature):
        return self.clf.predict(feature)

    def predict_proba(self, feature):
        return self.clf.predict_proba(feature)

def model_main():
    # model = jc_model()
    model = baseline_model()
    model.load_data()
    model.train()
    model.test_model()
    model.save_model()

if __name__ == '__main__':
    model_main()