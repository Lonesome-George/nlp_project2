#coding=utf-8

from base import raw_trainingset
from preprocess import new_trainingset
from feature import feature
from corpus import gen_training_corpus

from sklearn import svm
from sklearn.svm import LinearSVC, SVC, NuSVR, SVR
from sklearn import multiclass
from sklearn import cross_validation
from sklearn.externals import joblib
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB


class jc_model:

    features = None
    targets  = None
    clf = None

    def __init__(self):
        # self.clf = svm.LinearSVC()
        # self.clf = svm.NuSVR(kernel='sigmoid')
        # self.clf = multiclass.OneVsOneClassifier(SVC(kernel='sigmoid', probability=True)) # 里面的参数可以随意设置
        # self.clf = multiclass.OneVsRestClassifier(SVR())
        self.clf = svm.SVC(kernel='sigmoid', probability=True)

    def load_model(self):
        # self.clf = joblib.load('./pkl/linearsvc.pkl')
        # self.clf = joblib.load('./pkl/nusvr.pkl')
        # self.clf = joblib.load('./pkl/ovoclf.pkl')
        # self.clf = joblib.load('./pkl/ovrclf.pkl')
        self.clf = joblib.load('./pkl/svc.pkl')

    def save_model(self):
        # joblib.dump(self.clf, './pkl/linearsvc.pkl')
        # joblib.dump(self.clf, './pkl/nusvr.pkl')
        # joblib.dump(self.clf, './pkl/ovoclf.pkl')
        # joblib.dump(self.clf, './pkl/ovrclf.pkl')
        joblib.dump(self.clf, './pkl/svc.pkl')

    def load_data(self):
        features = []
        targets = []
        # f_train = open(raw_trainingset, 'r')
        f_train = open(new_trainingset, 'r') # 使用80%训练集
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
        self.clf = BernoulliNB()

    def load_model(self):
        self.clf = joblib.load('./pkl/bernoullinb.pkl')

    def save_model(self):
        joblib.dump(self.clf, './pkl/bernoullinb.pkl')

    def load_data(self):
        features = []
        targets = []
        # f_train = open(raw_trainingset, 'r')
        f_train = open(new_trainingset, 'r')
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