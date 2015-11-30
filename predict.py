#coding=utf-8

from base import relations
from corpus import gen_training_corpus, gen_test_corpora
from feature import feature
from model import jc_model

test_file = './TrainingSet/project2_TrainingSet7000'
classfier = jc_model()
classfier.load_model()

def predict_main():
    fi_test = open(test_file, 'r')
    for line in fi_test:
        corpus = gen_training_corpus(line)
        test_corpora = gen_test_corpora(corpus.title)
        if test_corpora == None: continue
        print '--------------'
        max_proba = 0.0
        pred_label = -1
        for corpus in test_corpora:
            feats = feature(corpus.title, corpus.person1, corpus.person2)
            cls = classfier.predict_proba(feats)
            for i in range(len(relations)):
                if cls[0][i] > max_proba:
                    max_proba = cls[0][i]
                    pred_label = i
        print pred_label, max_proba
        print '--------------'
    fi_test.close()

if __name__ == '__main__':
    predict_main()