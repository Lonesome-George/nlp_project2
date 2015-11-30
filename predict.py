#coding=utf-8

from base import relation_list
from corpus import gen_training_corpus, gen_test_corpora
from feature import feature
from model import baseline_model, jc_model

test_file = './TrainingSet/project2_TrainingSet7000'
cls_model = jc_model()
# cls_model = baseline_model()
cls_model.load_model()

def predict_main():
    fi_test = open(test_file, 'r')
    pred_result = {}
    total_preds = 0
    for line in fi_test:
        corpus = gen_training_corpus(line)
        test_corpora = gen_test_corpora(corpus.title)
        if test_corpora == None: continue
        total_preds += 1
        max_proba = 0.0
        pred_label = -1
        for corpus in test_corpora:
            features = feature(corpus.title, corpus.person1, corpus.person2)
            cls = cls_model.predict_proba(features)
            for i in range(len(relation_list)):
                if cls[0][i] > max_proba:
                    max_proba = cls[0][i]
                    pred_label = i
        if not pred_result.has_key(pred_label):
            pred_result[pred_label] = 0
        pred_result[pred_label] += 1
        print pred_label, max_proba
        print '--------------'
    fi_test.close()
    print total_preds, pred_result

if __name__ == '__main__':
    predict_main()