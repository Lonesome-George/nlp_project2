#coding=utf-8

from __future__ import division
from base import relation_list, logger
from preprocess import validation_set, proced_trainingset
from corpus import gen_training_corpus, gen_test_corpora
from feature import feature
from model import baseline_model, jc_model

test_file = './TrainingSet/project2_TrainingSet7000_Proced'
# cls_model = jc_model()
cls_model = baseline_model()
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
            pred_cls = cls_model.predict_proba(features)
            for i in range(len(relation_list)):
                if pred_cls[0][i] > max_proba:
                    max_proba = pred_cls[0][i]
                    pred_label = i
        if not pred_result.has_key(pred_label):
            pred_result[pred_label] = 0
        pred_result[pred_label] += 1
        print pred_label, max_proba
        print '--------------'
    fi_test.close()
    print total_preds, pred_result

# validate
def validate():
    # fi_validate = open(validation_set, 'r')
    fi_validate = open(proced_trainingset, 'r')
    # pred_result = {}
    total_insts = [] # 预测样本中每一类关系的实例数
    pred_insts = []  # 预测的实例数目
    right_insts = [] # 预测正确的实例数目
    for label in range(len(relation_list)):
        total_insts.append(0)
        pred_insts.append(0)
        right_insts.append(0)
    for line in fi_validate:
        corpus = gen_training_corpus(line)
        total_insts[corpus.label] += 1
        max_proba = 0.0
        pred_label = -1
        # predict
        features = feature(corpus.title, corpus.person1, corpus.person2)
        pred_cls = cls_model.predict_proba(features)
        for i in range(len(relation_list)):
            if pred_cls[0][i] > max_proba:
                max_proba = pred_cls[0][i]
                pred_label = i
        pred_insts[pred_label] += 1
        if pred_label == corpus.label:
            right_insts[corpus.label] += 1
    fi_validate.close()
    total_fmeasure = 0.0
    for label in range(len(relation_list)):
        if pred_insts[label] == 0:
            precision = 0.0
        else:
            precision = right_insts[label] / pred_insts[label]
        recall = right_insts[label] / total_insts[label]
        if precision + recall == 0:
            fmeasure = 0.0
        else:
            fmeasure = 2 * precision * recall / (precision + recall)
        total_fmeasure += fmeasure
        logger.info('[%s]%s:%d, %s=%f, %s=%f, %s=%f' \
              % (relation_list[label], 'total instances', total_insts[label],
                 'precision', precision, 'recall', recall,
                 'f-measure', fmeasure))
    logger.info('average f-measure: %f' % (total_fmeasure / len(relation_list)))

if __name__ == '__main__':
    # predict_main()
    validate()