#coding=utf-8

from __future__ import division
from base import relation_list, logger
from preprocess import validation_set, proced_trainingset
from corpus import gen_training_corpus, gen_test_corpora
from feature import feature
from model import baseline_model, jc_model
from nlpir import Seg

test_file = './TestSet/project2_TestSet3000'
test_result = './TestSet/test_result'

def predict(cls_model):
    fi_test = open(test_file, 'r')
    fo_result = open(test_result, 'w')
    total_insts = 0  # 预测样本的实例数
    pred_insts = []  # 预测的实例数目
    for label in range(len(relation_list)):
        pred_insts.append(0)
    for line in fi_test:
        title = line.rstrip('\n')
        test_corpora = gen_test_corpora(title)
        if test_corpora == None:
            fo_result.write('null\tnull\tnull\t%s\t0\n' % title)
            continue
        total_insts += 1
        person1 = 'null'
        person2 = 'null'
        max_proba = 0.0
        pred_label = -1
        for corpus in test_corpora:
            features = feature(corpus.title, corpus.person1, corpus.person2)
            pred_cls = cls_model.predict_proba(features)
            for label in range(len(relation_list)):
                if pred_cls[0][label] > max_proba:
                    person1 = corpus.person1
                    person2 = corpus.person2
                    max_proba = pred_cls[0][label]
                    pred_label = label
            # if pred_label > -1 and pred_label < 19:
            #     fo_result.write('%s\t%s\t%s\t%s\t1\n' % (relation_list[pred_label], corpus.person1, corpus.person2, title))
            # else:
            #     fo_result.write('null\t%s\t%s\t%s\t0\n' % (corpus.person1, corpus.person2, title))
        pred_insts[pred_label] += 1
        if pred_label > -1 and pred_label < 19:
            fo_result.write('%s\t%s\t%s\t%s\t1\n' % (relation_list[pred_label], person1, person2, title))
        else:
            fo_result.write('null\tnull\tnull\t%s\t0\n' % (title))
    fi_test.close()
    fo_result.close()
    print total_insts, pred_insts

# validate
def validate(cls_model):
    fi_validate = open(validation_set, 'r')
    # fo_validate_result = open('./validate_result.txt', 'a+')
    # fo_validate_result.write('Relation,Precision,Recall,F1\n')
    print 'Relation,Precision,Recall,F1'
    # fi_validate = open(proced_trainingset, 'r')
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
    total_precision = 0.0
    total_recall = 0.0
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
        if label < 19:
            total_precision += precision
            total_recall += recall
            total_fmeasure += fmeasure
        print '%s,%f,%f,%f' \
              % (relation_list[label], precision, recall, fmeasure)
        # fo_validate_result.write('%s,%f,%f,%f\n' \
        #       % (relation_list[label], precision, recall, fmeasure))
    #     logger.info('[%s]%s:%d, %s=%f, %s=%f, %s=%f' \
    #           % (relation_list[label], 'total instances', total_insts[label],
    #              'precision', precision, 'recall', recall,
    #              'f-measure', fmeasure))
    # logger.info('average f-measure: %f' % (total_fmeasure / 19))
    print '%f,%f,%f' % (total_precision / 19, total_recall / 19, total_fmeasure / 19)
    # print 'average precision: %f' % (total_precision / 19)
    # print 'average recall: %f' % (total_recall / 19)
    # print 'average f-measure: %f' % (total_fmeasure / 19)
    # print '  total instances', total_insts
    # print 'predict instances', pred_insts
    # print '  right instances', right_insts

# 提取出不能识别出人名的语料
def fetch_corpora_nner():
    fi_test = open(test_file, 'r')
    fo_copora_nner = open('./TestSet/corpora_nner.txt', 'w')
    fo_copora_nner.write('title\t已经识别出来的人名\n')
    for line in fi_test:
        title = line.rstrip('\n')
        persons = []
        for t in Seg(title):
            if t[1] == 'nr': # 识别人名
                persons.append(t[0])
        length = len(persons)
        if length < 2: # 无法识别出两个人名
            fo_copora_nner.write('%s==>' % title)
            for person in persons:
                fo_copora_nner.write('%s\t' % person)
            fo_copora_nner.write('\n')
    fi_test.close()
    fo_copora_nner.close()

if __name__ == '__main__':
    # cls_model = jc_model()
    cls_model = baseline_model()
    cls_model.load_model()
    # predict(cls_model)
    validate(cls_model)
    # fetch_copora_nner()