#coding=utf-8

import os
from base import *
from utils import proc_line, divide_line
from corpus import gen_training_corpus
from nlpir import Seg

trainset_dir = "./TrainingSet"
parsing_prefix = "parsing_"
new_trainingset = './TrainingSet/project2_TrainingSet7000_New'
validation_set = './TrainingSet/project2_ValidationSet'

# 预处理训练集,保存成person1\tperson2\ttitle\tlabel的格式
def preproc_trainset():
    fi_raw_train = open(raw_trainingset, 'r')
    fo_proced_train = open(proced_trainingset, 'w')
    # 读取训练集
    for line in fi_raw_train:
        seg_list = proc_line(line, '\t')
        relation = seg_list[0].strip()
        person1 = seg_list[1].strip()
        person2 = seg_list[2].strip()
        title = seg_list[3].strip()
        has_rel = seg_list[4].strip() # 是否的确有对应关系
        if has_rel == '0':
            relation = 'null'
        label = fetch_label(relation)
        fo_proced_train.write('%s\t%s\t%s\t%d\n' %(person1, person2, title, label))
    fi_raw_train.close()
    fo_proced_train.close()

# 处理训练集，将每一种关系实例归类
def proc_trainset():
    fi_train = open(proced_trainingset, 'r')
    # 为每一种关系建立一个预处理文件
    file_out_list = []
    fo_relation_list = []
    for i in range(len(relation_list)):
        filename = parsing_prefix + str(i)
        filename = os.path.join(trainset_dir, filename)
        file_out_list.append(filename)
        fo_relation_list.append(open(filename.decode('utf-8'), 'w'))
    # 读取训练集
    for line in fi_train:
        corpus = gen_training_corpus(line)
        sp_list = divide_line(corpus.title, corpus.person1, corpus.person2)# 将新闻标题以人名为分隔符划分成3部分
        string = ""
        idx = 0
        for sp in sp_list:
            if sp != '':
                for t in Seg(sp): # 调用中科院的分词
                    s = '%s:%s;' % (t[0],t[1])
                    string += s
            idx += 1
            if idx < len(sp_list):
                string += '||'
        string += '\n'
        fo_relation_list[corpus.label].write(string)
    fi_train.close()
    for fo in fo_relation_list:
        fo.close()

# divide training set into 2 parts: one for training, the other for validating
def divide_trainset():
    stat_samples = [] # 每一类关系的训练样本数
    samples_in_trainset = [] # 当前写入新训练集中的样本数
    for label in range(len(relation_list)):
        stat_samples.append(0)
        samples_in_trainset.append(0)
    fo_new_trainset = open(new_trainingset, 'w')
    fo_validation_set = open(validation_set, 'w')
    fi_train = open(proced_trainingset, 'r')
    for line in fi_train:
        corpus = gen_training_corpus(line)
        stat_samples[corpus.label] += 1
    fi_train.seek(0)
    for line in fi_train:
        corpus = gen_training_corpus(line)
        if samples_in_trainset[corpus.label] > 0.8*stat_samples[corpus.label]:
            fo_validation_set.write(line)
        else:
            fo_new_trainset.write(line)
            samples_in_trainset[corpus.label] += 1
    fi_train.close()
    fo_new_trainset.close()
    fo_validation_set.close()

if __name__ == '__main__':
    # preproc_trainset()
    # proc_trainset()
    # divide_trainset()
    pass