#coding=utf-8

import os
from base import *
from utils import divide_line
from corpus import gen_training_corpus
from nlpir import Seg

trainset_dir = "./TrainingSet"
trainset_prefix = "project2_TrainingSet7000_"

# 将每一种关系实例归类
if __name__ == '__main__':
    fi_train = open(raw_trainingset, 'r')
    # 为每一种关系建立一个预处理文件
    file_out_list = []
    fo_relation_list = []
    for i in range(len(relation_list)):
        filename = trainset_prefix + str(i)
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