#coding=utf-8

import os
from base import *
from utils import proc_line, divide_line
from nlpir import Seg

data_dir = "./TrainingSet"
trainset = "./TrainingSet/project2_TrainingSet7000"
trainset_prefix = "project2_TrainingSet7000_"

# 将每一种关系实例归类
if __name__ == '__main__':
    fi_train = open(trainset, 'r')
    # 为每一种关系建立一个预处理文件
    file_out_list = []
    fo_relation_dict = {}
    for relation in relations:
        filename = trainset_prefix + relation
        filename = os.path.join(data_dir, filename)
        file_out_list.append(filename)
        fo_relation_dict[relation] = open(filename.decode('utf-8'), 'w')
    # 读取训练集
    for line in fi_train:
        seg_list = proc_line(line, '\t')
        relation = seg_list[0]
        person1 = seg_list[1]
        person2 = seg_list[2]
        title = seg_list[3]
        label = seg_list[4] # 是否的确是对应的关系
        sp_list = divide_line(title, person1, person2)# 将新闻标题以人名为分隔符划分成3部分
        # segments_list = []  # 每部分的分词结果
        string = ""
        idx = 0
        for sp in sp_list:
            # segments_list.append(Seg(sp))
            if sp != '':
                for t in Seg(sp): # 调用中科院的分词
                    s = '%s:%s;' % (t[0],t[1])
                    string += s
            idx += 1
            if idx < len(sp_list):
                string += '||'
        string += '\n'
        filename = relation
        if label == '0': # 关系负例
            filename = label
        fo_relation_dict[filename].write(string)
    fi_train.close()
    for filename, fo in fo_relation_dict.items():
        fo.close()