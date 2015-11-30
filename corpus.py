#coding=utf-8

from base import fetch_label
from utils import proc_line
from nlpir import Seg

# class for training corpus and test corpus
class corpus:
    title = None
    person1 = None
    person2 = None
    label = None

    def __init__(self, title, person1, person2, label=None):
        self.title = title
        self.person1 = person1
        self.person2 = person2
        self.label = label

def gen_training_corpus(text):
    seg_list = proc_line(text, '\t')
    relation = seg_list[0].strip()
    person1 = seg_list[1].strip()
    person2 = seg_list[2].strip()
    title = seg_list[3].strip()
    has_rel = seg_list[4].strip() # 是否的确有对应关系
    if has_rel == '0':
        relation = 'null'
    label = fetch_label(relation)
    return corpus(title, person1, person2, label)

def gen_test_corpora(title):
    persons = []
    test_corpora = []
    # 进行词法分析
    for t in Seg(title):
        if t[1] == 'nr': # 识别人名
            persons.append(t[0])
    length = len(persons)
    if length < 2: # 无法识别出两个人名
        return None
    # 两两组合人名,生成候选关系集合
    for i in range(length):
        for j in range(i+1, length):
            test_corpora.append(corpus(title, persons[i], persons[j]))
    return test_corpora