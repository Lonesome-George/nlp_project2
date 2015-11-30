#coding=utf-8

# 选取以下四类特征
# 人物实体对的上下文特征: 两个人物实体之间的 W 个词语、 前一个人物实体的左边的两个词语和后一个人物实体右边的两个词语。
# 实体上下文的词性: 人物实体前后 W 个词的词性。
# 距离特征: 前一个人物实体和后一个人物实体之间的词距。
# 句法特征: 人物实体之间的实体对与句法路径的最近公共祖先。

import os
from base import *
from utils import proc_line, divide_line
from nlpir import Seg

data_dir = "./TrainingSet"
trainset_prefix = "project2_TrainingSet7000_"
featureset_dir = os.path.join(data_dir, "featureset")
total_featset_prefix = "total_featureset_"
extracted_featset_prefix = "extracted_featureset_"

W = 1

# 特征名称
feature_name_list = ['left_words', 'middle_words', 'right_words',
                     'p1_left_pos', 'p1_right_pos', 'p2_left_pos', 'p2_right_pos',
                     '2p_dist', 'nearest_common_ancestor']

# 特征全集
total_featureset_dict = {}
# 每一种关系的每一条新闻标题对应的特征
featureset_dict = {}
# 抽取的特征词全集
extracted_featureset_dict = {}

def init_feature_set():
    for relation in relations:
        total_featureset_dict[relation] = dict()
        featureset_dict[relation] = dict()
        for feature_name in feature_name_list:
            total_featureset_dict[relation][feature_name] = set()
            featureset_dict[relation][feature_name] = []

# 处理第一个人物实体左边的文本,生成该文本的词语和词性
def proc_left_segments(relation, segments):
    seg_list = proc_line(segments, ';')
    word_list = []
    pos_list =[]
    for seg in seg_list:
        if seg == '': continue
        sp_list = proc_line(seg, ':')
        word = sp_list[0]
        pos = sp_list[1]
        word_list.append(word)
        pos_list.append(pos)
        total_featureset_dict[relation]['left_words'].add(word)
        total_featureset_dict[relation]['p1_left_pos'].add(pos)
    featureset_dict[relation]['left_words'].append(word_list)
    featureset_dict[relation]['p1_left_pos'].append(pos_list)

# 处理两个人物实体之间的文本,生成该文本的词语和词性
def proc_middle_segments(relation, segments):
    seg_list = proc_line(segments, ';')
    word_list = []
    pos_list =[]
    for seg in seg_list:
        if seg == '': continue
        sp_list = proc_line(seg, ':')
        word = sp_list[0]
        pos = sp_list[1]
        word_list.append(word)
        pos_list.append(pos)
        total_featureset_dict[relation]['middle_words'].add(word)
        total_featureset_dict[relation]['p1_right_pos'].add(pos)
        total_featureset_dict[relation]['p2_left_pos'].add(pos)
    featureset_dict[relation]['middle_words'].append(word_list)
    featureset_dict[relation]['p1_right_pos'].append(pos_list)
    featureset_dict[relation]['p2_left_pos'].append(pos_list)

# 处理第二个人物实体右边的文本,生成该文本的词语和词性
def proc_right_segments(relation, segments):
    seg_list = proc_line(segments, ';')
    word_list = []
    pos_list =[]
    for seg in seg_list:
        if seg == '': continue
        sp_list = proc_line(seg, ':')
        word = sp_list[0]
        pos = sp_list[1]
        word_list.append(word)
        pos_list.append(pos)
        total_featureset_dict[relation]['right_words'].add(word)
        total_featureset_dict[relation]['p2_right_pos'].add(pos)
    featureset_dict[relation]['right_words'].append(word_list)
    featureset_dict[relation]['p2_right_pos'].append(pos_list)

# 生成特征全集
def gen_total_featureset(relation):
    filename = trainset_prefix + relation
    file_in = os.path.join(data_dir, filename)
    fi = open(file_in.decode('utf-8'), 'r')
    for line in fi:
        seg_list = proc_line(line, '||')
        proc_left_segments(relation, seg_list[0])
        proc_middle_segments(relation, seg_list[1])
        proc_right_segments(relation, seg_list[2])
    fi.close()

# 保存特征全集至文件
def save_total_featureset():
    if not os.path.isdir(featureset_dir):
        os.mkdir(featureset_dir)
    for feature_name in feature_name_list:
        filename = total_featset_prefix + feature_name
        file_out = os.path.join(featureset_dir, filename)
        fo = open(file_out.decode('utf-8'), 'w')
        feature_set = set()
        for relation in relations:
            featset_dict = total_featureset_dict[relation]
            feature_set = feature_set.union(featset_dict[feature_name])
        string = ""
        for feature in feature_set:
            string += feature + ';'
        fo.write(string)
        fo.close()

# 统计一个特征在特征集中的频数分布
def stat_chi_dist(feature, featureset_list):
    A = C = 0 # A表示包含词w的标题数目,C表示不包含词w的标题数目
    for featureset in featureset_list:
        if feature in featureset:
            A += 1
        else:
            C += 1
    return A,C

# 使用标准CHI算法计算每个单词的得分,cls_featureset_list表示本类特征集列表,other_featureset_list表示其他类特征集列表
def std_chi_scores(featureset, cls_featureset_list, other_featureset_list):
    scores = dict()
    for feat in featureset:
        A,C = stat_chi_dist(feat, cls_featureset_list)  # A表示包含词w并且属于类c的标题数目,C表示不包含词w但属于类c的标题数目
        B,D = stat_chi_dist(feat, other_featureset_list)# B表示包含词w但不属于类c的标题数目,D表示不包含词w并且不属于类c的标题数目
        # print A,B,C,D
        scores[feat] = (B*C - A*D)**2 / ((A+B)*(C+D)+1) # 标准CHI，分母加1防止出现分母为零的情况
    return scores

# 抽取特征
def extract_feature():
    # 为每一种关系的每一种特征抽取出特定数目的特征词
    for feature_name in feature_name_list:
        extracted_featset = set() # 抽取的特征列表
        # 遍历关系种类列表,每一种关系都抽取出一定数目的特征词
        for rel1 in relations:
            rel_featset_dict = total_featureset_dict[rel1]
            # 构造这种关系对应的本特征集和其他特征集
            cls_featset_list = featureset_dict[rel1][feature_name]
            other_featset_list = []
            for rel2 in relations:
                if rel1 == rel2: continue
                other_featset_list.append(featureset_dict[rel2][feature_name])
            rel_featset = rel_featset_dict[feature_name]
            scores = std_chi_scores(rel_featset, cls_featset_list, other_featset_list)
            scores_list = sorted(scores.items(), key=lambda x:x[0], reverse=True)
            # 取出前面W个特征词
            for feat, score in scores_list[0:W]:
                extracted_featset.add(feat)
        extracted_featureset_dict[feature_name] = extracted_featset
        # 将抽取的特征集写入文件
        filename = extracted_featset_prefix + feature_name
        file_out = os.path.join(featureset_dir, filename)
        fo = open(file_out, 'w')
        string = ""
        for extracted_feat in extracted_featset:
            string += extracted_feat + ';'
        string += '\n'
        fo.write(string)
        fo.close()

# 读取抽取的特征集
def read_extracted_featureset():
    for feature_name in feature_name_list:
        filename = extracted_featset_prefix + feature_name
        file_in = os.path.join(featureset_dir, filename)
        fi = open(file_in, 'r')
        line = fi.readline()
        seg_list = proc_line(line, ';')
        featset = []
        for seg in seg_list:
            if seg == '': continue
            featset.append(seg)
        extracted_featureset_dict[feature_name] = featset
        fi.close()

# 将新闻标题表示成特征
def feature(title, person1, person2):
    if len(extracted_featureset_dict) == 0:
        # 读取文件
        read_extracted_featureset()
    sp_list = divide_line(title, person1, person2)
    word_list = []
    pos_list = []
    idx = 0
    features = []
    for sp in sp_list:
        if sp == '':
            sp = ' ' # 以防下面调用Seg()报错
        for t in Seg(sp):
            word_list.append(t[0])
            pos_list.append(t[1])
        idx += 1
        if idx == 1:
            feats = sub_feature(word_list, extracted_featureset_dict['left_words'])
            add_features(feats, features)
            feats = sub_feature(pos_list, extracted_featureset_dict['p1_left_pos'])
            add_features(feats, features)
        elif idx == 2:
            feats = sub_feature(word_list, extracted_featureset_dict['middle_words'])
            add_features(feats, features)
            feats = sub_feature(pos_list, extracted_featureset_dict['p1_right_pos'])
            add_features(feats, features)
            feats = sub_feature(pos_list, extracted_featureset_dict['p2_left_pos'])
            add_features(feats, features)
        elif idx == 3:
            feats = sub_feature(word_list, extracted_featureset_dict['right_words'])
            add_features(feats, features)
            feats = sub_feature(pos_list, extracted_featureset_dict['p2_right_pos'])
            add_features(feats, features)
    return features

# 生成子特征
def sub_feature(featureset, total_featureset):
    features = []
    for feat in total_featureset:
        if feat in featureset:
            features.append(1)
        else:
            features.append(0)
    return features

# 将子特征加入特征向量
def add_features(feats, features):
    for feat in feats:
        features.append(feat)

if __name__ == '__main__':
    init_feature_set()
    for relation in relations:
        gen_total_featureset(relation)
    save_total_featureset()
    extract_feature()
    # features = feature('成龙羡慕房祖名签到自己偶像', '成龙', '房祖名')
    # features = feature('黄义达与朱孝天前女友佐藤麻衣擦出爱火花(图)', '朱孝天', '佐藤麻衣')
    # print features
