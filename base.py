#coding=utf-8

raw_trainingset = './TrainingSet/project2_TrainingSet7000'
proced_trainingset = './TrainingSet/project2_TrainingSet7000_Proced'

# 关系负例也当作一种关系
relation_list = ['同为校花', '昔日情敌', '老师', '闺蜜','撞衫','同学', '前女友', '经纪人', '妻子', '分手',
                 '偶像', '老乡', '暧昧', '翻版', '同居', '绯闻女友', '传闻不和', '前妻', '朋友', 'null']

def fetch_label(rel):
    length = len(relation_list)
    for i in range(length):
        if rel == relation_list[i]:
            return i
    return -1

# log
import logging
import logging.config

LOG_FILENAME = 'logging.conf'
logging.config.fileConfig(LOG_FILENAME)
logger = logging.getLogger("NLP")