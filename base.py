#coding=utf-8

raw_trainingset = './TrainingSet/project2_TrainingSet7000'

# 关系负例也当作一种关系
relations = ['同为校花', '昔日情敌', '老师', '闺蜜','撞衫','同学', '前女友', '经纪人', '妻子', '分手',
            '偶像', '老乡', '暧昧', '翻版', '同居', '绯闻女友', '传闻不和', '前妻', '朋友', 'null']

def fetch_label(rel):
    length = len(relations)
    for i in range(length):
        if rel == relations[i]:
            return i
    return -1