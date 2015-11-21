#coding=utf-8

def proc_line(line, sep):
    line = line.rstrip('\n')
    seg_list = line.split(sep)
    return seg_list

def divide_line(line, sep1, sep2):
    # 确保sep1在sep2的左边
    idx1 = line.find(sep1)
    idx2 = line.find(sep2)
    if idx1 > idx2:
        temp = sep1
        sep1 = sep2
        sep2 = temp
    sp_list = proc_line(line, sep1)
    left = sp_list[0]
    sp_list = proc_line(sp_list[1], sep2)
    if len(sp_list) == 1:
        middle = sp_list[0]
        right = ''
    else:
        middle = sp_list[0]
        right = sp_list[1]
    return [left, middle, right]

# # 分词
# def tokenize(text):
#     text = text.rstrip('\n') # 去除行尾的换行符
#     seg_list = list(jieba.cut(text, cut_all=False))# 精确模式
#     seg_newlist = []
#     for seg in seg_list:
#         seg = seg.strip()# 去除单词前后的空格
#         if seg != '':    # 如果整个单词由空格构成则删除
#             seg_newlist.append(seg)
#     return seg_newlist
