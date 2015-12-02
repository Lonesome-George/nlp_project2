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

# 取出k个元素
def fetch_elements(elements, k, reverse=False):
    if k >= len(elements):
        return elements
    if not reverse:
        return elements[0:k]
    else:
        return elements[len(elements)-k:len(elements)]

# 将元素列表添加进集合
def add_to_set(e_list, e_set):
    for elem in e_list:
        e_set.add(elem)
    return e_set
