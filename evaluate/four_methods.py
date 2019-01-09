# coding:utf-8
import numpy as np
import math
from sklearn.metrics import roc_auc_score

file_vec = '../data/vec_n2v.txt'
test = '../data/test_undirected.txt'
train = '../data/train_undirected.txt'
test_un = '../data/test_negative.txt'
train_un = '../data/train_negative.txt'
user_file = '../data/sorted_nodes1.txt'
add_vec_train = '../data/add_vec_train.txt'
add_vec_test = '../data/add_vec_test.txt'
mul_vec_train = '../data/mul_vec_train.txt'
mul_vec_test = '../data/mul_vec_test.txt'
red_vec_train = '../data/red_vec_train.txt'
red_vec_test = '../data/red_vec_test.txt'
squ_vec_train = '../data/squ_vec_train.txt'
squ_vec_test = '../data/squ_vec_test.txt'

f = open(file_vec)
dictionary = {}
next(f)
for line in f:
    dictionary[line.split()[0]] = line.split()[1:]
f.close()

user_lst = []
u_f = open(user_file)
for line in u_f.readlines():
    user_lst.append(line.split('\n')[0])
u_f.close()


def length(v):
    return math.sqrt(np.dot(v, v))


def get_nodes(line):
    new_str = line.split('\n')[0]
    new_list = new_str.split(' ')
    line_l = new_list[0]
    line_r = new_list[1]
    return line_l, line_r


def add_calc(node_l, node_r):
    vec_lst_1 = []
    for element in dictionary[node_l]:
        vec_lst_1.append(float(element))

    vec_1 = np.array(vec_lst_1)

    vec_lst_2 = []
    for element in dictionary[node_r]:
        vec_lst_2.append(float(element))

    vec_2 = np.array(vec_lst_2)

    vec_add = (vec_1 + vec_2) / 2
    vec_mul = vec_1 * vec_2
    vec_red = abs(vec_1 - vec_2)
    vec_squ = np.power(vec_red, 2)
    return vec_add, vec_mul, vec_red, vec_squ


def write_file(file_name, lst_content, pos_neg):
    for lc in lst_content:
        file_name.write(str(lc) + ' ')
    file_name.write(pos_neg)


def gen_vec_file(target_file, original_file, vec_method, pos_neg):
    f_t = open(target_file, 'a')

    count = 0
    f_o = open(original_file)
    for line in f_o.readlines():
        node_l, node_r = get_nodes(line)
        va, vm, vr, vs = add_calc(node_l, node_r)
        if vec_method == 'add':
            positive_sample = list(va)
            write_file(f_t, positive_sample, pos_neg)
            count += 1
        elif vec_method == 'mul':
            positive_sample = list(vm)
            write_file(f_t, positive_sample, pos_neg)
            count += 1
        elif vec_method == 'red':
            positive_sample = list(vr)
            write_file(f_t, positive_sample, pos_neg)
            count += 1
        else:
            positive_sample = list(vs)
            write_file(f_t, positive_sample, pos_neg)
            count += 1
    f_o.close()
    f_t.close()


gen_vec_file(add_vec_test, test, 'add', '1\n')
gen_vec_file(add_vec_test, test_un, 'add', '0\n')
gen_vec_file(add_vec_train, train, 'add', '1\n')
gen_vec_file(add_vec_train, train_un, 'add', '0\n')

gen_vec_file(mul_vec_test, test, 'mul', '1\n')
gen_vec_file(mul_vec_test, test_un, 'mul', '0\n')
gen_vec_file(mul_vec_train, train, 'mul', '1\n')
gen_vec_file(mul_vec_train, train_un, 'mul', '0\n')

gen_vec_file(red_vec_test, test, 'red', '1\n')
gen_vec_file(red_vec_test, test_un, 'red', '0\n')
gen_vec_file(red_vec_train, train, 'red', '1\n')
gen_vec_file(red_vec_train, train_un, 'red', '0\n')

gen_vec_file(squ_vec_test, test, 'squ', '1\n')
gen_vec_file(squ_vec_test, test_un, 'squ', '0\n')
gen_vec_file(squ_vec_train, train, 'squ', '1\n')
gen_vec_file(squ_vec_train, train_un, 'squ', '0\n')