# coding:utf-8
# sa+ta,sb+tb,lsh(sa+ta),lsh(sb+tb),concat
import math

import numpy as np
import utils
from sklearn.metrics import roc_auc_score

fri_vec = '../data/vec_n2v.txt'
user_file = '../data/sorted_nodes1.txt'
traj_vec = '../data/user_vec.txt'
new_vec = '../data/fri_traj_vec.txt'  # new representation


# traj_vec_dict
traj_dict = {}

t_v = open(traj_vec)
u_f = open(user_file)
for user,t_vector in zip(u_f,t_v):
    traj_dict[user.split('\n')[0]] = t_vector.split('\n')[0]
t_v.close()
u_f.close()

f_v = open(fri_vec)
next(f_v)
n_v = open(new_vec,'w')
n_v.write('start\n')
for line in f_v:
    new_line = line.split('\n')[0]+' '+traj_dict[line.split()[0]]+'\n'
    n_v.write(new_line)
n_v.close()
f_v.close()

from lsh.lshash import LSHash
user_tranj_vec = '../data/fri_traj_vec.txt'

user_lsh = LSHash(512, 256)

vec_lst = []
nodes = []
u_t_v = open(user_tranj_vec)
next(u_t_v)
for line in u_t_v:
    vec_lst.append(map(float,line.replace('\n','').split(' ')[1:]))
    nodes.append(line.split()[0])
u_t_v.close()

def flatten(sublist):
    my_str = ''
    count = 0
    for element in sublist:
        count += 1
        if count == len(sublist):
            my_str += str(element) + '\n'
        else:
            my_str += str(element) + ' '
    return my_str

for embedding in vec_lst:
        user_lsh.index(embedding)

user_hamming_code = user_lsh.hamming_code()
print ''

test1 = '../data/fri_traj_hash_vec.txt'
t1 = open(test1,'w')
for hamming,node in zip(user_hamming_code,nodes):
    t1.write(node+' '+flatten(hamming))
t1.close()

fri_traj_dictionary = {}
t1 = open(test1)
for line in t1.readlines():
    fri_traj_dictionary[line.split()[0]] = line[(len(line.split()[0])+1):-1]
t1.close()

test = '../data/test_undirected.txt'  # test positive
train = '../data/train_undirected.txt'  # train positive
test_un = '../data/test_negative.txt'  # test negative
train_un = '../data/train_negative.txt'  # train negative

fri_traj_train = '../data/fri_traj_train.txt'
fri_traj_test = '../data/fri_traj_test.txt'

f_t_train = open(fri_traj_train,'w')
train_positive = open(train)
for line in train_positive.readlines():
    node_l,node_r = utils.get_nodes(line)
    new_line = fri_traj_dictionary[node_l]+' '+fri_traj_dictionary[node_r]+' 1\n'
    f_t_train.write(new_line)
train_positive.close()
f_t_train.close()

f_t_train = open(fri_traj_train,'a')
train_negative = open(train_un)
for line in train_negative.readlines():
    node_l,node_r = utils.get_nodes(line)
    new_line = fri_traj_dictionary[node_l]+' '+fri_traj_dictionary[node_r]+' 0\n'
    f_t_train.write(new_line)
train_negative.close()
f_t_train.close()

f_t_test = open(fri_traj_test,'w')
test_positive = open(test)
for line in test_positive.readlines():
    node_l,node_r = utils.get_nodes(line)
    new_line = fri_traj_dictionary[node_l]+' '+fri_traj_dictionary[node_r]+' 1\n'
    f_t_test.write(new_line)
test_positive.close()
f_t_test.close()

f_t_test = open(fri_traj_test,'a')
test_negative = open(test_un)
for line in test_negative.readlines():
    node_l,node_r = utils.get_nodes(line)
    new_line = fri_traj_dictionary[node_l]+' '+fri_traj_dictionary[node_r]+' 0\n'
    f_t_test.write(new_line)
test_negative.close()
f_t_test.close()
