# coding:utf-8
import gc
import random

import utils

BK_file = '../data/subgraph_positive.txt'
BK_edges = '../data/edges_undirected1.txt'
nodes_file = '../data/vec_1.txt'
sorted_nodes = '../data/sorted_nodes1.txt'
nodes_pair = '../data/nodes_pair.txt'
nodes_degree = '../data/nodes_degree1.txt'
train = '../data/train_undirected.txt'
test = '../data/test_undirected.txt'
train_new = '../data/train_double.txt'
test_ne = '../data/test_negative.txt'
train_ne = '../data/train_negative.txt'

# obtain all nodes by embedding
from gensim.models import word2vec

sentence = word2vec.LineSentence('../data/subgraph_positive.txt')
model = word2vec.Word2Vec(sentence, size=5, min_count=0, workers=15)
model.wv.save_word2vec_format('../data/vec_1.txt')


# place the nodes in ascending order
nodes = []
f = open(nodes_file)
next(f)
for line in f:
    nodes.append(int(line.split(' ')[0]))
f.close()
nodes = sorted(nodes)

fs = open(sorted_nodes, 'w')
for node in nodes:
    fs.write(str(node) + '\n')
fs.close()

#
# key is the content in BK_file, value is ‘0’. Using dictionary can accelerate the running time
dictionary = {}
total_len = 0
f = open(BK_file)
for line in f.readlines():
    total_len += 1
    if line.split('\n')[0] in dictionary.keys():
        print 'ok'
    dictionary[line.split('\n')[0]] = '0'
f.close()

# undirected
fe = open(BK_edges, 'w')
ff = open(BK_file)
for line in ff.readlines():
    if dictionary[line.split('\n')[0]] == '0':
        fe.write(line)
        dictionary[line.split('\n')[0]] = '1'
        line_l, line_r = utils.get_nodes(line)
        new_line = line_r + ' ' + line_l
        dictionary[new_line] = '1'
    else:
        pass
fe.close()
ff.close()

# calculate the degree of every node
fs = open(sorted_nodes)
dictionary_node = {}
for line in fs.readlines():
    dictionary_node[line.split('\n')[0]] = '0'

#
fe = open(BK_edges)
for line in fe.readlines():
    line_l, line_r = utils.get_nodes(line)
    assert dictionary_node.has_key(line_l)
    my_str_l = dictionary_node[line_l]
    dictionary_node[line_l] = str(int(my_str_l) + 1)
    assert dictionary_node.has_key(line_r)
    my_str_r = dictionary_node[line_r]
    dictionary_node[line_r] = str(int(my_str_r) + 1)
fe.close()

#
fn = open(nodes_degree, 'w')
for key in dictionary_node.keys():
    value = dictionary_node.get(key)
    node_degree = key + ' ' + value + '\n'
    fn.write(node_degree)
fn.close()

# test set occupy 20%, the rest data set is train set

# create dictionary by node degree
dictionary_node = {}
fn = open(nodes_degree)
for line in fn.readlines():
    node, degree = utils.get_nodes(line)
    dictionary_node[node] = degree
fn.close()

#
inputs = []
fe = open(BK_edges)
for line in fe.readlines():
    inputs.append(line.split('\n')[0])
fe.close()

del_candidate = utils.random_nodes_pair(inputs)


del_lst = []
count = 0
# total_len*0.1 means choose 20% edges as test set
test_len = int(total_len * 0.1)
for del_c in del_candidate:
    node_l, node_r = utils.get_nodes(del_c)
    if int(dictionary_node[node_l]) > 1 and int(dictionary_node[node_r]) > 1:
        del_lst.append(del_c)
        dictionary_node[node_l] = str(int(dictionary_node[node_l]) - 1)
        dictionary_node[node_r] = str(int(dictionary_node[node_r]) - 1)
        count += 1
        if count == test_len:
            break

#
dictionary_test = {}
f_te = open(test, 'w')
test_ne_len = 0
for del_i in del_lst:
    test_ne_len += 1
    dictionary_test[del_i] = '0'
    f_te.write(del_i + '\n')
f_te.close()

fe = open(BK_edges)
f_tr = open(train, 'w')
train_ne_len = 0
for line in fe.readlines():
    if dictionary_test.has_key(line.split('\n')[0]):
        pass
    else:
        f_tr.write(line)
        train_ne_len += 1
f_tr.close()
fe.close()
#
# bi-direction edges
f_tr_new = open(train_new, 'w')
f_tr = open(train)
for line in f_tr.readlines():
    node_l, node_r = utils.get_nodes(line)
    f_tr_new.write(line)
    f_tr_new.write(node_r + ' ' + node_l + '\n')
f_tr.close()
f_tr_new.close()

# generate negative sample
nodes = []
del nodes[:]
gc.collect()
fs = open(sorted_nodes)
for line in fs.readlines():
    nodes.append(line.split('\n')[0])
fs.close()

#
gen_nodes_pair = {}
f = open(nodes_pair, 'w')
pairs = []
# count negative sample
count = 0
for node_l in nodes:
    # if count == 13724:
    #     break
    for node_r in nodes:
        # exclude self
        if node_l != node_r:
            # check whether exits
            if gen_nodes_pair.has_key(str(str(node_l) + ' ' + str(node_r))):
                pass
            else:
                my_str = ''
                my_str_reverse = ''
                my_str += str(node_l) + ' ' + str(node_r)
                my_str_reverse += str(node_r) + ' ' + str(node_l)
                # check whether unrelated
                if dictionary.has_key(my_str):
                    pass
                else:
                    pairs.append(my_str + '\n')
                    gen_nodes_pair[my_str] = '0'
                    gen_nodes_pair[my_str_reverse] = '0'
                    count += 1
        else:
            pass
        # if count == 13724:
        #     break

for pair in pairs:
    f.write(pair)
f.close()

inputs = []
unrelated_len = 0
fn = open(nodes_pair)
for line in fn.readlines():
    unrelated_len += 1
    inputs.append(line.split('\n')[0])
fn.close()


test_unrelated = utils.random_nodes_pair(inputs, float(test_ne_len) / float(len(inputs)))
dictionary_split = {}
f_te_ne = open(test_ne,'w')
for tu in test_unrelated:
    f_te_ne.write(tu + '\n')
    dictionary_split[tu] = '0'
f_te_ne.close()

new_input = []
for ip in inputs:
    if dictionary_split.has_key(ip):
        pass
    else:
        new_input.append(ip)

train_unrelated = utils.random_nodes_pair(new_input, float(train_ne_len) / float(len(new_input)))
f_tr_ne = open(train_ne,'w')
for tu in train_unrelated:
    f_tr_ne.write(tu + '\n')
f_tr_ne.close()
