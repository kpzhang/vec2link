# coding:utf-8
import utils
# delete user records < 10 and poi < 5
# adj_dictionary:user [poi_1,poi_2...poi_n]
# poi_dictionary save: poi-id poi-frequency
file_bk_traj = '../data/poi_filter2.txt'

adj_dictionary = {}
poi_dictionary = {}
f_n = open(file_bk_traj)
i = 0
for line in f_n.readlines():
    user = line.replace('\n','').split(' ')[0]
    location = line.replace('\n','').split(' ')[4]
    i += 1
    print i
    # edges_dictionary[user+' '+location] = ''
    if utils.key_in_dic(location, poi_dictionary):
        poi_dictionary[location] += 1
    else:
        poi_dictionary[location] = 0

    if utils.key_in_dic(user,adj_dictionary):
        adj_dictionary[str(user)].append(location)
    else:
        adj_dictionary[str(user)] = list()
        adj_dictionary[str(user)].append(location)
f_n.close()

# user records < 10
condition1 = 0
user_lst = []
adj_dictionary1 = {}
for key,value in adj_dictionary.iteritems():
    if len(adj_dictionary[key]) < 10:
        condition1 += 1
        user_lst.append(key)
    else:
        adj_dictionary1[key] = value

print condition1

#
file_filter1 = '../data/brightkite_filter1.txt'
b_k_t = open(file_bk_traj)
f_f_1 = open(file_filter1,'w')
i = 0
for line in b_k_t.readlines():
    user = line.replace('\n', '').split(' ')[0]
    location = line.replace('\n', '').split(' ')[4]
    i += 1
    print i

    if utils.key_in_dic(user, adj_dictionary1):
        f_f_1.write(line)
    else:
        pass
b_k_t.close()
f_f_1.close()

# poi < 5
condition2 = 0
poi_lst = []
poi_dictionary2 = {}
for key,value in poi_dictionary.iteritems():
    if value < 1:
        condition2 += 1
        poi_lst.append(key)
    else:
        poi_dictionary2[key] = value

print 'filter2'

file_filter2 = '../data/brightkite_filter2.txt'
f_f_1 = open(file_filter1)
f_f_2 = open(file_filter2,'w')
i = 0
for line in f_f_1.readlines():
    user = line.replace('\n', '').split(' ')[0]
    location = line.replace('\n', '').split(' ')[4]
    i += 1
    print i

    if utils.key_in_dic(location, poi_dictionary2):
        f_f_2.write(line)
    else:
        pass
f_f_1.close()
f_f_2.close()

#
file_traj = '../data/brightkite_filter2.txt'
user_dicitonary = {}
f_t = open(file_traj)
for line in f_t.readlines():
    user = line.replace('\n', '').split(' ')[0]
    # location = line.replace('\n', '').split(' ')[5]
    user_dicitonary[user] = ''
f_t.close()

file_G = '../data/Gowalla_edges.txt'
file_N = '../data/Brightkite_edges_regular.txt'
# ‘\t’ replaced by‘ ’
f = open(file_G)
f_n = open(file_N, 'w')
for line in f.readlines():
    new_line = line.replace('\t', ' ')
    f_n.write(new_line)
f_n.close()
f.close()

# social_edges process
file_social = '../data/Brightkite_edges_regular.txt'
file_social_filter = '../data/Brightkite_edges.txt'
f_s = open(file_social)
f_s_f = open(file_social_filter,'w')
for line in f_s.readlines():
    line_l,line_r = utils.get_nodes(line)
    if utils.key_in_dic(line_l,user_dicitonary) and utils.key_in_dic(line_r,user_dicitonary):
        f_s_f.write(line)
    else:
        pass
f_s.close()
f_s_f.close()


# sub-graph which is connected
BK_file = '../data/Brightkite_edges.txt'
BK_edges = '../data/edges_undirected.txt'
nodes_file = '../data/vec_all.txt'
sorted_nodes = '../data/sorted_nodes.txt'
nodes_degree = '../data/nodes_degree.txt'

# obtain all nodes
from gensim.models import word2vec

sentence = word2vec.LineSentence('../data/Brightkite_edges.txt')
model = word2vec.Word2Vec(sentence, size=3, min_count=0, workers=15)
model.wv.save_word2vec_format('../data/vec_all.txt')

# nodes in ascending order
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
i = 0
f = open(BK_file)
for line in f.readlines():
    dictionary[line.split('\n')[0]] = '0'
    i += 1
    print 'related edges:', i
f.close()

#
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

# calculate node degree
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

import linecache
file_N = '../data/Brightkite_edges.txt'
file_nodes_degree = '../data/nodes_degree.txt'
file_sorted_degree = '../data/sorted_degree.txt'
file_subgraph_pos = '../data/subgraph_positive.txt'
file_subgraph_neg = '../data/subgraph_negative.txt'

#
dictionary_degree = {}
f_n_d = open(file_nodes_degree)
for line in f_n_d.readlines():
    new_line = line.replace('\n','')
    dictionary_degree[new_line.split(' ')[0]] = int(new_line.split(' ')[1])

f_s_d = open(file_sorted_degree,'w')
sorted_degree_len = 0
for k,v in sorted(dictionary_degree.iteritems(), key=lambda asd:asd[1], reverse=True):
    f_s_d.write(str(k)+' '+str(v)+'\n')
    sorted_degree_len += 1
f_s_d.close()

# generate sub-graph
adj_dictionary = {}
edges_dictionary = {}
f_n = open(file_N)
for line in f_n.readlines():
    node_a = line.replace('\n','').split(' ')[0]
    node_b = line.replace('\n','').split(' ')[1]
    edges_dictionary[node_a+' '+node_b] = ''
    if utils.key_in_dic(node_a,adj_dictionary):
        adj_dictionary[str(node_a)].append(node_b)
    else:
        adj_dictionary[str(node_a)] = list()
        adj_dictionary[str(node_a)].append(node_b)
f_n.close()

dictionary_degree = {}
f_n_d = open(file_nodes_degree)
for line in f_n_d.readlines():
    new_line = line.replace('\n','')
    dictionary_degree[new_line.split(' ')[0]] = int(new_line.split(' ')[1])
f_n_d.close()

# choose 2298 users
content = linecache.getline(file_sorted_degree,1)
root_node = content.replace('\n' ,'').split(' ')[0]
save_nodes = []
son_nodes = []
for i in utils.related_three_node(adj_dictionary[root_node],save_nodes,dictionary_degree):
    son_nodes.append(i)
save_nodes.append(root_node)
# save node a,b,c
for node in son_nodes:
    save_nodes.append(node)
# depth-first traversal
for node in son_nodes:
    for i in utils.related_three_node(adj_dictionary[node],save_nodes,dictionary_degree):
        save_nodes.append(i)
        son_nodes.append(i)
    if len(save_nodes) >= 2298:
        break

f_s_p = open(file_subgraph_pos,'w')
f_s_n = open(file_subgraph_neg,'w')

# find edges which have the 2298 users
for node_l in save_nodes:
    for node_r in save_nodes:
        if node_l == node_r:
            pass
        elif utils.key_in_dic(node_l+' ' + node_r, edges_dictionary):
            f_s_p.write(node_l + ' ' + node_r + '\n')
        else:
            f_s_n.write(node_l + ' ' + node_r + '\n')
f_s_p.close()
f_s_n.close()
