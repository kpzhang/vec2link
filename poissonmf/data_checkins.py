# coding:utf-8
import utils
# adj_dictionary save: user [poi_1,poi_2...poi_n]
file_bk_traj = '../data/brightkite_filter2.txt'

adj_dictionary = {}
# edges_dictionary = {}
f_n = open(file_bk_traj)
i = 0
for line in f_n.readlines():
    user = line.replace('\n','').split(' ')[0]
    location = line.replace('\n','').split(' ')[4]
    i += 1
    print i
    # edges_dictionary[user+' '+location] = ''
    if utils.key_in_dic(user,adj_dictionary):
        adj_dictionary[str(user)].append(location)
    else:
        adj_dictionary[str(user)] = list()
        adj_dictionary[str(user)].append(location)

f_n.close()

# obtain user-location_id
user_file = '../data/sorted_nodes1.txt'
doc_file = '../data/user_traj.txt'
f_b_u = open(doc_file,'w')
f_u = open(user_file)
for line in f_u.readlines():
    if utils.key_in_dic(line.split('\n')[0],adj_dictionary):
        j = 0
        for sub_trah in adj_dictionary[line.split('\n')[0]]:
            j += 1
            if j == len(adj_dictionary[line.split('\n')[0]]):
                f_b_u.write(sub_trah + '\n')
            else:
                f_b_u.write(sub_trah + ' ')
    # user in edges without check-ins
    else:
        print 'lost user', line
        f_b_u.write('\n')
f_b_u.close()
f_u.close()

# get user
poi_file = '../data/user_poi.txt'
p_f = open(poi_file,'w')
f_b_t = open(file_bk_traj)
u_f = open(user_file)
user_dict = {}
for line in u_f.readlines():
    user_dict[line.split()[0]] = ''
u_f.close()

for line in f_b_t.readlines():
    if utils.key_in_dic(line.replace('\n','').split('\t')[0],user_dict):
        p_f.write(line)
    else:
        pass
f_b_t.close()
p_f.close()



#
from gensim.models.doc2vec import TaggedLineDocument, Doc2Vec
user_tranj_vec = '../data/user_tranj_vec.txt'

documents = TaggedLineDocument(doc_file)
model = Doc2Vec(documents, size=128, negative=10, window=8, hs=0, min_count=0, workers=15, iter=30)

user_id_list = []
u_f = open(user_file)
for line in u_f:
    user_id_list.append(line.split('\n')[0])
u_f.close()

#
assert len(user_id_list) == len(model.docvecs)
# model.save(save_path)
f_s = open(user_tranj_vec, 'w')
for i, docvec in enumerate(model.docvecs):
    j = 0
    for v in docvec:
        j += 1
        if j == len(docvec):
            f_s.write(str(v) + '\n')
        else:
            f_s.write(str(v) + ' ')
f_s.close()


# get poi
from gensim.models import word2vec

sentence = word2vec.LineSentence('../data/user_traj.txt')
model = word2vec.Word2Vec(sentence, size=5, min_count=0, workers=15)
model.wv.save_word2vec_format('../data/tmp_traj_vec.txt')

# count poi - frequency
user_traj = '../data/user_traj.txt'
poi_all = '../data/tmp_traj_vec.txt'
poi_frequency = '../data/poi_frequency.txt'

poi_dictionary = {}

p_a = open(poi_all)
next(p_a)
for line in p_a:
    poi_dictionary[line.split()[0]] = 0
p_a.close()

u_t = open(user_traj)
for line in u_t.readlines():
    for element in line.split():
        if utils.key_in_dic(element,poi_dictionary):
            poi_dictionary[element] += 1
        else:
            print element
u_t.close()

#
p_f = open(poi_frequency,'w')
for key,value in poi_dictionary.iteritems():
    p_f.write(key+' '+str(value)+'\n')
p_f.close()