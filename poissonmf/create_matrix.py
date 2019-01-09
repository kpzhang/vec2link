# coding:utf-8
import gc
import random
# create user poi-frequency matrix

import utils

poi_freq = '../data/poi_frequency.txt'
poi_dict = {}
p_f = open(poi_freq)
for (line_num,line_content) in enumerate(p_f):
    poi_dict[line_num] = line_content.split(' ')[0]
p_f.close()


lst = []
ll = []
user_traj = '../data/user_traj.txt'
u_t = open(user_traj)
user_index = 0
for line in u_t.readlines():
    dir = utils.count_poi(line)
    for poi_index in range(0,len(poi_dict)):
        if utils.key_in_dic(poi_dict[poi_index], dir):
            ll.append(dir[poi_dict[poi_index]])
        else:
            ll.append(0)
    lst.append(ll)
        # ll = []
    ll = []
        # gc.collect()
    user_index += 1
    print user_index
u_t.close()
#
import numpy as np
X = np.array(lst)

from pmf import PoissonMF
pmf = PoissonMF()
pmf.fit(X)
pmf.transform(X)

user_matrix = pmf.Et

user_vec = '../data/user_vec.txt'
u_v = open(user_vec,'w')
for e in user_matrix.tolist():
    u_v.write(utils.flatten(e))
u_v.close()

