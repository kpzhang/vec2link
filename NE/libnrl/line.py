import random
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from classify import Classifier, read_node_label


class _LINE(object):

    def __init__(self, graph, rep_size=128, batch_size=100, negative_ratio=5, order=3):
        self.cur_epoch = 0
        self.order = order
        self.g = graph
        self.node_size = graph.G.number_of_nodes()
        self.rep_size = rep_size
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio

        self.gen_sampling_table()
        self.sess = tf.Session()
        cur_seed = random.getrandbits(32)
        initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            self.build_graph()
        self.sess.run(tf.global_variables_initializer())

    def build_graph(self):
        self.pos_h = tf.placeholder(tf.int32, [None])
        self.pos_t = tf.placeholder(tf.int32, [None])
        self.pos_h_v = tf.placeholder(tf.int32, [None, self.negative_ratio])
        self.neg_t = tf.placeholder(tf.int32, [None, self.negative_ratio])

        cur_seed = random.getrandbits(32)
        self.embeddings = tf.get_variable(name="embeddings"+str(self.order), shape=[self.node_size, self.rep_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False, seed=cur_seed))
        self.context_embeddings = tf.get_variable(name="context_embeddings"+str(self.order), shape=[self.node_size, self.rep_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False, seed=cur_seed))
        self.pos_h_e = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.embeddings, self.pos_h), 1)
        self.pos_t_e = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.embeddings, self.pos_t), 1)
        self.pos_t_e_context = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.context_embeddings, self.pos_t), 1)
        self.pos_h_v_e = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.embeddings, self.pos_h_v), 2)
        self.neg_t_e = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.embeddings, self.neg_t), 2)
        self.neg_t_e_context = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.context_embeddings, self.neg_t), 2)
        # self.sample_sum2 = tf.reduce_sum(tf.log(tf.nn.sigmoid(tf.reduce_sum(tf.multiply(self.pos_h_v_e, self.neg_t_e_context), axis=2))), axis=1)
        # self.second_loss = tf.reduce_mean(-tf.log(tf.nn.sigmoid(tf.reduce_sum(tf.multiply(self.pos_h_e, self.pos_t_e_context), axis=1))) +
        #                            self.sample_sum2)
        # self.sample_sum1 = tf.reduce_sum(tf.log(tf.nn.sigmoid(tf.reduce_sum(tf.multiply(self.pos_h_v_e, self.neg_t_e), axis=2))), axis=1)
        # self.first_loss = tf.reduce_mean(-tf.log(tf.nn.sigmoid(tf.reduce_sum(tf.multiply(self.pos_h_e, self.pos_t_e), axis=1))) +
        #                            self.sample_sum1)
        self.sample_sum2 = tf.reduce_sum(tf.exp(tf.reduce_sum(tf.multiply(self.pos_h_v_e, self.neg_t_e_context), axis=2)), axis=1)
        self.second_loss = tf.reduce_mean(-tf.reduce_sum(tf.multiply(self.pos_h_e, self.pos_t_e_context), axis=1) +
                                   tf.log(self.sample_sum2))
        self.sample_sum1 = tf.reduce_sum(tf.exp(tf.reduce_sum(tf.multiply(self.pos_h_v_e, self.neg_t_e), axis=2)), axis=1)
        self.first_loss = tf.reduce_mean(-tf.reduce_sum(tf.multiply(self.pos_h_e, self.pos_t_e), axis=1) +
                                   tf.log(self.sample_sum1))
        if self.order == 1:
            self.loss = self.first_loss
        else:
            self.loss = self.second_loss
        optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = optimizer.minimize(self.loss)


    def train_one_epoch(self):
        sum_loss = 0.0
        batches = self.batch_iter()
        batch_id = 0
        for batch in batches:
            pos_h, pos_h_v, pos_t, neg_t = batch
            feed_dict = {
                self.pos_h : pos_h,
                self.pos_h_v : pos_h_v,
                self.pos_t : pos_t,
                self.neg_t : neg_t,
            }
            _, cur_loss = self.sess.run([self.train_op, self.loss],feed_dict)
            sum_loss += cur_loss
            batch_id += 1
        print 'epoch:{} sum of loss:{!s}'.format(self.cur_epoch, sum_loss)
        self.cur_epoch += 1

    def batch_iter(self):
        look_up = self.g.look_up_dict

        table_size = 1e8
        numNodes = self.node_size

        edges = [(look_up[x[0]], look_up[x[1]]) for x in self.g.G.edges()]

        data_size = self.g.G.number_of_edges()
        edge_set = set([x[0]*numNodes+x[1] for x in edges])
        shuffle_indices = np.random.permutation(np.arange(data_size))
        start_index = 0
        end_index = min(start_index+self.batch_size, data_size)
        while start_index < data_size:
            pos_h = []
            pos_h_v = []
            pos_t = []
            neg_t = []

            for i in range(start_index, end_index):
                if not random.random() < self.edge_prob[shuffle_indices[i]]:
                    shuffle_indices[i] = self.edge_alias[shuffle_indices[i]]
                cur_h = edges[shuffle_indices[i]][0]
                head = cur_h*numNodes
                cur_t = edges[shuffle_indices[i]][1]
                cur_h_v = []
                cur_neg_t = []
                for j in range(self.negative_ratio):
                    rn = self.sampling_table[random.randint(0, table_size-1)]
                    while head+rn in edge_set or cur_h == rn or rn in cur_neg_t:
                        rn = self.sampling_table[random.randint(0, table_size-1)]
                    cur_h_v.append(cur_h)
                    cur_neg_t.append(rn)
                pos_h.append(cur_h)
                pos_h_v.append(cur_h_v)
                pos_t.append(cur_t)
                neg_t.append(cur_neg_t)

            yield pos_h, pos_h_v, pos_t, neg_t
            start_index = end_index
            end_index = min(start_index+self.batch_size, data_size)

    def gen_sampling_table(self):
        table_size = 1e8
        power = 0.75
        numNodes = self.node_size

        print "Pre-procesing for non-uniform negative sampling!"
        node_degree = np.zeros(numNodes) # out degree

        look_up = self.g.look_up_dict
        for edge in self.g.G.edges():
            node_degree[look_up[edge[0]]] += self.g.G[edge[0]][edge[1]]["weight"]

        norm = sum([math.pow(node_degree[i], power) for i in range(numNodes)])

        self.sampling_table = np.zeros(int(table_size), dtype=np.uint32)

        p = 0
        i = 0
        for j in range(numNodes):
            p += float(math.pow(node_degree[j], power)) / norm
            while i < table_size and float(i) / table_size < p:
                self.sampling_table[i] = j
                i += 1

        data_size = self.g.G.number_of_edges()
        self.edge_alias = np.zeros(data_size, dtype=np.int32)
        self.edge_prob = np.zeros(data_size, dtype=np.float32)
        large_block = np.zeros(data_size, dtype=np.int32)
        small_block = np.zeros(data_size, dtype=np.int32)

        total_sum = sum([self.g.G[edge[0]][edge[1]]["weight"] for edge in self.g.G.edges()])
        norm_prob = [self.g.G[edge[0]][edge[1]]["weight"]*data_size/total_sum for edge in self.g.G.edges()]
        num_small_block = 0
        num_large_block = 0
        cur_small_block = 0
        cur_large_block = 0
        for k in range(data_size-1, -1, -1):
            if norm_prob[k] < 1:
                small_block[num_small_block] = k
                num_small_block += 1
            else:
                large_block[num_large_block] = k
                num_large_block += 1
        while num_small_block and num_large_block:
            num_small_block -= 1
            cur_small_block = small_block[num_small_block]
            num_large_block -= 1
            cur_large_block = large_block[num_large_block]
            self.edge_prob[cur_small_block] = norm_prob[cur_small_block]
            self.edge_alias[cur_small_block] = cur_large_block
            norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] -1
            if norm_prob[cur_large_block] < 1:
                small_block[num_small_block] = cur_large_block
                num_small_block += 1
            else:
                large_block[num_large_block] = cur_large_block
                num_large_block += 1

        while num_large_block:
            num_large_block -= 1
            self.edge_prob[large_block[num_large_block]] = 1
        while num_small_block:
            num_small_block -= 1
            self.edge_prob[small_block[num_small_block]] = 1


    def get_embeddings(self):
        vectors = {}
        embeddings = self.sess.run(tf.nn.l2_normalize(self.embeddings.eval(session=self.sess), 1))
        look_back = self.g.look_back_list
        for i, embedding in enumerate(embeddings):
            vectors[look_back[i]] = embedding
        return vectors

class LINE(object):

    def __init__(self, graph, rep_size=128, batch_size=1000, epoch=10, negative_ratio=5, order=3, label_file = None, clf_ratio = 0.5, auto_stop = True):
        self.rep_size = rep_size
        self.order = order
        self.best_result = 0
        self.vectors = {}
        if order == 3:
            self.model1 = _LINE(graph, rep_size/2, batch_size, negative_ratio, order=1)
            self.model2 = _LINE(graph, rep_size/2, batch_size, negative_ratio, order=2)
            for i in range(epoch):
                self.model1.train_one_epoch()
                self.model2.train_one_epoch()
                if label_file:
                    self.get_embeddings()
                    X, Y = read_node_label(label_file)
                    print "Training classifier using {:.2f}% nodes...".format(clf_ratio*100)
                    clf = Classifier(vectors=self.vectors, clf=LogisticRegression())
                    result = clf.split_train_evaluate(X, Y, clf_ratio)

                    if result['micro'] < self.best_result and auto_stop:
                        self.vectors = self.last_vectors
                        print 'Auto stop!'
                        return
                    elif result['micro'] > self.best_result:
                        self.best_result = result['micro']

        else:
            self.model = _LINE(graph, rep_size, batch_size, negative_ratio, order=self.order)
            for i in range(epoch):
                self.model.train_one_epoch()
                if label_file:
                    self.get_embeddings()
                    X, Y = read_node_label(label_file)
                    print "Training classifier using {:.2f}% nodes...".format(clf_ratio*100)
                    clf = Classifier(vectors=self.vectors, clf=LogisticRegression())
                    result = clf.split_train_evaluate(X, Y, clf_ratio)

                    if result['micro'] < self.best_result and auto_stop:
                        self.vectors = self.last_vectors
                        print 'Auto stop!'
                        return
                    elif result['micro'] > self.best_result:
                        self.best_result = result['micro']

        self.get_embeddings()

    def get_embeddings(self):
        self.last_vectors = self.vectors
        self.vectors = {}
        if self.order == 3:
            vectors1 = self.model1.get_embeddings()
            vectors2 = self.model2.get_embeddings()
            for node in vectors1.keys():
                self.vectors[node] = np.append(vectors1[node], vectors2[node])
        else:
            self.vectors = self.model.get_embeddings()

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()