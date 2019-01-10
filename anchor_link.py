# coding:utf-8
#
import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import roc_auc_score

time_record_2 = []
loss_record_2 = []

class Mlp:
    def __init__(self, sess, inputs_data, outputs_data, index_lst1, index_lst2, output_dim, lr=0.0001,
                 batch_size=128, run_time=100000, learning_rate_decay_factor=0.98, output_interval=200):
        self._sess = sess
        self._inputs_data = inputs_data
        self._output_dim = output_dim
        self._outputs_data = outputs_data
        self._x = tf.placeholder(tf.float32, [None, len(inputs_data[0])])
        self._y = tf.placeholder(tf.float32, [None, self._output_dim])
        self._inputs_num = len(self._inputs_data)
        self._index_lst1 = index_lst1
        self._index_lst2 = index_lst2
        # self._lr = lr
        self.learning_rate = tf.Variable(
            float(lr), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)

        self.learning_rate_2 = tf.Variable(
            float(lr), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op_2 = self.learning_rate_2.assign(
            self.learning_rate_2 * learning_rate_decay_factor)
        self._current_index = 0
        self._batch_size = batch_size
        self._run_time = run_time
        self._output_interval = output_interval

        self._build_net()

    # # normalization
    def _normalize_vector(self,vector):
        norm = tf.sqrt(tf.reduce_sum(tf.square(vector), keep_dims=True))
        normalized_embeddings = vector / norm
        return normalized_embeddings

    def _build_net(self):
        l0 = tf.layers.dense(self._x, 128, tf.nn.relu)
        l1 = tf.layers.dense(l0, 64, tf.nn.relu)
        l2 = tf.layers.dense(l1, 32, tf.nn.relu)
        self._output = tf.layers.dense(l2, self._output_dim)  # output layer

        self.loss = tf.reduce_mean(tf.square(self._output - self._y))
        # self.loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self._y, logits=self._output)  # compute cost
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        cosine_min = 0.0

        for cycle in range(0, len(self._inputs_data)):
            result = tf.multiply(self._normalize_vector(self._output[self._index_lst1[cycle]]),
                                 self._normalize_vector(self._output[self._index_lst2[cycle]]))
            result = tf.reduce_sum(result)
            cosine_min = tf.add(cosine_min,result)

        cosine_min = tf.div(cosine_min,len(self._inputs_data))

        self.loss_2 = tf.reduce_mean(0.176758350078-cosine_min)


        self.train_op_2 = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_2)



    def _get_batch(self):
        result_x = []
        result_y = []
        if self._current_index + self._batch_size <= self._inputs_num:
            result_x = self._inputs_data[self._current_index:self._current_index + self._batch_size]
            result_y = self._outputs_data[self._current_index:self._current_index + self._batch_size]
            self._current_index += self._batch_size
        else:
            result_x = list(self._inputs_data[self._current_index:])
            result_y = list(self._outputs_data[self._current_index:])
            length = self._batch_size - len(result_x)
            for i in range(length):
                result_x.append(self._inputs_data[i])
                result_y.append(self._outputs_data[i])
            self._current_index = length
        assert len(result_x) == len(result_y) == self._batch_size
        # return result_x, result_y
        return result_x, result_y


    def _get_train(self):
        return self._inputs_data, [[i] for i in self._outputs_data]

    def tran_net(self):
        total_loss = 0
        loss_history = [0., 0., 0.]
        loss_history_2 = [0., 0., 0.]
        count = 0
        second_count = 0
        begin = time.time()
        time_record = []
        loss_record = []
        log_cosine = '/media/ubuntu-01/ML/sub_bk_link_prediction_1/log/cosine_loss.txt'

        l_s = open(log_cosine,'w')
        for i in range(self._run_time):
            x_batch, y_batch = self._get_batch()
            count += 1
            if (len(self._inputs_data) / self._batch_size) + 1 >= count:
                lr_, _loss, _, _output = self._sess.run([self.learning_rate, self.loss, self.train_op, self._output],
                                                    {self._x: x_batch, self._y: y_batch})
                total_loss += _loss
                if i != 0 and i % ((len(self._inputs_data) / self._batch_size) + 1) == 0:
                    current_loss = total_loss / ((len(self._inputs_data) / self._batch_size) + 1)

                    total_loss = 0
                    if i > 1000 and current_loss > max(loss_history):
                        self._sess.run(self.learning_rate_decay_op)
                    loss_history[i % 3] = current_loss


                    print 'step:', i, 'lr:', lr_, 'train_loss:', current_loss
                    time_record.append(time.time() - begin)
                    loss_record.append(current_loss)
            else:
                count = 0
                lr_2, _loss_2, _, _output_2 = self._sess.run([self.learning_rate_2, self.loss_2, self.train_op_2, self._output],
                                                    {self._x: self._inputs_data, self._y: self._outputs_data})
                if second_count > 3 and _loss_2 > max(loss_history_2):
                    self._sess.run(self.learning_rate_decay_op_2)
                loss_history_2[second_count % 3] = _loss_2
                second_count += 1

                print 'step:', i, 'lr:', lr_2, 'second_loss:', _loss_2
                time_record.append(time.time() - begin)
                loss_record.append(_loss_2)

                cosine_loss = 'step:'+str(i)+' lr:'+str(lr_2)+' cosine_loss:'+str(_loss_2)+'\n'
                l_s.write(cosine_loss)

                time_record_2.append(i)
                loss_record_2.append(_loss_2)


    def predict(self, inputs):
        return self._sess.run(self._output, {self._x: inputs})


train_inputs = []
train_labels = []
file_test = 'network_vec.txt'
file_train = '/media/ubuntu-01/ML/poission_mf/user_vec1.txt'
f_tr = open(file_train)
f_te = open(file_test)
for line1,line2 in zip(f_tr,f_te):
    train_inputs.append(map(float, list(line1.split(' ')[:])))
    train_labels.append(map(float, list(line2.split(' ')[:])))
f_tr.close()
f_te.close()

index_file = './subgraph_positive_index.txt'
lst_1 = []
lst_2 = []
i_f = open(index_file)
for line in i_f.readlines():
    lst_1.append(int(line.split()[0]))
    lst_2.append(int(line.split()[1]))
i_f.close()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf_model = Mlp(sess, train_inputs, train_labels, lst_1, lst_2, 128,
               lr=0.0000003, run_time=100000, batch_size=64, learning_rate_decay_factor=0.9)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

try:
    for it in range(100000):
        tf_model.tran_net()

except KeyboardInterrupt:
    print
save_path = saver.save(sess,'./Model/save_model.ckpt')

# def get_newline(lst):
#     i = 0
#     new_line = ''
#     for v in lst:
#         if i < len(lst)-1:
#             new_line += str(v)+' '
#             i += 1
#         else:
#             new_line += str(v)+'\n'
#     return new_line
#
# saver.restore(sess, './Model/save_model.ckpt')
# result = tf_model.predict(train_inputs)
# f = open('./new_vec.txt', 'a')
# for line in result:
#     new_lst = line.tolist()
#     new_line = get_newline(new_lst)
#     f.write(new_line)
# f.close()
