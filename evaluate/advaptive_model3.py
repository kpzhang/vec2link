# coding: utf-8
# adaptive threshold method

import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import roc_auc_score
from sklearn import metrics


class Mlp:
    def __init__(self, sess, inputs_data, outputs_data, test_inputs, test_labels, output_dim, lr=0.0001,
                 batch_size=128, run_time=100000, learning_rate_decay_factor=0.98, output_interval=200):
        self._sess = sess
        self._inputs_data = inputs_data
        self._output_dim = output_dim
        self._outputs_data = outputs_data
        self._test_inputs_data = test_inputs
        self._test_labels_data = test_labels
        self._x = tf.placeholder(tf.float32, [None, len(inputs_data[0])])
        self._y = tf.placeholder(tf.int32, [None, self._output_dim])
        # self._w = tf.Variable(tf.float32)
        # self._b = tf.Variable(tf.float32)
        self._inputs_num = len(self._inputs_data)
        # self._lr = lr
        self.learning_rate = tf.Variable(
            float(lr), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self._current_index = 0
        self._batch_size = batch_size
        self._run_time = run_time
        self._output_interval = output_interval
        self._keep_prob = 0.5

        self._build_net()

    def _build_net(self):
        with tf.variable_scope('CNN'):
            # l0 = tf.layers.dense(self._x, 1024, tf.nn.sigmoid)
            l0 = tf.layers.dense(self._x, 10000, tf.nn.sigmoid)
            l1 = tf.layers.dense(l0, 1024, tf.nn.sigmoid)
            image = tf.reshape(l1,[-1,32,32,1])
            # CNN
            conv1 = tf.layers.conv2d(  # shape (16, 16, 1)
                inputs=image,
                filters=16,
                kernel_size=5,
                strides=1,
                padding='same',
                activation=tf.nn.relu
            )  # -> (16, 16, 16)
            pool1 = tf.layers.max_pooling2d(
                conv1,
                pool_size=2,
                strides=2,
            )  # -> (8, 8, 16)
            conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)  # -> (8, 8, 32)
            pool2 = tf.layers.max_pooling2d(conv2, 2, 2)  # -> (4, 4, 32)
            flat = tf.reshape(pool2, [-1, 8 * 8 * 32])  # -> (4*4*32, )
            flat1 = tf.nn.dropout(flat, keep_prob=self._keep_prob)
            self._output = tf.layers.dense(flat1, self._output_dim)  # output layer
            self.cnn_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CNN')
            # self.loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self._y, logits=self._output) + 0.01 * (tf.nn.l2_loss(self.cnn_params[0])+tf.nn.l2_loss(self.cnn_params[1])+tf.nn.l2_loss(self.cnn_params[2])+tf.nn.l2_loss(self.cnn_params[3])+tf.nn.l2_loss(self.cnn_params[4])+tf.nn.l2_loss(self.cnn_params[5])+tf.nn.l2_loss(self.cnn_params[6])+tf.nn.l2_loss(self.cnn_params[7]))  # compute cost
            self.loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self._y, logits=self._output)
            # self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self._y, logits=self._output)  # compute cost
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


            # l1 = tf.layers.dense(l0, 128, tf.nn.sigmoid)
            # l2 = tf.layers.dense(l1, 64, tf.nn.sigmoid)
            # self._output = tf.nn.sigmoid(tf.layers.dense(l2, self._output_dim))  # output layer
            #
            # self.loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self._y, logits=self._output)  # compute cost
            # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

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
        return result_x, [[i] for i in result_y]

    def _get_test(self):
        return self._test_inputs_data, [[i] for i in self._test_labels_data]

    def _get_train(self):
        return self._inputs_data, [[i] for i in self._outputs_data]

    def tran_net(self):
        total_loss = 0
        loss_history = [0., 0., 0.]
        for i in range(self._run_time):
            x_batch, y_batch = self._get_batch()
            try:
                lr_, _loss, _, _output = self._sess.run([self.learning_rate, self.loss, self.train_op, self._output],
                                                        {self._x: x_batch, self._y: y_batch})
            except Exception, e:
                print
            total_loss += _loss
            if i != 0 and i % self._output_interval == 0:
                current_loss = total_loss / self._output_interval

                total_loss = 0
                if i > 1000 and current_loss > max(loss_history):
                    self._sess.run(self.learning_rate_decay_op)
                loss_history[i % 3] = current_loss

                # auc on training
                x_train, y_train = self._get_train()
                _train_output = self._sess.run([self._output], {self._x: x_train, self._y: y_train})
                train_score = []
                for tr_out in _train_output[0].tolist():
                    for tr_o in tr_out:
                        train_score.append(tr_o)

                train_true = []
                for y_tr in y_train:
                    for y in y_tr:
                        train_true.append(y)

                train_auc = roc_auc_score(train_true, train_score)

                # loss on testing and the score
                x_test, y_test = self._get_test()
                _test_loss, _test_output = self._sess.run([self.loss, self._output],
                                                          {self._x: x_test, self._y: y_test})
                test_score = []
                for te_out in _test_output.tolist():
                    for te_o in te_out:
                        test_score.append(te_o)

                # number of pos samples in the test set
                test_pos = 0

                test_true = []
                for y_te in y_test:
                    for y_t in y_te:
                        test_true.append(y_t)
                        if y_t == 1:
                            test_pos += 1
                        else:
                            pass

                # number of neg samples in the test set
                test_neg = len(y_test) - test_pos

            # avg score of pos samples，avg score of neg sampels，the the average of these two
                test_pos_score_mean = 0.0
                test_neg_score_mean = 0.0
                for k in range(test_pos):
                    test_pos_score_mean += test_score[k]
                for j in range(test_neg):
                    test_neg_score_mean += test_score[test_pos+j]
                test_pos_score_mean /= test_pos
                test_neg_score_mean /= test_neg

                boundary = (test_pos_score_mean + test_neg_score_mean) / 2.0
                tp = 0
                fp = 0
                fn = 0
                tn = 0


                # if pos avg value > threshold
                if test_pos_score_mean - test_neg_score_mean > 1e-10:
                    test_score1 = []
                    index = 0
                    for e in test_score:
                        if e - boundary > 1e-10:
                            test_score1.append(1)
                            if test_true[index] == 1:
                                tp += 1 # pred: pos, true: pos
                            else:
                                fp += 1 # pred: pos, true: neg
                        else:
                            if test_true[index] == 1:
                                fn += 1 # pred: neg, true: pos
                            else:
                                tn += 1 # pred: neg, true: neg
                            test_score1.append(0)
                        index += 1
                else:
                    test_score1 = []
                    index = 0
                    for e in test_score:
                        if e - boundary < 1e-10:
                            test_score1.append(1)
                            if test_true[index] == 1:
                                tp += 1 #pred: pos, true: pos
                            else:
                                fp += 1 #pred: pos, true: neg
                        else:
                            if test_true[index] == 1:
                                fn += 1 #red: neg, true: pos
                            else:
                                tn += 1 #pred: neg, true: neg
                            test_score1.append(0)
                        index += 1

                try:
                    precision = tp * 1.0 / (tp + fp)
                    recall = tp * 1.0 / (tp + fn)
                    F1 = 2 * precision * recall * 1.0 / (precision + recall)
                except Exception as err:
                    print(err)
                finally:

                    test_auc = roc_auc_score(test_true, test_score)

                    print 'step:', i, 'lr:', lr_, 'train_loss:', current_loss, 'train_auc:', train_auc,\
                        'test_auc:', test_auc, 'precision:', precision,'recall:',recall,'F1:',F1


    def predict(self, inputs):
        return self._sess.run(self._output, {self._x: inputs})


train_inputs = []
train_labels = []
test_inputs = []
test_labels = []
file_train = 'fri_traj_train.txt'
file_test = 'fri_traj_test.txt'

f_tr = open(file_train)
for line in f_tr.readlines():
    train_inputs.append(list(line.split(' ')[:-1]))
    train_labels.append(int(line[-2]))
f_tr.close()
f_te = open(file_test)
for line in f_te.readlines():
    test_inputs.append(list(line.split(' ')[:-1]))
    test_labels.append(int(line[-2]))
f_te.close()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# with tf.device("/cpu:0"):
tf_model = Mlp(sess, train_inputs, train_labels, test_inputs, test_labels, 1,
               lr=0.00003, run_time=100000, batch_size=64, learning_rate_decay_factor=0.9)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

try:
    begin_time = time.time()
    for it in range(1):
        tf_model.tran_net()
    print 'train using time:', time.time() - begin_time
except KeyboardInterrupt:
    print
save_path = saver.save(sess,'./Model/save_model.ckpt')
