# 模型
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
import json


class NlpModel:
    """
    改造的 lstm 模型
    """

    def __init__(self, sentence_fixed_len, learning_rate, word_vec_size, hidden_num, label_num):
        """

        :param sentence_fixed_len: 句子长度
        :param learning_rate: 学习速率
        :param word_vec_size: 词向量的大小
        :param hidden_num: lstm 隐藏层 size 大小
        :param label_num: 类别数量.
        """
        tf.reset_default_graph()
        self.sentence_fixed_len = sentence_fixed_len
        self.learning_rate = learning_rate
        self.word_vec_size = word_vec_size
        self.hidden_num = hidden_num
        self.batch_size = None
        self.sess = None
        self.merged_summary = None
        self.label_num = label_num

        # 输入一个batch的句子 shape = [batch_size,sentence_fixed_len,word_vec_size]
        self.__input_sentences = tf.placeholder(
            shape=[self.batch_size, self.sentence_fixed_len, self.word_vec_size],
            name='input_sentences', dtype=tf.float32)
        # 意图label shape = [batch_size]
        self.__input_labels = tf.placeholder(shape=[self.batch_size], name='input_labels', dtype=tf.int32)
        # 句子的真实长度 shape = [batch_size]
        self.__sentences_ready_length = tf.placeholder(shape=[self.batch_size], name='sentences_ready_length',
                                                       dtype=tf.int32)
        self.__input_keep_prob = tf.placeholder(shape=None, name='input_keep_prob', dtype=tf.float32)
        self.__state_keep_prob = tf.placeholder(shape=None, name='state_keep_prob', dtype=tf.float32)
        self.__output_keep_prob = tf.placeholder(shape=None, name='output_keep_prob', dtype=tf.float32)
        # 用于将 那些句子长度为0 的句子的loss和score设置为0
        self.__score_weight = tf.to_float(tf.not_equal(self.__sentences_ready_length, 0))
        self.__valid_sentence = tf.reduce_sum(self.__score_weight)

    def build(self, sess):
        """
        :param sess: tf.Session() 子模型的Session
        :return: void
        """
        self.sess = sess
        with tf.variable_scope("attention_lstm_layer"):
            encoder_outputs, encoder_final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.get_drop_lstm(num_units=self.hidden_num, input_keep_prob=self.__input_keep_prob,
                                           state_keep_prob=self.__state_keep_prob,
                                           output_keep_prob=self.__output_keep_prob),
                cell_bw=self.get_drop_lstm(num_units=self.hidden_num, input_keep_prob=self.__input_keep_prob,
                                           state_keep_prob=self.__state_keep_prob,
                                           output_keep_prob=self.__output_keep_prob),
                inputs=self.__input_sentences,
                sequence_length=self.__sentences_ready_length,
                dtype=tf.float32,
                time_major=False)
            encoder_outputs = tf.concat(encoder_outputs, 2)

            attention_v = tf.get_variable(shape=[self.hidden_num * 2],
                                          initializer=tf.initializers.random_normal(mean=0, stddev=1),
                                          name='attention_v')

            auto_attention_rate = tf.tanh(tf.tensordot(tf.tanh(encoder_outputs), attention_v, axes=1))
            self.attention_rate_length = tf.sequence_mask(self.__sentences_ready_length, self.sentence_fixed_len)
            N_TF_INF = tf.constant([-np.array(1e10)]) + auto_attention_rate
            auto_attention_rate = tf.where(self.attention_rate_length, auto_attention_rate, N_TF_INF)
            self.__auto_attention_rate = tf.nn.softmax(auto_attention_rate, name='alphas')

        with tf.variable_scope('word_vec_add_attention'):
            self.__manual_attention_rate = tf.placeholder(dtype=tf.float32,
                                                          shape=[self.batch_size, self.sentence_fixed_len])
            # # 如果人工是0，那么 机器学的注意力 返回是1 人工是0
            # auto_attention_rate_weight = tf.to_float(tf.equal(tf.reduce_sum(self.__manual_attention_rate, 1), 0))
            # auto_attention_rate_weight = tf.expand_dims(auto_attention_rate_weight, 1)
            # auto_attention_rate_weight = tf.tile(auto_attention_rate_weight, multiples=[1, self.sentence_fixed_len])

            # auto_attention_rate = tf.multiply(self.__auto_attention_rate, auto_attention_rate_weight)
            # manual_attention_rate = tf.multiply(self.__manual_attention_rate, 1 - auto_attention_rate_weight)
            # self.__attention_rate = auto_attention_rate + manual_attention_rate
            self.__attention_rate = self.__auto_attention_rate
            attention_rate = tf.expand_dims(self.__attention_rate, 2)
            add_atten_input = tf.map_fn(lambda x: x[0] * x[1], (encoder_outputs, attention_rate),
                                        dtype=tf.float32)

            intent_output = tf.tanh(tf.reduce_sum(add_atten_input, 1))

            intent_matrix = tf.get_variable(
                name='intent_matrix',
                dtype=tf.float32,
                shape=[self.hidden_num * 2, self.label_num],
                initializer=tf.initializers.random_normal(mean=0, stddev=1),
            )
            intent_bias = tf.get_variable(
                dtype=tf.float32,
                shape=[self.label_num],
                initializer=tf.initializers.random_normal(mean=0, stddev=1),
                name="intent_bias"
            )
            self.__predict_intent_vector = tf.matmul(intent_output, intent_matrix) + intent_bias
            self.__predict_intent = tf.argmax(self.__predict_intent_vector, 1, output_type=tf.int32, name='intent')

        with tf.variable_scope('loss_function'):
            self.__loss = self.blank_entropy_loss(
                prediction_tensor=self.__predict_intent_vector,
                target_tensor=self.__input_labels,
            )
            optimizer = tf.train.AdamOptimizer(name="a_optimizer", learning_rate=self.learning_rate)
            self.__grads, self.__vars = zip(*optimizer.compute_gradients(self.__loss))
            self.__gradients, _ = tf.clip_by_global_norm(self.__grads, 5)
            self.__train_op = optimizer.apply_gradients(zip(self.__gradients, self.__vars))

        with tf.variable_scope('evaluate_indicator'):
            self.__correct = tf.reduce_sum(
                tf.multiply(tf.to_float(tf.equal(self.__predict_intent, self.__input_labels)),
                            self.__score_weight))

            self.__mistake = tf.subtract(self.__valid_sentence, self.__correct)

    def train(self, input_sentences, input_labels, sentences_ready_length, input_keep_prob, state_keep_prob,
              output_keep_prob, manual_attention_rate, verbose=1):
        """

        :param input_sentences: 训练的句子 shape [batch_size,sentence_fixed_len,word_vec_size]
        :param input_labels: 训练的label的idx shape [batch_size]
        :param sentences_ready_length:  训练的句子的真实长度 shape [batch_size]
        :param input_keep_prob:
        :param state_keep_prob:
        :param output_keep_prob:
        :return:
        """
        # TODO F1 / Acc 计算
        # TODO 代码要修改的 因为verbose，
        # TODO 后期可以加个Score之类的列表 来选评价指标
        if verbose is 1:
            loss, correct, mistake, _ = self.sess.run(
                [self.__loss, self.__correct, self.__mistake, self.__train_op], feed_dict={
                    self.__input_sentences: input_sentences,
                    self.__input_labels: input_labels,
                    self.__sentences_ready_length: sentences_ready_length,
                    self.__input_keep_prob: input_keep_prob,
                    self.__state_keep_prob: state_keep_prob,
                    self.__output_keep_prob: output_keep_prob,
                    self.__manual_attention_rate: manual_attention_rate,
                })
            return loss, correct, mistake
        loss, correct, mistake, result, _ = self.sess.run(
            [self.__loss, self.__correct, self.__mistake, self.__predict_intent, self.__train_op], feed_dict={
                self.__input_sentences: input_sentences,
                self.__input_labels: input_labels,
                self.__sentences_ready_length: sentences_ready_length,
                self.__input_keep_prob: input_keep_prob,
                self.__state_keep_prob: state_keep_prob,
                self.__output_keep_prob: output_keep_prob,
                self.__manual_attention_rate: manual_attention_rate,
            })
        return loss, correct, mistake, result

    def predict(self, input_sentences, input_labels, sentences_ready_length, manual_attention_rate, verbose=1):
        """

        :param input_sentences: 预测的句子 shape = [batch_size, sentence_fixed_len, word_vec_size],
        :param sentences_ready_length: 预测句子的真实长度 shape = [batch_size]
        :return: 预测的结果 和 注意力的权重
        """

        if verbose is 1:
            correct, mistake = self.sess.run([self.__correct, self.__mistake], feed_dict={
                self.__input_sentences: input_sentences,
                self.__input_labels: input_labels,
                self.__sentences_ready_length: sentences_ready_length,
                self.__input_keep_prob: 1,
                self.__state_keep_prob: 1,
                self.__output_keep_prob: 1,
                self.__manual_attention_rate: manual_attention_rate,
            })
            return correct, mistake

    def get_attention(self, input_sentences, sentences_ready_length, manual_attention_rate):

        predict_intent, attention_rate = self.sess.run([self.__predict_intent, self.__attention_rate], feed_dict={
            self.__input_sentences: input_sentences,
            self.__sentences_ready_length: sentences_ready_length,
            self.__input_keep_prob: 1,
            self.__state_keep_prob: 1,
            self.__output_keep_prob: 1,
            self.__manual_attention_rate: manual_attention_rate,
        })

        return predict_intent, attention_rate

    @staticmethod
    def get_drop_lstm(num_units, input_keep_prob, state_keep_prob, output_keep_prob):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units, initializer=tf.orthogonal_initializer())
        return tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=input_keep_prob,
                                             state_keep_prob=state_keep_prob,
                                             output_keep_prob=output_keep_prob,
                                             dtype=tf.float32)

    def blank_entropy_loss(self, prediction_tensor, target_tensor):
        """
        由于数据中存在一些句子长度为0的句子, 而他们不应该参与到loss计算中来。所以要把他们的loss处理掉
        :param prediction_tensor:
        :param target_tensor:
        :return:
        """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_tensor, labels=target_tensor)
        denominator = tf.reduce_sum(self.__score_weight)
        return tf.reduce_sum(tf.multiply(loss, self.__score_weight)) / denominator

    @property
    def graph(self):
        return self.__score_weight.graph


if __name__ == '__main__':
    for _ in range(1):
        lstm_model = NlpModel(sentence_fixed_len=16, learning_rate=0.001, word_vec_size=1, hidden_num=20,
                              label_num=4)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        lstm_model.build(sess)
        sess.run(tf.global_variables_initializer())

        for _ in range(1000):
            tmp = lstm_model.train(
                input_sentences=[[[0.1] for _ in range(1)] + [[1] for _ in range(15)],
                                 [[1] for _ in range(15)] + [[0.2] for _ in range(1)],
                                 [[0.3] for _ in range(1)] + [[1] for _ in range(15)],
                                 [[0.4] for _ in range(1)] + [[1] for _ in range(15)],
                                 ],
                input_labels=[1, 2, 3, 1],
                manual_attention_rate=[
                    (np.array([0])).repeat([16]),
                    (np.array([0])).repeat([16]),
                    (np.array([0])).repeat([16]),
                    (np.array([0])).repeat([16])
                ],
                sentences_ready_length=[10, 16, 10, 0],
                input_keep_prob=1,
                state_keep_prob=1,
                output_keep_prob=1,
                verbose=0
            )
            print(tmp)
        result = lstm_model.predict(
            input_sentences=[[[0.1] for _ in range(1)] + [[1] for _ in range(15)],
                             [[1] for _ in range(15)] + [[0.2] for _ in range(1)],
                             [[0.3] for _ in range(1)] + [[1] for _ in range(15)],
                             [[0.4] for _ in range(1)] + [[1] for _ in range(15)],
                             ],
            input_labels=[1, 2, 3, 1],
            sentences_ready_length=[10, 16, 10, 0],
            manual_attention_rate=[
                (np.array([0])).repeat([16]),
                (np.array([0])).repeat([16]),
                (np.array([0])).repeat([16]),
                (np.array([0])).repeat([16])
            ],
        )
        print(result[0])
        print(result[1])

        result = lstm_model.get_attention(
            input_sentences=[[[0.1] for _ in range(1)] + [[1] for _ in range(15)],
                             [[1] for _ in range(15)] + [[0.2] for _ in range(1)],
                             [[0.3] for _ in range(1)] + [[1] for _ in range(15)],
                             [[0.4] for _ in range(1)] + [[1] for _ in range(15)],
                             ],
            sentences_ready_length=[10, 16, 10, 0],
            manual_attention_rate=[
                (np.array([0.5])).repeat([16]),
                (np.array([0])).repeat([16]),
                (np.array([0])).repeat([16]),
                (np.array([0])).repeat([16])
            ],
        )

        print(result[0])
        print(result[1])
