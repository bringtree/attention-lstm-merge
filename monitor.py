# 旨在 将 原数据 处理成 数字 后 给模型训练。
#
# TODO 站在 monitor 的角度的话。一份数据对应一个 nlp模型 训练.
# TODO but monitor need to return the information about attention. In order to
# TODO make the data can looks straight. we need to return the raw data(in Chinese,
# TODO not in number). So the Monitor need to be feed the raw data.
# TODO 可视化训练过程
# TODO 后期还要有人为的 attention 检测

from model import *
from data_model import *
import joblib
from data_control import *
import os
import json
import shutil


class Monitor(NlpModel, DataControl):
    def __init__(self, data, word2idx_dict, label2idx_dict, idx2label_dict, idx2vec_dict,
                 sentence_fixed_len, learning_rate, word_vec_size, hidden_num,label_num,
                 k_model_src):
        if data:
            DataControl.__init__(self, data=data, word2idx_dict=word2idx_dict, label2idx_dict=label2idx_dict,
                                 idx2vec_dict=idx2vec_dict, sentence_fixed_len=sentence_fixed_len,word_vec_size=word_vec_size)

        NlpModel.__init__(self, sentence_fixed_len, learning_rate, word_vec_size, hidden_num,label_num)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)), graph=self.graph)
        self.build(self.sess)
        # 模型初始化
        self.sess.run(tf.global_variables_initializer())
        self.idx2label_dict = idx2label_dict
        self.k_model_src = k_model_src
        self.saver = tf.train.Saver(max_to_keep=int(1))
        self.count_down = 10
        self.best_score = 0

    def train(self, batch_size, iter_num, count_down, input_keep_prob, state_keep_prob, output_keep_prob,
              manual_attention_rate=None):
        # TODO 留了个隐患 那些长度为0 的句子。在result中仍然可以看到。
        self.batch_size = batch_size
        self.count_down = count_down
        self.input_keep_prob = input_keep_prob
        self.state_keep_prob = state_keep_prob
        self.output_keep_prob = output_keep_prob

        dev_x_batches, dev_y_batches, dev_x_length_batches, dev_batches_num = self.generator_batches(batch_size,
                                                                                                     train_mode=False,
                                                                                                     shuffle_mode=False)
        for iter in range(iter_num):
            train_loss, train_correct, train_mistake = 0, 0, 0
            train_x_batches, train_y_batches, train_x_length_batches, train_batches_num = self.generator_batches(
                batch_size, train_mode=True,
                shuffle_mode=True)
            for batch_idx in range(train_batches_num):
                tmp_loss, tmp_correct, tmp_mistake = super().train(
                    input_sentences=train_x_batches[batch_idx],
                    input_labels=train_y_batches[batch_idx],
                    sentences_ready_length=train_x_length_batches[batch_idx],
                    input_keep_prob=input_keep_prob,
                    state_keep_prob=state_keep_prob,
                    output_keep_prob=output_keep_prob,
                    manual_attention_rate=manual_attention_rate or np.zeros([batch_size, self.sentence_fixed_len],
                                                                            dtype=np.float)

                )
                train_loss += tmp_loss
                train_correct += tmp_correct
                train_mistake += tmp_mistake
            train_acc = train_correct / (train_mistake + train_correct)
            if np.isnan(train_loss):
                print('nan_stop')
                return
            print(str(iter) + ' Train: ' + str(train_loss), train_acc)

            dev_correct, dev_mistake = 0, 0
            for batch_idx in range(dev_batches_num):
                tmp_correct, tmp_mistake = super().predict(
                    input_sentences=dev_x_batches[batch_idx],
                    input_labels=dev_y_batches[batch_idx],
                    sentences_ready_length=dev_x_length_batches[batch_idx],
                    manual_attention_rate=manual_attention_rate or np.zeros([batch_size, self.sentence_fixed_len],
                                                                            dtype=np.float)
                )
                dev_correct += tmp_correct
                dev_mistake += tmp_mistake
            dev_acc = dev_correct / (dev_correct + dev_mistake)
            print('Develop: ' + str(dev_acc))

            # early stopping

            if dev_acc > 0.7:
                if dev_acc > self.best_score:
                    # remove the pre model
                    if os.path.exists(self.get_model_src(self.best_score)):
                        shutil.rmtree(self.get_model_src(self.best_score))

                    if not os.path.exists(self.get_model_src(dev_acc)):
                        os.makedirs(self.get_model_src(dev_acc))

                    self.saver.save(self.sess, self.get_model_src(dev_acc), global_step=iter)
                    self.best_score = dev_acc
                    self.count_down = count_down
                else:
                    self.count_down -= 1
                    if self.count_down == 0:
                        return

    def get_model_src(self, acc):
        return self.k_model_src + "acc_" + str(round(acc, 5)) \
               + "_lstm_hid_" + str(int(self.hidden_num)) \
               + "_dropout_" + str(self.input_keep_prob) \
               + "_" + str(str(self.state_keep_prob)) \
               + "_" + str(str(self.output_keep_prob)) \
               + '/'

    def load_model(self, model_src):
        ckpt = tf.train.get_checkpoint_state(model_src)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("load fail")
    def get_attention_information(self, output_file, train_data_mode=False, manual_attention_rate=None):
        """

        :param train_data_mode:
        :param output_file:
        :return:
        """

        x_batches, y_batches, x_length_batches, batches_num = self.generator_batches(50, train_mode=train_data_mode,
                                                                                     shuffle_mode=False)
        x = self.x_train if train_data_mode else self.x_dev
        y = self.y_dev if train_data_mode else self.y_dev
        x_length = self.x_length_train if train_data_mode else self.x_length_dev

        predict_intent = []
        attention_rate = []

        for batch_idx in range(batches_num):
            tmp_predict_intent, tmp_attention_rate = self.get_attention(
                input_sentences=x_batches[batch_idx],
                sentences_ready_length=x_length_batches[batch_idx],
                manual_attention_rate=manual_attention_rate or np.zeros(
                    [len(x_batches[batch_idx]), self.sentence_fixed_len],
                    dtype=np.float)

            )
            predict_intent.append(tmp_predict_intent)
            attention_rate.append(tmp_attention_rate)

        predict_intent = np.concatenate(predict_intent, axis=0)
        tmp = np.concatenate(attention_rate, axis=0)
        attention_rate = []

        if output_file == '':
            # 如果不输出的话 输出的attention 不要设置成str.
            for i in tmp:
                attention_rate.append([(round(v, 6)) for v in i])
            attention_rate = np.array(attention_rate)
            return x, y, x_length, attention_rate, predict_intent

        for i in tmp:
            attention_rate.append([str(round(v, 6)) for v in i])

        with open(output_file, 'w') as fp:
            for i in range(len(x)):
                fp.writelines((y[i] + ',' + ','.join(x[i][:x_length[i]]) + ',\n'))
                fp.writelines((self.idx2label_dict[predict_intent[i]] + ',' + ','.join(
                    attention_rate[i][:x_length[i]]) + ',\n'))


if __name__ == '__main__':

    word2idx = joblib.load('./preprocessing_data/word2idx.pkl')
    label2idx = joblib.load('./preprocessing_data/label2idx.pkl')
    idx2vec = joblib.load('./preprocessing_data/idx2vec.pkl')
    idx2label = joblib.load('./preprocessing_data/idx2label.pkl')

    fold_number = 10
    raw_data = DataModel(fold_number)
    for fold_idx in range(fold_number):
        test_monitor = Monitor(data=raw_data.choice_fold(fold_idx), word2idx_dict=word2idx, label2idx_dict=label2idx,
                               idx2vec_dict=idx2vec, idx2label_dict=idx2label,
                               sentence_fixed_len=50, learning_rate=0.001, word_vec_size=400, hidden_num=50,
                               label_num=31, k_model_src='./test_model/' + str(fold_idx) + '/')
        # test_monitor.load_model(model_src='./test_model/')
        test_monitor.train(batch_size=32, iter_num=200, count_down=10, input_keep_prob=0.4, state_keep_prob=0.5,
                           output_keep_prob=0.6)
        test_monitor.get_attention_information('./data' + str(fold_idx) + '.csv')
