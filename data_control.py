# 筛选出符合条件的的数据,并编码打包成batch。
from data_model import *
import joblib
import jieba
from keras.preprocessing.sequence import pad_sequences
import numpy as np


class DataControl:
    """
    In training, we need to feed the model with the data.
    And the DataControl with help us to make the raw data(in Chinese) convert to the word vector.
    Also it will pack the data to batches.
    We also can make the batch of data in unordered.
    """

    def __init__(self, data, word2idx_dict, label2idx_dict, idx2vec_dict, sentence_fixed_len, word_vec_size):
        """

        :param data:
        :param word2idx_dict: A dict.   key: word           value: the index of word
        :param label2idx_dict:  A dict. key: label          value: the index of label
        :param idx2vec_dict:  A dict.   key: index of word  value: the word vector of word
        :param sentence_fixed_len:  A number means the max length of sentences.
        """
        self.x_train, self.y_train, self.x_dev, self.y_dev = data

        self.sentence_fixed_len = sentence_fixed_len
        self.word2idx_dict = word2idx_dict
        self.label2idx_dict = label2idx_dict
        self.idx2vec_dict = idx2vec_dict
        self.word_vec_size = word_vec_size

        self.x_train = [jieba.lcut(v) for v in self.x_train]
        self.y_train = [v for v in self.y_train]
        self.x_length_train = [len(v) for v in self.x_train]

        self.x_dev = [jieba.lcut(v) for v in self.x_dev]
        self.y_dev = [v for v in self.y_dev]
        self.x_length_dev = [len(v) for v in self.x_dev]

        self.idx_x_train, self.idx_y_train, self.idx_x_dev, self.idx_y_dev = self.encoder_data()

    def encoder_data(self):
        """
        Encoding the raw data to number.
        the sentence will be padding to the fixed length. (length = self.sentence_fixed_len.)
        the word will be embedding to word vector.
        the label will be encoder to index.
        :return: idx_x_train, idx_y_train, idx_x_dev, idx_y_dev. they had been encoder to number.
        """
        idx_x_train = [[self.word2idx_dict[word] for word in sentence] for sentence in self.x_train]
        idx_y_train = [self.label2idx_dict[label] for label in self.y_train]
        idx_x_dev = [[self.word2idx_dict[word] for word in sentence] for sentence in self.x_dev]
        idx_y_dev = [self.label2idx_dict[label] for label in self.y_dev]

        # make indefinite length to fixed length
        idx_x_train = pad_sequences(idx_x_train, value=-1, maxlen=self.sentence_fixed_len, padding='post')
        idx_x_dev = pad_sequences(idx_x_dev, value=-1, maxlen=self.sentence_fixed_len, padding='post')

        # embedding the word
        idx_x_train = self.embedding2vec(idx_x_train)
        idx_x_dev = self.embedding2vec(idx_x_dev)

        return idx_x_train, idx_y_train, idx_x_dev, idx_y_dev

    def embedding2vec(self, x):
        """
        let the word which has been encoding to index be embedded to word vectors.
        Warning：
            when the word index is not in the idx2vec_dict, it will be embedding np.ones(400).
            when the word is blank word, it will be embedding np.zeros(400).
        :param x: the sentences list whose word has been encoding to index.
        :return: the sentences list whose word has been encoding to word vector.
        """
        tmp_x = []
        for sentences in x:
            tmp_sentence = []
            for word in sentences:
                if word == -1:
                    # blank_word np.zeros(400)
                    tmp_sentence.append(np.zeros(self.word_vec_size))
                elif word not in self.idx2vec_dict:
                    # unknown_word np.ones(400)
                    tmp_sentence.append(np.ones(self.word_vec_size))
                else:
                    tmp_sentence.append(self.idx2vec_dict[word])
            tmp_x.append(tmp_sentence)
        return tmp_x

    def generator_batches(self, batch_size, train_mode, shuffle_mode):
        """
        Pack the data to batches.
        :param train_mode: A boolean make a decision to return the type of data
        :param batch_size: A int to decision the size of batches
        :param shuffle_mode: A boolean can use to shuffle the data.
        :return: x_batches, y_batches, x_length_batches, batches_num
        """

        if train_mode is True:
            idx_x, idx_y, x_length = self.idx_x_train, self.idx_y_train, self.x_length_train
        else:
            idx_x, idx_y, x_length = self.idx_x_dev, self.idx_y_dev, self.x_length_dev

        x_batches = []
        y_batches = []
        x_length_batches = []
        batches_num = int(len(idx_x) / batch_size) + 1

        # expand the data until can be divided by batch_size
        idx_x = idx_x + [np.zeros(shape=[self.sentence_fixed_len, self.word_vec_size]) for _ in
                         range(batches_num * batch_size - len(idx_x))]
        idx_y = idx_y + [0 for _ in range(batches_num * batch_size - len(idx_y))]
        x_length = x_length + [0 for _ in range(batches_num * batch_size - len(x_length))] + \
                   [0 for _ in range(batches_num * batch_size - len(idx_y))]

        if shuffle_mode is True:
            train_data = list(zip(idx_x, idx_y, x_length))
            np.random.shuffle(train_data)
            idx_x, idx_y, x_length = zip(*train_data)

        for i in range(batches_num):
            x_batches.append(idx_x[i * batch_size:(i + 1) * batch_size])
            y_batches.append(idx_y[i * batch_size:(i + 1) * batch_size])
            x_length_batches.append(x_length[i * batch_size:(i + 1) * batch_size])

        return x_batches, y_batches, x_length_batches, batches_num


if __name__ == "__main__":
    data_model = DataModel(10)
    word2idx = joblib.load('./preprocessing_data/word2idx.pkl')
    label2idx = joblib.load('./preprocessing_data/label2idx.pkl')
    idx2vec = joblib.load('./preprocessing_data/idx2vec.pkl')

    o = DataControl(data_model.choice_fold(1), word2idx, label2idx, idx2vec, 20,400)
    tmp = o.generator_batches(64, True, shuffle_mode=True)
