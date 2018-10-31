# 数据的模型
from sklearn.model_selection import StratifiedKFold


class DataModel:
    def __init__(self, fold_number):
        self.data_file_list = [
            'develop_app.txt',
            'develop_health.txt',
            'develop_schedule.txt',
            'train_calc.txt',
            'train_map.txt',
            'train_telephone.txt',
            'develop_bus.txt',
            'develop_lottery.txt',
            'develop_stock.txt',
            'train_chat.txt',
            'train_match.txt',
            'train_train.txt',
            'develop_calc.txt',
            'develop_map.txt',
            'develop_telephone.txt',
            'train_cinemas.txt',
            'train_message.txt',
            'train_translation.txt',
            'develop_chat.txt',
            'develop_match.txt',
            'develop_train.txt',
            'train_contacts.txt',
            'train_music.txt',
            'train_tvchannel.txt',
            'develop_cinemas.txt',
            'develop_message.txt',
            'develop_translation.txt',
            'train_cookbook.txt',
            'train_news.txt',
            'train_video.txt',
            'develop_contacts.txt',
            'develop_music.txt',
            'develop_tvchannel.txt',
            'train_datetime.txt',
            'train_novel.txt',
            'train_weather.txt',
            'develop_cookbook.txt',
            'develop_news.txt',
            'develop_video.txt',
            'train_email.txt',
            'train_poetry.txt',
            'train_website.txt',
            'develop_datetime.txt',
            'develop_novel.txt',
            'develop_weather.txt',
            'train_epg.txt',
            'train_radio.txt',
            'develop_email.txt',
            'develop_poetry.txt',
            'develop_website.txt',
            'train_flight.txt',
            'train_riddle.txt',
            'develop_epg.txt',
            'develop_radio.txt',
            'train_app.txt',
            'train_health.txt',
            'train_schedule.txt',
            'develop_flight.txt',
            'develop_riddle.txt',
            'train_bus.txt',
            'train_lottery.txt',
            'train_stock.txt',
        ]
        self.fold_number = fold_number
        self.train_data_sentences, self.train_data_labels = self.load_data(file_list=self.data_file_list, mode='train_')
        self.test_data_sentences, self.test_data_labels = self.load_data(file_list=self.data_file_list, mode='develop_')
        self.x_train_k_fold, self.y_train_k_fold, \
        self.x_dev_k_fold, self.y_dev_k_fold = self.k_fold(self.train_data_sentences,
                                                           self.train_data_labels,
                                                           self.fold_number)

    def choice_fold(self, fold_idx):
        return self.x_train_k_fold[fold_idx], self.y_train_k_fold[fold_idx], \
               self.x_dev_k_fold[fold_idx], self.y_dev_k_fold[fold_idx]

    @staticmethod
    def load_data(file_list, mode):
        """
        加载文件，返回他对应的数据
        :param file_list: 数据文件列表
        :param mode: 加载的数据文件的类型 train/develop/test
        :return: sentences和对应的labels
        """
        sentences = []
        labels = []
        for file_name in file_list:
            if mode in file_name:
                with open('./data/' + file_name, encoding='utf8') as fp:
                    tmp = [v.replace('\n', '') for v in fp.readlines()]
                    sentences += tmp
                    labels += [file_name.replace(mode, '').replace('.txt', '') for _ in
                               range(len(tmp))]

        return sentences, labels

    @staticmethod
    def k_fold(sentences, labels, k):

        x_train_k_fold = []
        y_train_k_fold = []

        x_test_k_fold = []
        y_test_k_fold = []

        if k == 1:
            return [sentences], [labels], [], []
        else:
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=2)
            for train_index, test_index in skf.split(sentences, labels):
                x_train_k_fold.append([sentences[idx] for idx in train_index])
                y_train_k_fold.append([labels[idx] for idx in train_index])

                x_test_k_fold.append([sentences[idx] for idx in test_index])
                y_test_k_fold.append([labels[idx] for idx in test_index])

        return x_train_k_fold, y_train_k_fold, x_test_k_fold, y_test_k_fold

    @property
    def test_data(self):
        return self.load_data(file_list=self.data_file_list, mode='develop_')

    @property
    def train_data(self):
        return self.load_data(file_list=self.data_file_list, mode='train_')


if __name__ == '__main__':
    dataMonitor = DataModel(fold_number=10)
    x_train_k_fold, y_train_k_fold, x_dev_k_fold, y_dev_k_fold = dataMonitor.choice_fold(1)
    # how to get the test data
    test_data = dataMonitor.test_data
