from monitor import Monitor
import joblib
from data_model import DataModel
import multiprocessing

word2idx = joblib.load('./preprocessing_data/word2idx.pkl')
label2idx = joblib.load('./preprocessing_data/label2idx.pkl')
idx2vec = joblib.load('./preprocessing_data/idx2vec.pkl')
idx2label = joblib.load('./preprocessing_data/idx2label.pkl')

process_args_list = []
fold_number = 10

raw_data = DataModel(fold_number)


def sub_process(train_args):
    test_monitor = Monitor(word2idx_dict=word2idx, label2idx_dict=label2idx,
                           idx2vec_dict=idx2vec, idx2label_dict=idx2label,
                           data=raw_data.choice_fold(train_args['fold_idx']),
                           sentence_fixed_len=train_args['sentence_fixed_len'],
                           learning_rate=train_args['learning_rate'],
                           word_vec_size=train_args['word_vec_size'],
                           hidden_num=train_args['hidden_num'],
                           label_num=train_args['label_num'],
                           k_model_src=train_args['k_model_src'])

    test_monitor.train(batch_size=train_args['batch_size'], iter_num=train_args['iter_num'],
                       count_down=train_args['count_down'],
                       input_keep_prob=train_args['input_keep_prob'],
                       state_keep_prob=train_args['state_keep_prob'],
                       output_keep_prob=train_args['output_keep_prob'])


pool = multiprocessing.Pool(processes=1)

learning_rate_list = [0.001]
hidden_num_list = [50, 100, 150]
input_keep_prob_list = [0.2, 0.3]
state_keep_prob_list = [0.2, 0.3]
output_keep_prob_list = [0.3]

batch_size_list = [24]

for fold_idx in range(fold_number):
    for learning_rate in learning_rate_list:
        for hidden_num in hidden_num_list:
            for input_keep_prob in input_keep_prob_list:
                for output_keep_prob in output_keep_prob_list:
                    for state_keep_prob in state_keep_prob_list:
                        for batch_size in batch_size_list:
                            train_args = {
                                'fold_idx': fold_idx,
                                'learning_rate': learning_rate,
                                'hidden_num': hidden_num,
                                'input_keep_prob': input_keep_prob,
                                'output_keep_prob': output_keep_prob,
                                'state_keep_prob': state_keep_prob,
                                'batch_size': batch_size,
                                'iter_num': 200,
                                'label_num': 31,
                                'count_down': 10,
                                'word_vec_size': 400,
                                'sentence_fixed_len': 50,
                                'k_model_src': './test_model/' + str(fold_idx) + '/'
                            }
                            k_model_src = './test_model/' + str(fold_idx) + '/'
                            process_args_list.append([train_args])

pool.starmap(sub_process, process_args_list)
