import jieba
import pickle

data_file_list = [
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
word_dict = {}
word_dict_reversed = {}
all_sentence = []
for file in data_file_list:
    with open("../data/" + file) as fp:
        all_sentence.extend(fp.readlines())

i = 1
all_sentence = [jieba.lcut(v) for v in all_sentence]
for sentence in all_sentence:
    for word in sentence:
        if word not in word_dict:
            word_dict[word] = i
            word_dict_reversed[i] = word
            i += 1
word_dict['unknown_word'] = i
word_dict_reversed[i] = 'unknown_word'
i += 1
word_dict['blank_word'] = i
word_dict_reversed[i] = 'blank_word'
i += 1
len(word_dict)
with open("./word2idx.pkl", "wb") as fp:
    pickle.dump(word_dict, fp)
with open("./idx2word.pkl", "wb") as fp:
    pickle.dump(word_dict_reversed, fp)
