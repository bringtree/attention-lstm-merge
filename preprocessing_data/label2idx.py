import pickle

labels_list = ['calc',
               'map',
               'telephone',
               'chat',
               'match',
               'train',
               'cinemas',
               'message',
               'translation',
               'contacts',
               'music',
               'tvchannel',
               'cookbook',
               'news',
               'video',
               'datetime',
               'novel',
               'weather',
               'email',
               'poetry',
               'website',
               'epg',
               'radio',
               'flight',
               'riddle',
               'app',
               'health',
               'schedule',
               'bus',
               'lottery',
               'stock']
label2idx = {}
idx2label = {}

for idx, v in enumerate(labels_list):
    label2idx[v] = idx
    idx2label[idx] = v

with open('./label2idx.pkl', 'wb') as fp:
    pickle.dump(label2idx, fp)

with open('./idx2label.pkl', 'wb') as fp:
    pickle.dump(idx2label, fp)
