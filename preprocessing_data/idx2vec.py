import joblib
import pickle
import numpy as np

wordVec = joblib.load("/home/bringtree/wordvec/10G_dict.pkl")

with open("./word2idx.pkl", "rb") as fp:
    word2idx = pickle.load(fp)
idx2vec = {}

for k, v in word2idx.items():
    if k in wordVec:
        idx2vec[word2idx[k]] = wordVec[k]

idx2vec[word2idx['unknown_word']] = np.ones(400)
idx2vec[word2idx['blank_word']] = np.zeros(400)

with open('./idx2vec.pkl', "wb") as fp:
    pickle.dump(idx2vec, fp)
