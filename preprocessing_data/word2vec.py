import pickle

with open("./word2idx.pkl", "rb") as fp:
    word2idx = pickle.load(fp)

with open("./idx2vec.pkl", "rb") as fp:
    idx2vec = pickle.load(fp)

word2vec = {}

for k, v in word2idx.items():
    if v in idx2vec:
        word2vec[k] = idx2vec[v]

print(len(word2vec))
with open("./word2vec.pkl", "wb") as fp:
    pickle.dump(word2vec,fp)
