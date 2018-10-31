#!/usr/bin/env bash
python3 ./preprocessing_data/w2idx_idx2w.py
python3 ./preprocessing_data/label2idx.py
python3 ./preprocessing_data/idx2vec.py
python3 ./preprocessing_data/word2vec.py
python3 main.py