开始整理论文用到的代码。 主要是几篇常用于lstm的attention计算的代码。
代码太多太杂。  有空陆续释放。 
有5种

第一份
```BibTeX
@inproceedings{zhou2016attention,
  title={Attention-based bidirectional long short-term memory networks for relation classification},
  author={Zhou, Peng and Shi, Wei and Tian, Jun and Qi, Zhenyu and Li, Bingchen and Hao, Hongwei and Xu, Bo},
  booktitle={Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  volume={2},
  pages={207--212},
  year={2016}
}
```
可视化的代码在我github下的另外一个库里
[可视化attention](https://github.com/bringtree/attention_visual_heatmap)

数据集是数据集是SMP-EUPT数据集。
直接下载 然后在代码目录里新建```data``` 文件夹,把数据集解压进来.
```
├── data
│   ├── develop_app.txt
│   ├── develop_bus.txt
│   ├── develop_calc.txt
│   ├── develop_chat.txt
│   ├── develop_cinemas.txt
│   ├── develop_contacts.txt
│   ├── develop_cookbook.txt
│   ├── develop_datetime.txt
│   ├── develop_email.txt
│   ├── develop_epg.txt
│   ├── develop_flight.txt
│   ├── develop_health.txt
│   ├── develop_lottery.txt
│   ├── develop_map.txt
│   ├── develop_match.txt
│   ├── develop_message.txt
│   ├── develop_music.txt
│   ├── develop_news.txt
│   ├── develop_novel.txt
│   ├── develop_poetry.txt
│   ├── develop_radio.txt
│   ├── develop_riddle.txt
│   ├── develop_schedule.txt
│   ├── develop_stock.txt
│   ├── develop_telephone.txt
│   ├── develop_train.txt
│   ├── develop_translation.txt
│   ├── develop_tvchannel.txt
│   ├── develop_video.txt
│   ├── develop_weather.txt
│   ├── develop_website.txt
│   ├── train_app.txt
│   ├── train_bus.txt
│   ├── train_calc.txt
│   ├── train_chat.txt
│   ├── train_cinemas.txt
│   ├── train_contacts.txt
│   ├── train_cookbook.txt
│   ├── train_datetime.txt
│   ├── train_email.txt
│   ├── train_epg.txt
│   ├── train_flight.txt
│   ├── train_health.txt
│   ├── train_lottery.txt
│   ├── train_map.txt
│   ├── train_match.txt
│   ├── train_message.txt
│   ├── train_music.txt
│   ├── train_news.txt
│   ├── train_novel.txt
│   ├── train_poetry.txt
│   ├── train_radio.txt
│   ├── train_riddle.txt
│   ├── train_schedule.txt
│   ├── train_stock.txt
│   ├── train_telephone.txt
│   ├── train_train.txt
│   ├── train_translation.txt
│   ├── train_tvchannel.txt
│   ├── train_video.txt
│   ├── train_weather.txt
│   └── train_website.txt
├── data_control.py
├── data_model.py
├── main.py
├── model.py
├── monitor.py
├── online.ipynb
├── preprocessing_data
│   ├── idx2label.pkl
│   ├── idx2vec.pkl
│   ├── idx2vec.py
│   ├── idx2word.pkl
│   ├── label2idx.pkl
│   ├── label2idx.py
│   ├── w2idx_idx2w.py
│   ├── word2idx.pkl
│   ├── word2vec.pkl
│   └── word2vec.py
└── start.sh
```
