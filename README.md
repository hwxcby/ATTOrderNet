# ATTOrderNet
Code for the paper “Deep attentive sentence ordering network”.

### Requirements
python 2.7.12, tensorflow 1.12.3

### Run
1、Put the SIND data into the file *all_data_sind*.  

2、To generate the vocabulary, run: 
``` python2 data_vocab.py ```  

3、The config file is in the *save_file/lstm*.  

4、Train and evaluate, run:
```
CUDA_VISIBLE_DEVICES=1 python2 main_file.py --load-config --train-test train --weight-path ./save_file/lstm
```

### Citation
```
@inproceedings{cui2018deep,
title={Deep attentive sentence ordering network},
author={Cui, Baiyun and Li, Yingming and Chen, Ming and Zhang, Zhongfei},
booktitle={Proceedings of EMNLP},
pages={4340--4349},
year={2018}
}
```

Some codes refer to the paper “End-to-End Neural Sentence Ordering Using Pointer Network”. Thanks to [SentenceOrdering_PTR](https://github.com/FudanNLP/SentenceOrdering_PTR).
