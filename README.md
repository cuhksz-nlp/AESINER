# AESINER

This is the implementation of [Improving Named Entity Recognition with Attentive Ensemble of Syntactic Information](https://arxiv.org/pdf/2010.15466.pdf) at Findings of EMNLP-2020.

## Citations

If you use or extend our work, please cite our paper at EMNLP2020.
```
@inproceedings{nie-emnlp-2020-aesiner,
    title = "Improving Named Entity Recognition with Attentive Ensemble of Syntactic Information",
    author = "Nie, Yuyang and
      Tian, Yuanhe  and
      Song, Yan  and
      Ao, Xiang and
      Wan, Xiang",
    booktitle = "Findings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2020",
}
```

## Requirements

- `torch==1.1.0`
- `spacy==2.2.4`
- `tqdm==4.38.0`
- `fastNLP==0.5`

## Download Stanford CoreNLP

You can get the Stanford CoreNLP toolkits from the script `stanford.sh`

## Download Pre-trained Embeddings

For English NER, we use three types of word embeddings, namely GloVe, ELMo and BERT. Among them, GloVe and ELMo can be automatically 
downloaded by running the script `run_en.py`; bert can be downloaded pre-trained BERT-large-cased 
from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz). 
If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

For Chinese NER, we also use three types of word embeddings, namely Giga, Tencent Embedding and ZEN. Among them, Giga can be downloaded from [here](https://github.com/jiesutd/LatticeLSTM), 
Tencent Embedding can be downloaded from [here](https://ai.tencent.com/ailab/nlp/zh/embedding.html), ZEN can be downloaded from [here](https://github.com/sinovation/ZEN)

All pretrained embeddings should be placed in `./data/`

## Download AESINER

You can download the models we trained for each dataset from [here](data/aesiner.md). 

## Run on sample data

Run `run_en.py` to train a model on the small sample data under the `sample_data` directory.

## Datasets

We use three English datasets (ON5e, WN16, WN17) and three Chinese datasets (ON4c, RE, WE) in our paper. 

For `ON5e` and `ON4c`, you need to obtain the official data first, and then put the data in `data/ON5e` and `data/ON4c`, respectively.

For `WN16`, you can download the dataset from [here](https://github.com/aritter/twitter_nlp/tree/master/data/annotated/wnut16/data) and then put the `train`, `dev` and  `test` files in `data/W16`.

For `WN17`, you can download the dataset from [here](https://github.com/gaguilar/NER-WNUT17/tree/master/data) and then put the `emerging.train.conll`, `emerging.dev.conll` and `emerging.test.conll` files in `data/W17`. 

For `RE`, you can download the dataest from [here](https://github.com/jiesutd/LatticeLSTM/tree/master/ResumeNER) and then put the `train.char.bmes`, `dev.char.bmes` and `test.char.bmes` files in `data/RE`.

For `WE`, you can download the dataset from [here](https://github.com/hltcoe/golden-horse/tree/master/data) and then put the `weiboNER_2nd_conll.train`, `weiboNER_2nd_conll.dev` and `weiboNER_2nd_conll.test` files in `data/WE`.

All the data files should be named as `train.txt`, `test.txt` and `dev.txt` in corresponding dictionaries. 

For all datasets, you can run the script `data_preprocess.py` with `python data_preprocess.py --dataset ${dataset}$ --data_dir ${data_dir}$` to preprocess the aforementioned datasets automatically.

## Data Preprocess

After downloading `Tencent Embedding`, you need to extract the unigrams according to `python data_process.py --file_path=${PATH_TO_TENCENT_EMBEDDING}$`

For all syntactic information, you can run the script `data_helper.py` to get the syntactic information needed to run our model, you can run the script by `python data_helper.py --dataset $DATASET_NAME$`

## Training

You can find the command lines to train models on a specific dataset in `run_en.py` for English datasets and `run_cn.py` for Chinese datasets. 

