import os

dataset = "test_dataset"

# --bert_model: the path of the pre-trained BERT model
# --dataset: the dataset name

os.system("python3 train_bert_elmo_en.py --dataset {} --bert_model data/bert-large-cased --seed 14 --log log/{}.txt".format(dataset, dataset))
