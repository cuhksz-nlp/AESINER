import os

dataset = "test_dataset"

os.system("python3 train_bert_elmo_en.py --dataset {} --bert_model data/bert-large-cased --seed 14 --log log/{}.txt".format(dataset, dataset))
