from models.AESI import AESI
from fastNLP import cache_results
from fastNLP import Trainer, GradientClipCallback, WarmupCallback
from torch import optim
from fastNLP import SpanFPreRecMetric, BucketSampler
from fastNLP.embeddings import StaticEmbedding, BertEmbedding, StackEmbedding
from modules.pipe import CNNERPipe
from get_knowledge import generate_knowledge_api

import os
import argparse
from modules.callbacks import EvaluateCallback

from datetime import datetime
import random
import numpy as np

import torch

from run_token_level_classification import BertTokenizer, ZenNgramDict, ZenForTokenClassification, load_examples, DataLoader, SequentialSampler
from utils_token_level_task import PeopledailyProcessor

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='resume', choices=['weibo', 'resume', 'ontonote4', 'msra'])
parser.add_argument('--seed', type=int, default=14)
parser.add_argument('--log', type=str, default=None)
parser.add_argument('--bert_model', type=str, required=True)
parser.add_argument('--zen_model', type=str, default="")
parser.add_argument('--pool_method', type=str, default="first", choices=["first", "last", "avg", "max"])
parser.add_argument('--trans_dropout', type=float, default=0.2)
parser.add_argument('--kv_attn_type', type=str, default='dot')
parser.add_argument('--key_embed_dropout', type=float, default=0.2)
parser.add_argument('--pos_th', type=int, default=20)
parser.add_argument('--dep_th', type=int, default=20)
parser.add_argument('--chunk_th', type=int, default=20)
parser.add_argument('--memory_dropout', type=float, default=0.2)
parser.add_argument('--fusion_type', type=str, default='gate-concat',
                    choices=['concat', 'add', 'concat-add', 'gate-add', 'gate-concat'])
parser.add_argument('--fusion_dropout', type=float, default=0.2)
parser.add_argument('--highway_layer', type=int, default=0)
parser.add_argument('--fc_dropout', type=float, default=0.4)
parser.add_argument('--warmup_steps', type=float, default=0.01)
parser.add_argument('--feature_level', type=str, default="all", choices=["train", "all"])
parser.add_argument('--optim_type', type=str, default="sgd", choices=["sgd", "adam"])
parser.add_argument('--knowledge_type', type=str, default="123", choices=["123", "12", "13", "23"])


args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

save_path = None

setup_seed(args.seed)

dataset = args.dataset
if dataset == 'resume':
    n_heads = 2
    head_dims = 128
    num_layers = 2
    lr = 0.0007
    attn_type = 'adatrans'
    n_epochs = 50
elif dataset == 'weibo':
    n_heads = 4
    head_dims = 256
    num_layers = 1
    lr = 0.001
    attn_type = 'adatrans'
    n_epochs = 100
elif dataset == 'ontonote4':
    n_heads = 4
    head_dims = 48
    num_layers = 2
    lr = 0.0007
    attn_type = 'adatrans'
    n_epochs = 100
elif dataset == 'msra':
    n_heads = 6
    head_dims = 80
    num_layers = 2
    lr = 0.0007
    attn_type = 'adatrans'
    n_epochs = 100

pos_embed = None

batch_size = 32
warmup_steps = args.warmup_steps
after_norm = 1
model_type = 'transformer'
normalize_embed = True

dropout = 0.15
fc_dropout = 0.4

encoding_type = 'bioes'
name = 'caches/{}_{}_{}_{}_{}_{}_{}.pkl'.format(dataset, model_type, encoding_type, normalize_embed, args.pos_th,
                                                args.dep_th, args.chunk_th)
d_model = n_heads * head_dims
dim_feedforward = int(2 * d_model)

save_path = None

logPath = args.log

def write_log(sent):
    with open(logPath, "a+", encoding="utf-8") as f:
        f.write(sent)
        f.write("\n")


@cache_results(name, _refresh=False)
def load_data():
    # 替换路径
    if dataset == 'ontonote4':
        paths = {'train':'data/ontonote4/train.txt',
                 "dev":'data/ontonote4/test.txt',
                 "test":'data/ontonote4/test.txt'}
        min_freq = 2
    elif dataset == 'weibo':
        paths = {'train': 'data/weibo/train.txt',
                 'dev':'data/weibo/test.txt',
                 'test':'data/weibo/test.txt'}
        min_freq = 1
    elif dataset == 'resume':
        paths = {'train': 'data/resume/train.txt',
                 'dev':'data/resume/test.txt',
                 'test':'data/resume/test.txt'}
        min_freq = 1
    elif dataset == 'msra':
        paths = {'train': 'data/msra/train.txt',
                 'dev': 'data/msra/test.txt',
                 'test':'data/msra/test.txt'}
        min_freq = 2
    data_bundle = CNNERPipe(bigrams=True, encoding_type=encoding_type).process_from_file(paths)

    train_feature_data, test_feature_data, feature2count, feature2id, id2feature = generate_knowledge_api(
            os.path.join("data", dataset), "all", args.feature_level
        )

    embed = StaticEmbedding(data_bundle.get_vocab('chars'),
                            model_dir_or_name='data/gigaword_chn.all.a2b.uni.ite50.vec',
                            min_freq=1, only_norm_found_vector=normalize_embed, word_dropout=0.01, dropout=0.3)

    tencent_embed = StaticEmbedding(data_bundle.get_vocab('chars'),
                            model_dir_or_name='data/tencent_unigram.txt',
                            min_freq=1, only_norm_found_vector=normalize_embed, word_dropout=0.01, dropout=0.3)

    bi_embed = StaticEmbedding(data_bundle.get_vocab('bigrams'),
                               model_dir_or_name='data/gigaword_chn.all.a2b.bi.ite50.vec',
                               word_dropout=0.02, dropout=0.3, min_freq=min_freq,
                               only_norm_found_vector=normalize_embed, only_train_min_freq=True)

    bert_embed = BertEmbedding(vocab=data_bundle.get_vocab('chars'), model_dir_or_name=args.bert_model, layers='-1',
                               pool_method=args.pool_method, word_dropout=0, dropout=0.5, include_cls_sep=False,
                               pooled_cls=True, requires_grad=False, auto_truncate=False)

    embed = StackEmbedding([embed, tencent_embed, bert_embed], dropout=0, word_dropout=0.02)

    return data_bundle, embed, bi_embed, train_feature_data, test_feature_data, feature2count, feature2id, id2feature

data_bundle, embed, bi_embed, train_feature_data, test_feature_data, feature2count, feature2id, id2feature = load_data()

train_data = list(data_bundle.get_dataset("train"))
test_data = list(data_bundle.get_dataset("test"))

print(len(train_data), len(train_feature_data[0]), len(train_feature_data[1]), len(train_feature_data[2]))
print(len(test_data), len(test_feature_data[0]), len(test_feature_data[1]), len(test_feature_data[2]))

vocab_size = len(data_bundle.get_vocab('chars'))
feature_vocab_size = len(feature2id)

# ZEN part
zen_model = None
zen_train_dataset = None
zen_test_dataset = None
if args.zen_model:
    print("[Info] Use ZEN !!! ")
    zen_model_path = args.zen_model
    processor = PeopledailyProcessor(dataset=dataset)
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(zen_model_path, do_lower_case=False)
    ngram_dict = ZenNgramDict(zen_model_path, tokenizer=tokenizer)
    zen_model = ZenForTokenClassification.from_pretrained(zen_model_path,
                                                      cache_dir="caches/",
                                                      num_labels=len(label_list),
                                                      multift=False)
    zen_model = zen_model.bert
    zen_model.to(device)
    zen_model.eval()
    data_dir = os.path.join("data", dataset)
    max_seq_len = 512

    zen_train_dataset = load_examples(data_dir, max_seq_len, tokenizer, ngram_dict, processor, label_list, mode="train")
    zen_test_dataset = load_examples(data_dir, max_seq_len, tokenizer, ngram_dict, processor, label_list, mode="test")

    print("[Info] Zen Mode, Zen dataset loaded ...")


model = AESI(tag_vocab=data_bundle.get_vocab('target'), embed=embed, num_layers=num_layers,
              d_model=d_model, n_head=n_heads,
              feedforward_dim=dim_feedforward, dropout=args.trans_dropout,
              after_norm=after_norm, attn_type=attn_type,
              bi_embed=bi_embed,
              fc_dropout=fc_dropout,
              pos_embed=pos_embed,
              scale=attn_type=='transformer',
              use_knowledge=True,
              feature2count=feature2count,
              vocab_size=vocab_size,
              feature_vocab_size=feature_vocab_size,
              kv_attn_type=args.kv_attn_type,
              memory_dropout=args.memory_dropout,
              fusion_dropout=args.fusion_dropout,
              fusion_type=args.fusion_type,
              highway_layer=args.highway_layer,
              key_embed_dropout=args.key_embed_dropout,
              knowledge_type=args.knowledge_type,
              use_zen=args.zen_model!=""
              )

if args.optim_type == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
else:
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))


callbacks = []
clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
evaluate_callback = EvaluateCallback(data=data_bundle.get_dataset('test'),
                                     use_knowledge=True,
                                     knowledge_type=args.knowledge_type,
                                     pos_th=args.pos_th,
                                     dep_th=args.dep_th,
                                     chunk_th=args.chunk_th,
                                     test_feature_data=test_feature_data,
                                     feature2count=feature2count,
                                     feature2id=feature2id,
                                     id2feature=id2feature,
                                     use_zen=args.zen_model!="",
                                     zen_model=zen_model,
                                     zen_dataset=zen_test_dataset
                                     )
if warmup_steps>0:
    warmup_callback = WarmupCallback(warmup_steps, schedule='linear')
    callbacks.append(warmup_callback)
callbacks.extend([clip_callback, evaluate_callback])

trainer = Trainer(data_bundle.get_dataset('train'), model, optimizer, batch_size=batch_size, sampler=BucketSampler(),
                  num_workers=2, n_epochs=80, dev_data=data_bundle.get_dataset('test'),
                  metrics=SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'), encoding_type=encoding_type),
                  dev_batch_size=batch_size, callbacks=callbacks, device=device, test_use_tqdm=False,
                  use_tqdm=True, print_every=300, save_path=save_path,
                  use_knowledge=True,
                  knowledge_type=args.knowledge_type,
                  pos_th=args.pos_th,
                  dep_th=args.dep_th,
                  chunk_th=args.chunk_th,
                  train_feature_data=train_feature_data,
                  test_feature_data=test_feature_data,
                  feature2count=feature2count,
                  feature2id=feature2id,
                  id2feature=id2feature,
                  logger_func=write_log,
                  use_zen=args.zen_model!="",
                  zen_model=zen_model,
                  zen_train_dataset=zen_train_dataset,
                  zen_dev_dataset=zen_test_dataset
                  )

trainer.train(load_best_model=False)
