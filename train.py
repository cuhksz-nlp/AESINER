from models.SYNTAX import SyntaxNER
from support import cache_results
from support import Trainer, GradientClipCallback, WarmupCallback
from torch import optim
from support import SpanFPreRecMetric, BucketSampler
from support.embeddings import StaticEmbedding, StackEmbedding, ElmoEmbedding, BertEmbedding
from modules.pipe import ENNERPipe
from modules.callbacks import EvaluateCallback

from get_knowledge import generate_knowledge_api

import os
import argparse
from datetime import datetime

import random
import numpy as np
import torch


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='test_dataset')
parser.add_argument('--bert_model', type=str, required=True)
args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(14)


dataset = args.dataset

n_heads = 12
head_dims = 128
num_layers = 2
lr = 0.0001
attn_type = 'adatrans'
optim_type = 'adam'
trans_dropout = 0.2
batch_size = 32


char_type = 'adatrans'

# positional_embedding
pos_embed = None

model_type = 'bert_elmo'
elmo_model = "en-original"
warmup_steps = 0.01
after_norm = 1
fc_dropout = 0.2
normalize_embed = True

encoding_type = 'bioes'
d_model = n_heads * head_dims
dim_feedforward = int(2 * d_model)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

knowledge = True
knowledge_type = "123"
pos_th = 20
dep_th = 20
chunk_th = 20
feature_level = "all"
key_embed_dropout = 0.2
memory_dropout = 0.2
fusion_dropout = 0.2
kv_attn_type = "dot"
fusion_type = "gate-concat"
highway_layer = 0


def print_time():
    now = datetime.now()
    return "-".join([str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute), str(now.second)])


name = 'caches/bert_elmo_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(dataset, model_type, encoding_type, char_type,
                                                     normalize_embed, knowledge_type, pos_th, dep_th, chunk_th)
save_path = None

logPath = "log/log_{}_{}_{}.txt".format(dataset, knowledge_type, print_time())


def write_log(sent):
    with open(logPath, "a+", encoding="utf-8") as f:
        f.write(sent)
        f.write("\n")


# @cache_results(name, _refresh=False)
def load_data():
    paths = {
        "train": "data/{}/train.txt".format(dataset),
         "test": "data/{}/test.txt".format(dataset),
         "dev": "data/{}/dev.txt".format(dataset)
             }
    data = ENNERPipe(encoding_type=encoding_type).process_from_file(paths)

    if knowledge:
        train_feature_data, test_feature_data, dev_feature_data, feature2count, feature2id, id2feature = generate_knowledge_api(
            os.path.join("data", dataset), "all", feature_level
        )

    word_embed = StaticEmbedding(vocab=data.get_vocab('words'),
                                 model_dir_or_name='en-glove-6b-100d',
                                 requires_grad=True, lower=True, word_dropout=0, dropout=0.5,
                                 only_norm_found_vector=normalize_embed)
    data.rename_field('words', 'chars')

    embed = ElmoEmbedding(vocab=data.get_vocab('chars'), model_dir_or_name=elmo_model, layers='mix', requires_grad=False,
                          word_dropout=0.0, dropout=0.5, cache_word_reprs=False)
    embed.set_mix_weights_requires_grad()

    bert_embed = BertEmbedding(vocab=data.get_vocab('chars'), model_dir_or_name=args.bert_model, layers='-1',
                               pool_method="first", word_dropout=0, dropout=0.5, include_cls_sep=False,
                               pooled_cls=True, requires_grad=False, auto_truncate=True)

    embed = StackEmbedding([embed, bert_embed, word_embed], dropout=0, word_dropout=0.02)

    return data, embed, train_feature_data, test_feature_data, dev_feature_data, feature2count, feature2id, id2feature


data_bundle, embed, train_feature_data, test_feature_data, dev_feature_data, feature2count, feature2id, id2feature = load_data()

train_data = list(data_bundle.get_dataset("train"))
test_data = list(data_bundle.get_dataset("test"))
dev_data = list(data_bundle.get_dataset("dev"))

print(len(train_data), len(train_feature_data[0]), len(train_feature_data[1]), len(train_feature_data[2]))
print(len(test_data), len(test_feature_data[0]), len(test_feature_data[1]), len(test_feature_data[2]))
print(len(dev_data), len(dev_feature_data[0]), len(dev_feature_data[1]), len(dev_feature_data[2]))


vocab_size = len(data_bundle.get_vocab('chars'))
feature_vocab_size = len(feature2id)


model = SyntaxNER(tag_vocab=data_bundle.get_vocab('target'), embed=embed, num_layers=num_layers,
              d_model=d_model, n_head=n_heads,
              feedforward_dim=dim_feedforward, dropout=trans_dropout,
              after_norm=after_norm, attn_type=attn_type,
              bi_embed=None,
              fc_dropout=fc_dropout,
              pos_embed=pos_embed,
              scale=attn_type=='naive',
              use_knowledge=knowledge,
              feature2count=feature2count,
              vocab_size=vocab_size,
              feature_vocab_size=feature_vocab_size,
              kv_attn_type=kv_attn_type,
              memory_dropout=memory_dropout,
              fusion_dropout=fusion_dropout,
              fusion_type=fusion_type,
              highway_layer=highway_layer,
              key_embed_dropout=key_embed_dropout,
              knowledge_type=knowledge_type
              )

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

callbacks = []
clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
evaluate_callback = EvaluateCallback(data=data_bundle.get_dataset('test'),
                                     use_knowledge=knowledge,
                                     knowledge_type=knowledge_type,
                                     pos_th=pos_th,
                                     dep_th=dep_th,
                                     chunk_th=chunk_th,
                                     test_feature_data=test_feature_data,
                                     feature2count=feature2count,
                                     feature2id=feature2id,
                                     id2feature=id2feature
                                     )

if warmup_steps > 0:
    warmup_callback = WarmupCallback(warmup_steps, schedule='linear')
    callbacks.append(warmup_callback)
callbacks.extend([clip_callback, evaluate_callback])

trainer = Trainer(
    data_bundle.get_dataset('train'),
    model,
    optimizer,
    batch_size=batch_size,
    sampler=BucketSampler(),
    num_workers=0,
    n_epochs=100,
    dev_data=data_bundle.get_dataset('dev'),
    metrics=SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'), encoding_type=encoding_type),
    dev_batch_size=batch_size, callbacks=callbacks, device=device, test_use_tqdm=False,
    use_tqdm=True,
    print_every=300,
    save_path=save_path,
    use_knowledge=True,
    knowledge_type=knowledge_type,
    pos_th=pos_th,
    dep_th=dep_th,
    chunk_th=chunk_th,
    train_feature_data=train_feature_data,
    test_feature_data=dev_feature_data,
    feature2count=feature2count,
    feature2id=feature2id,
    id2feature=id2feature,
    logger_func=write_log
)

trainer.train(load_best_model=False)
