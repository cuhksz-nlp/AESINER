import os

dataset = "weibo"
seed = 14

os.system("python3 train_zen_cn.py --dataset {} --bert_model data/bert-base-chinese"
          " --pool_method first --seed {} --log log/{}_zen_enc_out_kv_th_10_seed_{}.txt "
          "--zen_model zen_base/ "
          "--trans_dropout 0.2 --key_embed_dropout 0.2 --pos_th 5 --dep_th 5 --chunk_th 5 "
          "--memory_dropout 0.2 --fusion_type gate-concat --fusion_dropout 0.2 --fc_dropout 0.4".format(dataset, seed, dataset, seed))
