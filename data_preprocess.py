import os
import argparse


def weibo_preprocessor(dataset, data_path, mode="train"):
    with open(data_path, "r", encoding="utf-8") as f:
        data = f.read().split("\n")
    new_path = "data/{}/{}.txt".format(dataset, mode)
    data_dir = "data/{}".format(dataset)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    ret_list = []
    for row in data:
        if not row:
            ret_list.append("")
        else:
            row = row.split("\t")
            if row:
                ret_list.append("{} {}".format(row[0][:1], row[1]))
    with open(new_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ret_list))


def w16_preprocess(dataset, data_path, mode="train"):
    with open(data_path, "r", encoding="utf-8") as f:
        data = f.read().split("\n")
    new_path = "data/{}/{}.txt".format(dataset, mode)
    data_dir = "data/{}".format(dataset)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    ret_list = []
    for row in data:
        ret_list.append(row.replace("\t", " "))
    with open(new_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ret_list))


def w17_preprocess(dataset, data_path, mode="train"):
    with open(data_path, "r", encoding="utf-8") as f:
        data = f.read().split("\n")
    new_path = "data/{}/{}.txt".format(dataset, mode)
    data_dir = "data/{}".format(dataset)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    ret_list = []
    for row in data:
        ret_list.append(row.replace("\t", " "))
    with open(new_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ret_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='weibo', choices=['WE', 'ON5e', 'ON4c', 'WN16', 'WN17', 'RE'])
    parser.add_argument('--data_dir', type=str, default='')
    args = parser.parse_args()

    dataset = args.dataset
    data_dir = args.data_dir

    if dataset == "WE":
        modes = ["train", "test", "dev"]
        for mode in modes:
            data_path = "{}/weiboNER_2nd_conll.{}".format(data_dir, mode)
            weibo_preprocessor(dataset, data_path, mode)
    elif dataset in ["ON5e", "ON4c"]:
        raise RuntimeError("you can refer https://github.com/yhcc/OntoNotes-5.0-NER to download and preprocess.")
    elif dataset == "RE":
        pass
    elif dataset == "WN16":
        modes = ["train", "test", "dev"]
        for mode in modes:
            data_path = "{}/{}".format(data_dir, mode)
            w16_preprocess(dataset, data_path, mode)
    elif dataset == "WN17":
        modes = ["dev"]
        for mode in modes:
            data_path = "{}/emerging.{}.conll".format(data_dir, mode)
            w17_preprocess(dataset, data_path, mode)
