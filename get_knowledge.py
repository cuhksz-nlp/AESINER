import os
import json
from os import path
from nltk.tree import Tree
from collections import defaultdict

punctuation = ['。', '，', '、', '：', '？', '！', '（', '）', '“', '”', '【', '】']
chunk_pos = ['NP', 'PP', 'VP', 'ADVP', 'SBAR', 'ADJP', 'PRT', 'INTJ', 'CONJP', 'LST']


class StanfordFeatureProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def read_json(self, data_path):
        data = []
        with open(data_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line == '':
                    continue
                data.append(json.loads(line))
        return data

    def _pre_processing(self):
        all_data = self.read_json(path.join(self.data_dir, 'train.stanford.json'))
        gram2count = defaultdict(int)
        pos_tag2count = defaultdict(int)
        chunk_tag2count = defaultdict(int)
        dep_tag2count = defaultdict(int)

        for data in all_data:
            sentences_list = data['sentences']
            for sentence_l in sentences_list:

                tokens = sentence_l['tokens']
                for token in tokens:
                    gram2count[token['originalText']] += 1
                    pos_tag2count[token['pos']] += 1
                    pos_tag2count[token['originalText'] + '_' + token['pos']] += 1
                deparse = sentence_l['basicDependencies']
                for word in deparse:
                    dep_tag2count[word['dep']] += 1
                    dep_tag2count[word['dependentGloss'] + '_' + word['dep']] += 1

                coparse = Tree.fromstring(sentence_l['parse'])
                for s in coparse.subtrees(lambda t: t.label() in chunk_pos):
                    leaves = s.leaves()
                    node = s.label()
                    chunk_tag2count[node] += 1
                    for leaf in leaves:
                        chunk_tag2count[leaf + '_' + node] += 1
                chunk_tag2count['ROOT'] = 100

        print('feature stat')
        print('# of gram: %d' % len(gram2count))
        print('# of pos: %d' % len(pos_tag2count))
        print('# of chunk_tag: %d' % len(chunk_tag2count))
        print('# of dep: %d' % len(dep_tag2count))
        feature2id = {'gram2count': gram2count, 'pos_tag2count': pos_tag2count,
                      'chunk_tag2count': chunk_tag2count, 'dep_tag2count': dep_tag2count}

        with open(path.join(self.data_dir, 'feature2count.json'), 'w', encoding='utf8') as f:
            json.dump(feature2id, f, ensure_ascii=False)
            f.write('\n')

    def read_feature2count(self):
        with open(path.join(self.data_dir, 'feature2count.json'), 'r', encoding='utf8') as f:
            return json.loads(f.read())

    def feature_stat(self):
        all_feature2count = self.read_feature2count()
        feature_num = []
        for feature in ['gram2count', 'pos_tag2count', 'chunk_tag2count', 'dep_tag2count']:
            feature2count = all_feature2count[feature]
            num = 0
            for f, n in feature2count.items():
                if n > 1:
                    num += 1
            feature_num.append(num)
        # feature_num.append(len(all_feature2count['gram2count']))
        # feature_num.append(len(all_feature2count['pos_tag2count']))
        # feature_num.append(len(all_feature2count['chunk_tag2count']))
        # feature_num.append(len(all_feature2count['dep_tag2count']))
        print('max # of features: %d' % max(feature_num))
        return max(feature_num)

    def read_features(self, flag):
        all_data = self.read_json(path.join(self.data_dir, flag + '.stanford.json'))
        all_feature_data = []
        print(len(all_data))
        for data in all_data:
            sentence_len=0
            sentence_feature = []
            sentence = ''
            words = []
            index=[]
            sentences=data['sentences']
            for sentence in sentences:
                tokens = sentence['tokens']
                for token in tokens:
                    feature_dict = {}
                    feature_dict['word'] = token['originalText']
                    words.append(token['word'].replace('\xa0',''))
                    # sentence += token['word']
                    start_index = token['characterOffsetBegin']
                    end_index = token['characterOffsetEnd']
                    feature_dict['char_index'] = [i for i in range(start_index, end_index)]
                    feature_dict['length']= sentence_len+ len(sentence)
                    feature_dict['pos'] = token['pos']
                    sentence_feature.append(feature_dict)
            # df = df.append([{'word': ' ', 'pos': ' '}], ignore_index=True)

                deparse = sentence['basicDependencies']
                for dep in deparse:

                    dependent_index = dep['dependent'] - 1
                    sentence_feature[dependent_index]['dep'] = dep['dep']
                    sentence_feature[dependent_index]['governed_index'] = dep['governor'] - 1

                c_parse = Tree.fromstring(sentence['parse'].replace('\xa0',''))
                current_index = 0
                for s in c_parse.subtrees(lambda t: t.label() in chunk_pos):
                    leaves = s.leaves()

                    if len(leaves) == 0:
                        continue
                    node = s.label()

                    index = words[current_index:].index(leaves[0]) + current_index
                    current_index = index
                    for i, leaf in enumerate(leaves):

                        if 'chunk_tags' not in sentence_feature[index + i]:
                            sentence_feature[index + i]['chunk_tags'] = []
                        sentence_feature[index + i]['chunk_tags'].append(
                            {'chunk_tag': node, 'height': 0, 'range': [index, index + len(leaves)-1]}
                        )
                        for chunk_tag in sentence_feature[index + i]['chunk_tags']:
                            chunk_tag['height'] += 1
                for token in sentence_feature:
                    if 'chunk_tags' not in token:
                        token['chunk_tags'] = [{'chunk_tag': 'ROOT', 'height': 1, 'range': [0, len(sentence_feature)-1]}]

            all_feature_data.append(sentence_feature)
        return all_feature_data


def get_chunk(chunk_tags):
    for chunk_tag in chunk_tags:
        if chunk_tag.get("height") == 1:
            return chunk_tag.get("chunk_tag"), chunk_tag.get("range")


def get_dep(sentence):
    words = [change_word(i["word"]) for i in sentence]
    deps = [i["dep"] + "_dep" for i in sentence]
    dep_matrix = [[0] * len(words) for _ in range(len(words))]
    for i, item in enumerate(sentence):
        governor = item["governed_index"]
        dep_matrix[i][i] = 1
        if governor != -1:
            dep_matrix[i][governor] = 1
            dep_matrix[governor][i] = 1

    ret_list = []
    for word, dep, dep_range in zip(words, deps, dep_matrix):
        ret_list.append({"word": word, "dep": dep, "range": dep_range})
    return ret_list


def change_word(word):
    if "-RRB-" in word:
        return word.replace("-RRB-", ")")
    if "-LRB-" in word:
        return word.replace("-LRB-", "(")
    return word


def filter_useful_feature(feature_list, feature_type):
    ret_list = []
    # [pos, dep, chunk]
    if feature_type == "all":
        ret_list = [[], [], []]
    for i, sentence in enumerate(feature_list):
        ret0 = []
        ret2 = []
        ret_list[1].append(get_dep(sentence))
        for word in sentence:
            ret0.append({"word": change_word(word['word']), "pos": word["pos"] + "_pos"})
            chunk_tag, range_chunk = get_chunk(word["chunk_tags"])
            ret2.append({"word": change_word(word['word']), "chunk": chunk_tag + "_chunk", "range": range_chunk})
        ret_list[0].append(ret0)
        ret_list[2].append(ret2)
        assert len(ret_list[0][i]) == len(ret_list[1][i]) == len(ret_list[2][i])
    print("length: ", len(ret_list[0]), len(ret_list[1]), len(ret_list[2]))
    return ret_list


def get_feature2count(train_features, test_features=None):
    train_pos_features, train_dep_features, train_chunk_features = train_features
    feature2count = defaultdict(int)
    for sent in train_pos_features:
        for item in sent:
            word = item["word"]
            pos = item["pos"]
            pos_feature = word + "_" + pos
            feature2count[pos] += 1
            feature2count[pos_feature] += 1
    for sent in train_dep_features:
        for item in sent:
            word = item["word"]
            dep = item["dep"]
            dep_feature = word + "_" + dep
            feature2count[dep] += 1
            feature2count[dep_feature] += 1
    for sent in train_chunk_features:
        for item in sent:
            word = item["word"]
            chunk = item["chunk"]
            chunk_feature = word + "_" + chunk
            feature2count[chunk] += 1
            feature2count[chunk_feature] += 1
    if test_features:
        test_pos_features, test_dep_features, test_chunk_features = test_features
        for sent in test_pos_features:
            for item in sent:
                word = item["word"]
                pos = item["pos"]
                pos_feature = word + "_" + pos
                feature2count[pos] += 1
                feature2count[pos_feature] += 1
        for sent in test_dep_features:
            for item in sent:
                word = item["word"]
                dep = item["dep"]
                dep_feature = word + "_" + dep
                feature2count[dep] += 1
                feature2count[dep_feature] += 1
        for sent in test_chunk_features:
            for item in sent:
                word = item["word"]
                chunk = item["chunk"]
                chunk_feature = word + "_" + chunk
                feature2count[chunk] += 1
                feature2count[chunk_feature] += 1
    return feature2count


def generate_knowledge_api(data_dir, feature_type="all", level="all"):
    sfp = StanfordFeatureProcessor(data_dir)

    train_feature_data = sfp.read_features(flag="train")
    print("len_train: ", len(train_feature_data))
    test_feature_data = sfp.read_features(flag="test")
    print("len_test: ", len(test_feature_data))
    train_feature_data = filter_useful_feature(train_feature_data, feature_type="all")
    test_feature_data = filter_useful_feature(test_feature_data, feature_type="all")

    assert level in ["all", "train"]

    if level == "train":
        feature2count = get_feature2count(train_feature_data)
    elif level == "all":
        feature2count = get_feature2count(train_feature_data, test_feature_data)

    feature2id = {"<PAD>": 0}
    id2feature = {0: "<PAD>"}
    index = 1

    for key in feature2count:
        feature2id[key] = index
        id2feature[index] = key
        index += 1

    return train_feature_data, test_feature_data, feature2count, feature2id, id2feature
