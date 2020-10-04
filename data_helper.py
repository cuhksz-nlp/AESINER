import argparse
import re

from tqdm import tqdm
import os
from os import path
from collections import defaultdict
from math import log

from corenlp import StanfordCoreNLP
from nltk.tree import Tree
import json
from random import randint


FULL_MODEL = './stanford-corenlp-full-2018-10-05'
punctuation = ['。', '，', '、', '：', '？', '！', '（', '）', '“', '”', '【', '】']

chunk_pos = ['NP', 'PP', 'VP', 'ADVP', 'SBAR', 'ADJP', 'PRT', 'INTJ', 'CONJP', 'LST']


class Find_Words:
    def __init__(self, min_count=10, max_count=10000000, min_pmi=0):
        self.min_count = min_count
        self.min_pmi = min_pmi
        self.chars, self.pairs = defaultdict(int), defaultdict(int)
        self.total = 0.
        self.max_count = max_count

    def text_filter(self, texts):
        for a in tqdm(texts):
            for t in re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', a):
                if t:
                    yield t

    def count(self, texts):
        mi_list = []
        for text in self.text_filter(texts):
            self.chars[text[0]] += 1
            for i in range(len(text)-1):
                self.chars[text[i+1]] += 1
                self.pairs[text[i:i+2]] += 1
                self.total += 1
        self.chars = {i:j for i,j in self.chars.items() if 100 * self.max_count > j > self.min_count}
        self.pairs = {i:j for i,j in self.pairs.items() if self.max_count > j > self.min_count}
        self.strong_segments = set()
        for i,j in self.pairs.items():
            if i[0] in self.chars and i[1] in self.chars:
                mi = log(self.total*j/(self.chars[i[0]]*self.chars[i[1]]))
                mi_list.append(mi)
                if mi >= self.min_pmi:
                    self.strong_segments.add(i)
        print('min mi: %.4f' % min(mi_list))
        print('max mi: %.4f' % max(mi_list))
        print('remaining: %d / %d (%.4f)' % (len(self.strong_segments), len(mi_list), len(self.strong_segments)/len(mi_list)))

    def find_words(self, texts, n):
        self.words = defaultdict(int)
        for text in self.text_filter(texts):
            s = text[0]
            for i in range(len(text)-1):
                if text[i:i+2] in self.strong_segments:
                    s += text[i+1]
                else:
                    self.words[s] += 1
                    s = text[i+1]
        self.words = {i:j for i,j in self.words.items() if j > self.min_count and n+1 > len(i) > 1}


def read_txt(file_path):
    sentence_list = []
    label_list = []
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentence = []
        labels = []
        for line in lines:
            line = line.strip()
            if line == '':
                if len(sentence) > 0:
                    sentence_list.append(sentence)
                    label_list.append(labels)
                    sentence = []
                    labels = []
                continue
            items = line.split(" ")
            character = items[0]
            label = items[-1]
            sentence.append(character)
            labels.append(label)

    return sentence_list, label_list


def get_word2id(data_dir):
    word2id_path = path.join(data_dir, 'word2id.json')
    word2count_path = path.join(data_dir, 'word2count.json')
    word2id = {'<PAD>': 0}
    word2count = {}
    word = ''
    index = 1
    with open(path.join(data_dir, "train.txt"), 'r', encoding='utf8') as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split('\t')
            character = splits[0]
            word += character
            if word not in word2id:
                word2id[word] = index
                word2count[word] = 1
                index += 1
            else:
                word2count[word] += 1
            word = ''
    with open(word2id_path, 'w', encoding='utf8') as f:
        json.dump(word2id, f, ensure_ascii=False)
        f.write('\n')
    with open(path.join(data_dir, 'word2id'), 'w', encoding='utf8') as f:
        for w, v in word2id.items():
            f.write('%s\t%d\n' % (w, v))

    with open(word2count_path, 'w', encoding='utf8') as f:
        json.dump(word2count, f, ensure_ascii=False)
        f.write('\n')
    with open(path.join(data_dir, 'word2count'), 'w', encoding='utf8') as f:
        for w, v in word2count.items():
            f.write('%s\t%d\n' % (w, v))


def change(char):
    if "(" in char:
        char = char.replace("(", "-LRB-")
    if ")" in char:
        char = char.replace(")", "-RRB-")
    return char


def request_features_from_stanford(data_dir, flag):
    all_sentences, _ = read_txt(path.join(data_dir, flag + '.txt'))
    sentences_str = []
    for sentence in all_sentences:
        sentence = [change(i) for i in sentence]
        # if sentence[-1] == '·':
        #     sentence[-1] = '.'
        sentences_str.append(' '.join(sentence))


    all_data = []
    with StanfordCoreNLP(FULL_MODEL, lang='en', port=randint(38400, 38596)) as nlp:
        for sentence in tqdm(sentences_str):

            props = {
                'timeout': '5000000',
                'annotators': 'pos, parse, depparse',
                'tokenize.whitespace': 'true',
                'ssplit.eolonly': 'true',
                'pipelineLanguage': 'en',
                'outputFormat': 'json'}
            results = nlp.annotate(sentence, properties=props)
            # results = nlp.request(annotators='deparser', data=sentence)
            # results = nlp.request(annotators='pos', data=sentence)
            # result = results['sentences'][0]


            all_data.append(results)
    # assert len(all_data) == len(sentences_str)
    with open(path.join(data_dir, flag + '.stanford.json'), 'w', encoding='utf8') as f:
        for data in all_data:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

def getlabels(data_dir):
    _, train_labels = read_txt(path.join(data_dir, 'train.txt'))
    _, test_labels = read_txt(path.join(data_dir, 'test.txt'))
    all_labels = train_labels + test_labels
    label2id = defaultdict(int)
    for label_list in all_labels:
        for label in label_list:
            label2id[label] = 0
    with open(path.join(data_dir, 'label2id'), 'w', encoding='utf8') as f:
        for key in label2id.keys():
            f.write(key)
            f.write('\n')


class stanford_feature_processor:

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
            line = f.readline()
            return json.loads(line)

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
                        sentence_feature[index + i]['chunk_tags'].append({'chunk_tag': node, 'height': 0,
                                                                          'range': [index, index + len(leaves)-1]})
                        for chunk_tag in sentence_feature[index + i]['chunk_tags']:
                            chunk_tag['height'] += 1
                for token in sentence_feature:
                    if 'chunk_tags' not in token:
                        token['chunk_tags'] = [{'chunk_tag': 'ROOT', 'height': 1, 'range': [0, len(sentence_feature)-1]}]

            all_feature_data.append(sentence_feature)
        return all_feature_data


def oov_stat(data_dir):
    oov_count = 0
    word_count = 0
    word = ''
    char_count = 0
    sentence_num = 0
    oov_dict = {}
    char_dict = {}
    word_dict = {}

    with open(path.join(data_dir, 'word2id.json'), 'r', encoding='utf8') as f:
        word2id = json.loads(f.readline())

    with open(path.join(data_dir, "test.txt"), 'r', encoding='utf8') as f:
        insentence = False
        sentence_len = []
        slen = 0
        long_num = 0
        for line in tqdm(f.readlines()):
            line = line.strip()
            if len(line) == 0:
                if insentence:
                    sentence_num += 1
                    sentence_len.append(slen)
                    if slen > 150:
                        long_num += 1
                    slen = 0
                    insentence = False
                continue
            insentence = True
            slen += 1
            splits = line.split('\t')
            character = splits[0]
            label = splits[-1][0]
            word += character
            char_count += 1
            char_dict[character] = 0
            if label in ['S', 'E']:
                word_count += 1
                word_dict[word] = 0
                if word not in word2id:
                    oov_dict[word] = 0
                    oov_count += 1
                word = ''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .txt files (or other data files) for the task.")

    args = parser.parse_args()
    base_min_freq = 1
    av_threshold = 2

    min_freq = base_min_freq

    print('min freq: %d' % min_freq)

    data_dir =args.dataset

    print(data_dir)

    getlabels(data_dir)

    get_word2id(data_dir)

    if os.path.exists(path.join(data_dir, 'train' + '.txt')):
        request_features_from_stanford(data_dir, 'train')
    if os.path.exists(path.join(data_dir, 'test' + '.txt')):
        request_features_from_stanford(data_dir, 'test')
    if os.path.exists(path.join(data_dir, 'dev' + '.txt')):
        request_features_from_stanford(data_dir, 'dev')

    sfp = stanford_feature_processor(data_dir)
    sfp._pre_processing()
    sfp.feature_stat()
