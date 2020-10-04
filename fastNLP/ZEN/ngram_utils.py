# coding: utf-8
# Copyright 2019 Sinovation Ventures AI Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""utils for ngram for ZEN model."""

import os
import logging
from random import shuffle
import math
import torch

NGRAM_DICT_NAME = 'ngram.txt'

logger = logging.getLogger(__name__)

class ZenNgramDict(object):
    """
    Dict class to store the ngram
    """
    def __init__(self, ngram_freq_path, tokenizer, max_ngram_in_seq=128):
        """Constructs ZenNgramDict

        :param ngram_freq_path: ngrams with frequency
        """
        if os.path.isdir(ngram_freq_path):
            ngram_freq_path = os.path.join(ngram_freq_path, NGRAM_DICT_NAME)
        self.ngram_freq_path = ngram_freq_path
        self.max_ngram_in_seq = max_ngram_in_seq
        self.id_to_ngram_list = ["[pad]"]
        self.ngram_to_id_dict = {"[pad]": 0}
        self.ngram_to_freq_dict = {}

        logger.info("loading ngram frequency file {}".format(ngram_freq_path))
        with open(ngram_freq_path, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                ngram,freq = line.split(",")
                tokens = tuple(tokenizer.tokenize(ngram))
                self.ngram_to_freq_dict[ngram] = freq
                self.id_to_ngram_list.append(tokens)
                self.ngram_to_id_dict[tokens] = i + 1

    def save(self, ngram_freq_path):
        with open(ngram_freq_path, "w", encoding="utf-8") as fout:
            for ngram,freq in self.ngram_to_freq_dict.items():
                fout.write("{},{}\n".format(ngram, freq))


def convert_examples_to_features(examples, vocab, max_seq_length, tokenizer, ngram_dict):
    """Loads a data file into a list of `InputBatch`s."""

    features = []

    total_ngram_ids = []
    total_ngram_positions = []
    for (ex_index, example) in enumerate(examples.data.tolist()):
        textlist = [vocab.idx2word.get(i, 0) for i in example]
        # textlist = example.text_a.split(' ')
        tokens = []
        valid = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            for m in range(len(token)):
                if m == 0:
                    valid.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            valid.append(1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(valid) == max_seq_length

        # ----------- code for ngram BEGIN-----------
        ngram_matches = []
        #  Filter the ngram segment from 2 to 7 to check whether there is a ngram
        for p in range(2, 8):
            for q in range(0, len(tokens) - p + 1):
                character_segment = tokens[q:q + p]
                # j is the starting position of the ngram
                # i is the length of the current ngram
                character_segment = tuple(character_segment)
                if character_segment in ngram_dict.ngram_to_id_dict:
                    ngram_index = ngram_dict.ngram_to_id_dict[character_segment]
                    ngram_matches.append([ngram_index, q, p, character_segment])

        shuffle(ngram_matches)

        max_ngram_in_seq_proportion = math.ceil((len(tokens) / max_seq_length) * ngram_dict.max_ngram_in_seq)
        if len(ngram_matches) > max_ngram_in_seq_proportion:
            ngram_matches = ngram_matches[:max_ngram_in_seq_proportion]

        ngram_ids = [ngram[0] for ngram in ngram_matches]
        ngram_positions = [ngram[1] for ngram in ngram_matches]
        ngram_lengths = [ngram[2] for ngram in ngram_matches]
        ngram_tuples = [ngram[3] for ngram in ngram_matches]
        ngram_seg_ids = [0 if position < (len(tokens) + 2) else 1 for position in ngram_positions]

        import numpy as np
        ngram_mask_array = np.zeros(ngram_dict.max_ngram_in_seq, dtype=np.bool)
        ngram_mask_array[:len(ngram_ids)] = 1

        # record the masked positions
        ngram_positions_matrix = np.zeros(shape=(max_seq_length, ngram_dict.max_ngram_in_seq), dtype=np.int32)
        for i in range(len(ngram_ids)):
            ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0

        # Zero-pad up to the max ngram in seq length.
        padding = [0] * (ngram_dict.max_ngram_in_seq - len(ngram_ids))
        ngram_ids += padding
        ngram_lengths += padding
        ngram_seg_ids += padding

        total_ngram_ids.append(ngram_ids)
        total_ngram_positions.append(ngram_positions)
        # ----------- code for ngram END-----------
        print(ngram_ids)
        print(ngram_positions)

    return torch.tensor(total_ngram_ids), torch.tensor(total_ngram_positions)
