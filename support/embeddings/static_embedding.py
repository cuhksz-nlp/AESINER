"""
.. todo::
    doc
"""

__all__ = [
    "StaticEmbedding"
]
import os
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from .embedding import TokenEmbedding
from ..core import logger
from ..core.vocabulary import Vocabulary
from ..io.file_utils import PRETRAIN_STATIC_FILES, _get_embedding_url, cached_path
from ..modules.utils import _get_file_name_base_on_postfix


class StaticEmbedding(TokenEmbedding):
    """
    """
    
    def __init__(self, vocab: Vocabulary, model_dir_or_name: str = 'en', embedding_dim=-1, requires_grad: bool = True,
                 init_method=None, lower=False, dropout=0, word_dropout=0, normalize=False, min_freq=1, **kwargs):
        """
        """
        super(StaticEmbedding, self).__init__(vocab, word_dropout=word_dropout, dropout=dropout)
        if embedding_dim > 0:
            model_dir_or_name = None
        
        if model_dir_or_name is None:
            assert embedding_dim >= 1, "The dimension of embedding should be larger than 1."
            embedding_dim = int(embedding_dim)
            model_path = None
        elif model_dir_or_name.lower() in PRETRAIN_STATIC_FILES:
            model_url = _get_embedding_url('static', model_dir_or_name.lower())
            model_path = cached_path(model_url, name='embedding')
        elif os.path.isfile(os.path.abspath(os.path.expanduser(model_dir_or_name))):
            model_path = os.path.abspath(os.path.expanduser(model_dir_or_name))
        elif os.path.isdir(os.path.abspath(os.path.expanduser(model_dir_or_name))):
            model_path = _get_file_name_base_on_postfix(os.path.abspath(os.path.expanduser(model_dir_or_name)), '.txt')
        else:
            raise ValueError(f"Cannot recognize {model_dir_or_name}.")
        
        # 根据min_freq缩小vocab
        truncate_vocab = (vocab.min_freq is None and min_freq > 1) or (vocab.min_freq and vocab.min_freq < min_freq)
        if truncate_vocab:
            truncated_vocab = deepcopy(vocab)
            truncated_vocab.min_freq = min_freq
            truncated_vocab.word2idx = None
            if lower:  # 如果有lower，将大小写的的freq需要同时考虑到
                lowered_word_count = defaultdict(int)
                for word, count in truncated_vocab.word_count.items():
                    lowered_word_count[word.lower()] += count
                for word in truncated_vocab.word_count.keys():
                    word_count = truncated_vocab.word_count[word]
                    if lowered_word_count[word.lower()] >= min_freq and word_count < min_freq:
                        truncated_vocab.add_word_lst([word] * (min_freq - word_count),
                                                     no_create_entry=truncated_vocab._is_word_no_create_entry(word))
            
            # 只限制在train里面的词语使用min_freq筛选
            if kwargs.get('only_train_min_freq', False) and model_dir_or_name is not None:
                for word in truncated_vocab.word_count.keys():
                    if truncated_vocab._is_word_no_create_entry(word) and truncated_vocab.word_count[word] < min_freq:
                        truncated_vocab.add_word_lst([word] * (min_freq - truncated_vocab.word_count[word]),
                                                     no_create_entry=True)
            truncated_vocab.build_vocab()
            truncated_words_to_words = torch.arange(len(vocab)).long()
            for word, index in vocab:
                truncated_words_to_words[index] = truncated_vocab.to_index(word)
            logger.info(f"{len(vocab) - len(truncated_vocab)} out of {len(vocab)} words have frequency less than {min_freq}.")
            vocab = truncated_vocab
        
        self.only_norm_found_vector = kwargs.get('only_norm_found_vector', False)
        # 读取embedding
        if lower:
            lowered_vocab = Vocabulary(padding=vocab.padding, unknown=vocab.unknown)
            for word, index in vocab:
                if vocab._is_word_no_create_entry(word):
                    lowered_vocab.add_word(word.lower(), no_create_entry=True)
                else:
                    lowered_vocab.add_word(word.lower())  # 先加入需要创建entry的
            logger.info(f"All word in the vocab have been lowered. There are {len(vocab)} words, {len(lowered_vocab)} "
                  f"unique lowered words.")
            if model_path:
                embedding = self._load_with_vocab(model_path, vocab=lowered_vocab, init_method=init_method)
            else:
                embedding = self._randomly_init_embed(len(vocab), embedding_dim, init_method)
                self.register_buffer('words_to_words', torch.arange(len(vocab)).long())
            if lowered_vocab.unknown:
                unknown_idx = lowered_vocab.unknown_idx
            else:
                unknown_idx = embedding.size(0) - 1  # 否则是最后一个为unknow
                self.register_buffer('words_to_words', torch.arange(len(vocab)).long())
            words_to_words = torch.full((len(vocab),), fill_value=unknown_idx).long()
            for word, index in vocab:
                if word not in lowered_vocab:
                    word = word.lower()
                    if word not in lowered_vocab and lowered_vocab._is_word_no_create_entry(word):
                        continue  # 如果不需要创建entry,已经默认unknown了
                words_to_words[index] = self.words_to_words[lowered_vocab.to_index(word)]
            self.register_buffer('words_to_words', words_to_words)
            self._word_unk_index = lowered_vocab.unknown_idx  # 替换一下unknown的index
        else:
            if model_path:
                embedding = self._load_with_vocab(model_path, vocab=vocab, init_method=init_method)
            else:
                embedding = self._randomly_init_embed(len(vocab), embedding_dim, init_method)
                self.register_buffer('words_to_words', torch.arange(len(vocab)).long())
        if not self.only_norm_found_vector and normalize:
            embedding /= (torch.norm(embedding, dim=1, keepdim=True) + 1e-12)
        
        if truncate_vocab:
            for i in range(len(truncated_words_to_words)):
                index_in_truncated_vocab = truncated_words_to_words[i]
                truncated_words_to_words[i] = self.words_to_words[index_in_truncated_vocab]
            del self.words_to_words
            self.register_buffer('words_to_words', truncated_words_to_words)
        self.embedding = nn.Embedding(num_embeddings=embedding.shape[0], embedding_dim=embedding.shape[1],
                                      padding_idx=vocab.padding_idx,
                                      max_norm=None, norm_type=2, scale_grad_by_freq=False,
                                      sparse=False, _weight=embedding)
        self._embed_size = self.embedding.weight.size(1)
        self.requires_grad = requires_grad
    
    def _randomly_init_embed(self, num_embedding, embedding_dim, init_embed=None):
        """
        """
        embed = torch.zeros(num_embedding, embedding_dim)

        if init_embed is None:
            nn.init.uniform_(embed, -np.sqrt(3 / embedding_dim), np.sqrt(3 / embedding_dim))
        else:
            init_embed(embed)

        return embed

    def _load_with_vocab(self, embed_filepath, vocab, dtype=np.float32, padding='<pad>', unknown='<unk>',
                         error='ignore', init_method=None):
        """
        """
        assert isinstance(vocab, Vocabulary), "Only support.Vocabulary is supported."
        if not os.path.exists(embed_filepath):
            raise FileNotFoundError("`{}` does not exist.".format(embed_filepath))
        with open(embed_filepath, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            parts = line.split()
            start_idx = 0
            if len(parts) == 2:
                dim = int(parts[1])
                start_idx += 1
            else:
                dim = len(parts) - 1
                f.seek(0)
            matrix = {}
            if vocab.padding:
                matrix[vocab.padding_idx] = torch.zeros(dim)
            if vocab.unknown:
                matrix[vocab.unknown_idx] = torch.zeros(dim)
            found_count = 0
            found_unknown = False
            for idx, line in enumerate(f, start_idx):
                try:
                    parts = line.strip().split()
                    word = ''.join(parts[:-dim])
                    nums = parts[-dim:]
                    # 对齐unk与pad
                    if word == padding and vocab.padding is not None:
                        word = vocab.padding
                    elif word == unknown and vocab.unknown is not None:
                        word = vocab.unknown
                        found_unknown = True
                    if word in vocab:
                        index = vocab.to_index(word)
                        matrix[index] = torch.from_numpy(np.fromstring(' '.join(nums), sep=' ', dtype=dtype, count=dim))
                        if self.only_norm_found_vector:
                            matrix[index] = matrix[index] / np.linalg.norm(matrix[index])
                        found_count += 1
                except Exception as e:
                    if error == 'ignore':
                        warnings.warn("Error occurred at the {} line.".format(idx))
                    else:
                        logger.error("Error occurred at the {} line.".format(idx))
                        raise e
            logger.info("Found {} out of {} words in the pre-training embedding.".format(found_count, len(vocab)))
            for word, index in vocab:
                if index not in matrix and not vocab._is_word_no_create_entry(word):
                    if found_unknown:  # 如果有unkonwn，用unknown初始化
                        matrix[index] = matrix[vocab.unknown_idx]
                    else:
                        matrix[index] = None
            # matrix中代表是需要建立entry的词
            vectors = self._randomly_init_embed(len(matrix), dim, init_method)
            
            if vocab.unknown is None:  # 创建一个专门的unknown
                unknown_idx = len(matrix)
                vectors = torch.cat((vectors, torch.zeros(1, dim)), dim=0).contiguous()
            else:
                unknown_idx = vocab.unknown_idx
            self.register_buffer('words_to_words', torch.full((len(vocab), ), fill_value=unknown_idx).long())
            for index, (index_in_vocab, vec) in enumerate(matrix.items()):
                if vec is not None:
                    vectors[index] = vec
                self.words_to_words[index_in_vocab] = index
            
            return vectors
    
    def forward(self, words):
        """
        """
        if hasattr(self, 'words_to_words'):
            words = self.words_to_words[words]
        words = self.drop_word(words)
        words = self.embedding(words)
        words = self.dropout(words)
        return words
