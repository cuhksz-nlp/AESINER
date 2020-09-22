"""
"""

__all__ = [
    "CNNCharEmbedding",
    "LSTMCharEmbedding"
]

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import TokenEmbedding
from .static_embedding import StaticEmbedding
from .utils import _construct_char_vocab_from_vocab
from .utils import get_embeddings
from ..core import logger
from ..core.vocabulary import Vocabulary
from ..modules.encoder.lstm import LSTM


class CNNCharEmbedding(TokenEmbedding):
    """
    """

    def __init__(self, vocab: Vocabulary, embed_size: int = 50, char_emb_size: int = 50, word_dropout: float = 0,
                 dropout: float = 0, filter_nums: List[int] = (40, 30, 20), kernel_sizes: List[int] = (5, 3, 1),
                 pool_method: str = 'max', activation='relu', min_char_freq: int = 2, pre_train_char_embed: str = None,
                 requires_grad:bool=True, include_word_start_end:bool=True):
        """
        """
        super(CNNCharEmbedding, self).__init__(vocab, word_dropout=word_dropout, dropout=dropout)
        
        for kernel in kernel_sizes:
            assert kernel % 2 == 1, "Only odd kernel is allowed."
        
        assert pool_method in ('max', 'avg')
        self.pool_method = pool_method
        # activation function
        if isinstance(activation, str):
            if activation.lower() == 'relu':
                self.activation = F.relu
            elif activation.lower() == 'sigmoid':
                self.activation = F.sigmoid
            elif activation.lower() == 'tanh':
                self.activation = F.tanh
        elif activation is None:
            self.activation = lambda x: x
        elif callable(activation):
            self.activation = activation
        else:
            raise Exception(
                "Undefined activation function: choose from: [relu, tanh, sigmoid, or a callable function]")
        
        logger.info("Start constructing character vocabulary.")
        self.char_vocab = _construct_char_vocab_from_vocab(vocab, min_freq=min_char_freq,
                                                           include_word_start_end=include_word_start_end)
        self.char_pad_index = self.char_vocab.padding_idx
        logger.info(f"In total, there are {len(self.char_vocab)} distinct characters.")
        max_word_len = max(map(lambda x: len(x[0]), vocab))
        if include_word_start_end:
            max_word_len += 2
        self.register_buffer('words_to_chars_embedding', torch.full((len(vocab), max_word_len),
                                                                fill_value=self.char_pad_index, dtype=torch.long))
        self.register_buffer('word_lengths', torch.zeros(len(vocab)).long())
        for word, index in vocab:
            # if index!=vocab.padding_idx:  # 如果是pad的话，直接就为pad_value了。修改为不区分pad, 这样所有的<pad>也是同一个embed
            if include_word_start_end:
                word = ['<bow>'] + list(word) + ['<eow>']
            self.words_to_chars_embedding[index, :len(word)] = \
                torch.LongTensor([self.char_vocab.to_index(c) for c in word])
            self.word_lengths[index] = len(word)
        # self.char_embedding = nn.Embedding(len(self.char_vocab), char_emb_size)
        if pre_train_char_embed:
            self.char_embedding = StaticEmbedding(self.char_vocab, model_dir_or_name=pre_train_char_embed)
        else:
            self.char_embedding = get_embeddings((len(self.char_vocab), char_emb_size))
        
        self.convs = nn.ModuleList([nn.Conv1d(
            char_emb_size, filter_nums[i], kernel_size=kernel_sizes[i], bias=True, padding=kernel_sizes[i] // 2)
            for i in range(len(kernel_sizes))])
        self._embed_size = embed_size
        self.fc = nn.Linear(sum(filter_nums), embed_size)
        self.requires_grad = requires_grad

    def forward(self, words):
        """
        """
        words = self.drop_word(words)
        batch_size, max_len = words.size()
        chars = self.words_to_chars_embedding[words]  # batch_size x max_len x max_word_len
        word_lengths = self.word_lengths[words]  # batch_size x max_len
        max_word_len = word_lengths.max()
        chars = chars[:, :, :max_word_len]
        # 为1的地方为mask
        chars_masks = chars.eq(self.char_pad_index)  # batch_size x max_len x max_word_len 如果为0, 说明是padding的位置了
        chars = self.char_embedding(chars)  # batch_size x max_len x max_word_len x embed_size
        chars = self.dropout(chars)
        reshaped_chars = chars.reshape(batch_size * max_len, max_word_len, -1)
        reshaped_chars = reshaped_chars.transpose(1, 2)  # B' x E x M
        conv_chars = [conv(reshaped_chars).transpose(1, 2).reshape(batch_size, max_len, max_word_len, -1)
                      for conv in self.convs]
        conv_chars = torch.cat(conv_chars, dim=-1).contiguous()  # B x max_len x max_word_len x sum(filters)
        conv_chars = self.activation(conv_chars)
        if self.pool_method == 'max':
            conv_chars = conv_chars.masked_fill(chars_masks.unsqueeze(-1), float('-inf'))
            chars, _ = torch.max(conv_chars, dim=-2)  # batch_size x max_len x sum(filters)
        else:
            conv_chars = conv_chars.masked_fill(chars_masks.unsqueeze(-1), 0)
            chars = torch.sum(conv_chars, dim=-2) / chars_masks.eq(0).sum(dim=-1, keepdim=True).float()
        chars = self.fc(chars)
        return self.dropout(chars)


class LSTMCharEmbedding(TokenEmbedding):
    """
    """
    
    def __init__(self, vocab: Vocabulary, embed_size: int = 50, char_emb_size: int = 50, word_dropout: float = 0,
                 dropout: float = 0, hidden_size=50, pool_method: str = 'max', activation='relu',
                 min_char_freq: int = 2, bidirectional=True, pre_train_char_embed: str = None,
                 requires_grad:bool=True, include_word_start_end:bool=True):
        """
        """
        super(LSTMCharEmbedding, self).__init__(vocab, word_dropout=word_dropout, dropout=dropout)
        
        assert hidden_size % 2 == 0, "Only even kernel is allowed."
        
        assert pool_method in ('max', 'avg')
        self.pool_method = pool_method
        # activation function
        if isinstance(activation, str):
            if activation.lower() == 'relu':
                self.activation = F.relu
            elif activation.lower() == 'sigmoid':
                self.activation = F.sigmoid
            elif activation.lower() == 'tanh':
                self.activation = F.tanh
        elif activation is None:
            self.activation = lambda x: x
        elif callable(activation):
            self.activation = activation
        else:
            raise Exception(
                "Undefined activation function: choose from: [relu, tanh, sigmoid, or a callable function]")
        
        logger.info("Start constructing character vocabulary.")
        self.char_vocab = _construct_char_vocab_from_vocab(vocab, min_freq=min_char_freq,
                                                           include_word_start_end=include_word_start_end)
        self.char_pad_index = self.char_vocab.padding_idx
        logger.info(f"In total, there are {len(self.char_vocab)} distinct characters.")
        max_word_len = max(map(lambda x: len(x[0]), vocab))
        if include_word_start_end:
            max_word_len += 2
        self.register_buffer('words_to_chars_embedding', torch.full((len(vocab), max_word_len),
                                                                fill_value=self.char_pad_index, dtype=torch.long))
        self.register_buffer('word_lengths', torch.zeros(len(vocab)).long())
        for word, index in vocab:
            if include_word_start_end:
                word = ['<bow>'] + list(word) + ['<eow>']
            self.words_to_chars_embedding[index, :len(word)] = \
                torch.LongTensor([self.char_vocab.to_index(c) for c in word])
            self.word_lengths[index] = len(word)
        if pre_train_char_embed:
            self.char_embedding = StaticEmbedding(self.char_vocab, pre_train_char_embed)
        else:
            self.char_embedding = nn.Embedding(len(self.char_vocab), char_emb_size)
        
        self.fc = nn.Linear(hidden_size, embed_size)
        hidden_size = hidden_size // 2 if bidirectional else hidden_size
        
        self.lstm = LSTM(char_emb_size, hidden_size, bidirectional=bidirectional, batch_first=True)
        self._embed_size = embed_size
        self.bidirectional = bidirectional
        self.requires_grad = requires_grad
    
    def forward(self, words):
        """
        """
        words = self.drop_word(words)
        batch_size, max_len = words.size()
        chars = self.words_to_chars_embedding[words]  # batch_size x max_len x max_word_len
        word_lengths = self.word_lengths[words]  # batch_size x max_len
        max_word_len = word_lengths.max()
        chars = chars[:, :, :max_word_len]
        # 为mask的地方为1
        chars_masks = chars.eq(self.char_pad_index)  # batch_size x max_len x max_word_len 如果为0, 说明是padding的位置了
        chars = self.char_embedding(chars)  # batch_size x max_len x max_word_len x embed_size
        chars = self.dropout(chars)
        reshaped_chars = chars.reshape(batch_size * max_len, max_word_len, -1)
        char_seq_len = chars_masks.eq(0).sum(dim=-1).reshape(batch_size * max_len)
        lstm_chars = self.lstm(reshaped_chars, char_seq_len)[0].reshape(batch_size, max_len, max_word_len, -1)
        # B x M x M x H
        
        lstm_chars = self.activation(lstm_chars)
        if self.pool_method == 'max':
            lstm_chars = lstm_chars.masked_fill(chars_masks.unsqueeze(-1), float('-inf'))
            chars, _ = torch.max(lstm_chars, dim=-2)  # batch_size x max_len x H
        else:
            lstm_chars = lstm_chars.masked_fill(chars_masks.unsqueeze(-1), 0)
            chars = torch.sum(lstm_chars, dim=-2) / chars_masks.eq(0).sum(dim=-1, keepdim=True).float()
        
        chars = self.fc(chars)
        
        return self.dropout(chars)
