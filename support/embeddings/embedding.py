"""
"""

__all__ = [
    "Embedding",
    "TokenEmbedding"
]

from abc import abstractmethod

import torch
import torch.nn as nn

from .utils import get_embeddings


class Embedding(nn.Module):
    """
    """
    
    def __init__(self, init_embed, word_dropout=0, dropout=0.0, unk_index=None):
        """
        """
        super(Embedding, self).__init__()
        
        self.embed = get_embeddings(init_embed)
        
        self.dropout = nn.Dropout(dropout)
        if not isinstance(self.embed, TokenEmbedding):
            if hasattr(self.embed, 'embed_size'):
                self._embed_size = self.embed.embed_size
            elif hasattr(self.embed, 'embedding_dim'):
                self._embed_size = self.embed.embedding_dim
            else:
                self._embed_size = self.embed.weight.size(1)
            if word_dropout > 0 and not isinstance(unk_index, int):
                raise ValueError("When drop word is set, you need to pass in the unk_index.")
        else:
            self._embed_size = self.embed.embed_size
            unk_index = self.embed.get_word_vocab().unknown_idx
        self.unk_index = unk_index
        self.word_dropout = word_dropout
    
    def forward(self, words):
        """
        :param torch.LongTensor words: [batch, seq_len]
        :return: torch.Tensor : [batch, seq_len, embed_dim]
        """
        if self.word_dropout > 0 and self.training:
            mask = torch.ones_like(words).float() * self.word_dropout
            mask = torch.bernoulli(mask).eq(1)  # dropout_word越大，越多位置为1
            words = words.masked_fill(mask, self.unk_index)
        words = self.embed(words)
        return self.dropout(words)
    
    @property
    def num_embedding(self) -> int:
        if isinstance(self.embed, nn.Embedding):
            return self.embed.weight.size(0)
        else:
            return self.embed.num_embedding
    
    def __len__(self):
        return len(self.embed)
    
    @property
    def embed_size(self) -> int:
        return self._embed_size
    
    @property
    def embedding_dim(self) -> int:
        return self._embed_size
    
    @property
    def requires_grad(self):
        """
        :return:
        """
        if not isinstance(self.embed, TokenEmbedding):
            return self.embed.weight.requires_grad
        else:
            return self.embed.requires_grad
    
    @requires_grad.setter
    def requires_grad(self, value):
        if not isinstance(self.embed, TokenEmbedding):
            self.embed.weight.requires_grad = value
        else:
            self.embed.requires_grad = value
    
    @property
    def size(self):
        if isinstance(self.embed, TokenEmbedding):
            return self.embed.size
        else:
            return self.embed.weight.size()


class TokenEmbedding(nn.Module):
    """
    """
    def __init__(self, vocab, word_dropout=0.0, dropout=0.0):
        super(TokenEmbedding, self).__init__()
        if vocab.rebuild:
            vocab.build_vocab()
        assert vocab.padding is not None, "Vocabulary must have a padding entry."
        self._word_vocab = vocab
        self._word_pad_index = vocab.padding_idx
        if word_dropout > 0:
            assert vocab.unknown is not None, "Vocabulary must have unknown entry when you want to drop a word."
        self.word_dropout = word_dropout
        self._word_unk_index = vocab.unknown_idx
        self.dropout_layer = nn.Dropout(dropout)
    
    def drop_word(self, words):
        """
        """
        if self.word_dropout > 0 and self.training:
            mask = torch.full_like(words, fill_value=self.word_dropout, dtype=torch.float, device=words.device)
            mask = torch.bernoulli(mask).eq(1)  # dropout_word越大，越多位置为1
            pad_mask = words.ne(self._word_pad_index)
            mask = mask.__and__(pad_mask)
            words = words.masked_fill(mask, self._word_unk_index)
        return words
    
    def dropout(self, words):
        """
        """
        return self.dropout_layer(words)
    
    @property
    def requires_grad(self):
        """
        """
        requires_grads = set([param.requires_grad for param in self.parameters()])
        if len(requires_grads) == 1:
            return requires_grads.pop()
        else:
            return None
    
    @requires_grad.setter
    def requires_grad(self, value):
        for param in self.parameters():
            param.requires_grad = value
    
    def __len__(self):
        return len(self._word_vocab)
    
    @property
    def embed_size(self) -> int:
        return self._embed_size
    
    @property
    def embedding_dim(self) -> int:
        return self._embed_size
    
    @property
    def num_embedding(self) -> int:
        """
        """
        return len(self._word_vocab)
    
    def get_word_vocab(self):
        """
        """
        return self._word_vocab
    
    @property
    def size(self):
        return torch.Size(self.num_embedding, self._embed_size)
    
    @abstractmethod
    def forward(self, words):
        raise NotImplementedError
