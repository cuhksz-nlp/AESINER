"""
.. todo::
    doc
"""

__all__ = [
    "StackEmbedding",
]

from typing import List

import torch
from torch import nn as nn

from .embedding import TokenEmbedding


class StackEmbedding(TokenEmbedding):
    """
    """
    
    def __init__(self, embeds: List[TokenEmbedding], word_dropout=0, dropout=0):
        """
        """
        vocabs = []
        for embed in embeds:
            if hasattr(embed, 'get_word_vocab'):
                vocabs.append(embed.get_word_vocab())
        _vocab = vocabs[0]
        for vocab in vocabs[1:]:
            assert vocab == _vocab, "All embeddings in StackEmbedding should use the same word vocabulary."
        
        super(StackEmbedding, self).__init__(_vocab, word_dropout=word_dropout, dropout=dropout)
        assert isinstance(embeds, list)
        for embed in embeds:
            assert isinstance(embed, TokenEmbedding), "Only TokenEmbedding type is supported."
        self.embeds = nn.ModuleList(embeds)
        self._embed_size = sum([embed.embed_size for embed in self.embeds])
    
    def append(self, embed: TokenEmbedding):
        """
        """
        assert isinstance(embed, TokenEmbedding)
        self._embed_size += embed.embed_size
        self.embeds.append(embed)
        return self
    
    def pop(self):
        """
        :return:
        """
        embed = self.embeds.pop()
        self._embed_size -= embed.embed_size
        return embed
    
    @property
    def embed_size(self):
        """
        :return:
        """
        return self._embed_size
    
    def forward(self, words):
        """
        """
        outputs = []
        words = self.drop_word(words)
        for embed in self.embeds:
            outputs.append(embed(words))
        outputs = self.dropout(torch.cat(outputs, dim=-1))
        return outputs
