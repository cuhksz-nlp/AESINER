"""
.. todo::
    doc
"""
import numpy as np
import torch
from torch import nn as nn

from ..core.vocabulary import Vocabulary

__all__ = [
    'get_embeddings'
]


def _construct_char_vocab_from_vocab(vocab: Vocabulary, min_freq: int = 1, include_word_start_end=True):
    """
    """
    char_vocab = Vocabulary(min_freq=min_freq)
    for word, index in vocab:
        if not vocab._is_word_no_create_entry(word):
            char_vocab.add_word_lst(list(word))
    if include_word_start_end:
        char_vocab.add_word_lst(['<bow>', '<eow>'])
    return char_vocab


def get_embeddings(init_embed):
    """
    """
    if isinstance(init_embed, tuple):
        res = nn.Embedding(
            num_embeddings=init_embed[0], embedding_dim=init_embed[1])
        nn.init.uniform_(res.weight.data, a=-np.sqrt(3 / res.weight.data.size(1)),
                         b=np.sqrt(3 / res.weight.data.size(1)))
    elif isinstance(init_embed, nn.Module):
        res = init_embed
    elif isinstance(init_embed, torch.Tensor):
        res = nn.Embedding.from_pretrained(init_embed, freeze=False)
    elif isinstance(init_embed, np.ndarray):
        init_embed = torch.tensor(init_embed, dtype=torch.float32)
        res = nn.Embedding.from_pretrained(init_embed, freeze=False)
    else:
        raise TypeError(
            'invalid init_embed type: {}'.format((type(init_embed))))
    return res
