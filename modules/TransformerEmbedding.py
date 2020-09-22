

from support.embeddings import TokenEmbedding
import torch
from support import Vocabulary
import torch.nn.functional as F
from support import logger
from support.embeddings.utils import _construct_char_vocab_from_vocab, get_embeddings
from torch import nn
from .transformer import TransformerEncoder


class TransformerCharEmbed(TokenEmbedding):
    def __init__(self, vocab: Vocabulary, embed_size: int = 30, char_emb_size: int = 30, word_dropout: float = 0,
                 dropout: float = 0, pool_method: str = 'max', activation='relu',
                 min_char_freq: int = 2, requires_grad=True, include_word_start_end=True,
                 char_attn_type='adatrans', char_n_head=3, char_dim_ffn=60, char_scale=False, char_pos_embed=None,
                 char_dropout=0.15, char_after_norm=False):
        super(TransformerCharEmbed, self).__init__(vocab, word_dropout=word_dropout, dropout=dropout)

        assert char_emb_size%char_n_head == 0, "d_model should divide n_head."

        assert pool_method in ('max', 'avg')
        self.pool_method = pool_method
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

        self.char_embedding = get_embeddings((len(self.char_vocab), char_emb_size))
        self.transformer = TransformerEncoder(1, char_emb_size, char_n_head, char_dim_ffn, dropout=char_dropout, after_norm=char_after_norm,
                                              attn_type=char_attn_type, pos_embed=char_pos_embed, scale=char_scale)
        self.fc = nn.Linear(char_emb_size, embed_size)

        self._embed_size = embed_size

        self.requires_grad = requires_grad

    def forward(self, words):
        words = self.drop_word(words)
        batch_size, max_len = words.size()
        chars = self.words_to_chars_embedding[words]
        word_lengths = self.word_lengths[words]
        max_word_len = word_lengths.max()
        chars = chars[:, :, :max_word_len]
        chars_masks = chars.eq(self.char_pad_index)
        char_embeds = self.char_embedding(chars)
        char_embeds = self.dropout(char_embeds)
        reshaped_chars = char_embeds.reshape(batch_size * max_len, max_word_len, -1)

        trans_chars = self.transformer(reshaped_chars, chars_masks.eq(0).reshape(-1, max_word_len))
        trans_chars = trans_chars.reshape(batch_size, max_len, max_word_len, -1)
        trans_chars = self.activation(trans_chars)
        if self.pool_method == 'max':
            trans_chars = trans_chars.masked_fill(chars_masks.unsqueeze(-1), float('-inf'))
            chars, _ = torch.max(trans_chars, dim=-2)
        else:
            trans_chars = trans_chars.masked_fill(chars_masks.unsqueeze(-1), 0)
            chars = torch.sum(trans_chars, dim=-2) / chars_masks.eq(0).sum(dim=-1, keepdim=True).float()

        chars = self.fc(chars)

        return self.dropout(chars)