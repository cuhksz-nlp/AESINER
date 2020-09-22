"""
.. todo::
    doc
"""

__all__ = [
    "ContextualEmbedding"
]

from abc import abstractmethod

import torch

from .embedding import TokenEmbedding
from ..core import logger
from ..core.batch import DataSetIter
from ..core.dataset import DataSet
from ..core.sampler import SequentialSampler
from ..core.utils import _move_model_to_device, _get_model_device
from ..core.vocabulary import Vocabulary


class ContextualEmbedding(TokenEmbedding):
    def __init__(self, vocab: Vocabulary, word_dropout: float = 0.0, dropout: float = 0.0):
        super(ContextualEmbedding, self).__init__(vocab, word_dropout=word_dropout, dropout=dropout)
    
    def add_sentence_cache(self, *datasets, batch_size=32, device='cpu', delete_weights: bool = True):
        """
        """
        for index, dataset in enumerate(datasets):
            try:
                assert isinstance(dataset, DataSet), "Only support.DataSet object is allowed."
                assert 'words' in dataset.get_input_name(), "`words` field has to be set as input."
            except Exception as e:
                logger.error(f"Exception happens at {index} dataset.")
                raise e
        
        sent_embeds = {}
        _move_model_to_device(self, device=device)
        device = _get_model_device(self)
        pad_index = self._word_vocab.padding_idx
        logger.info("Start to calculate sentence representations.")
        with torch.no_grad():
            for index, dataset in enumerate(datasets):
                try:
                    batch = DataSetIter(dataset, batch_size=batch_size, sampler=SequentialSampler())
                    for batch_x, batch_y in batch:
                        words = batch_x['words'].to(device)
                        words_list = words.tolist()
                        seq_len = words.ne(pad_index).sum(dim=-1)
                        max_len = words.size(1)
                        # 因为有些情况可能包含CLS, SEP, 从后面往前计算比较安全。
                        seq_len_from_behind = (max_len - seq_len).tolist()
                        word_embeds = self(words).detach().cpu().numpy()
                        for b in range(words.size(0)):
                            length = seq_len_from_behind[b]
                            if length == 0:
                                sent_embeds[tuple(words_list[b][:seq_len[b]])] = word_embeds[b]
                            else:
                                sent_embeds[tuple(words_list[b][:seq_len[b]])] = word_embeds[b, :-length]
                except Exception as e:
                    logger.error(f"Exception happens at {index} dataset.")
                    raise e
        logger.info("Finish calculating sentence representations.")
        self.sent_embeds = sent_embeds
        if delete_weights:
            self._delete_model_weights()
    
    def _get_sent_reprs(self, words):
        """
        获取sentence的表示，如果有缓存，则返回缓存的值; 没有缓存则返回None

        :param words: torch.LongTensor
        :return:
        """
        if hasattr(self, 'sent_embeds'):
            words_list = words.tolist()
            seq_len = words.ne(self._word_pad_index).sum(dim=-1)
            _embeds = []
            for b in range(len(words)):
                words_i = tuple(words_list[b][:seq_len[b]])
                embed = self.sent_embeds[words_i]
                _embeds.append(embed)
            max_sent_len = max(map(len, _embeds))
            embeds = words.new_zeros(len(_embeds), max_sent_len, self.embed_size, dtype=torch.float,
                                     device=words.device)
            for i, embed in enumerate(_embeds):
                embeds[i, :len(embed)] = torch.FloatTensor(embed).to(words.device)
            return embeds
        return None
    
    @abstractmethod
    def _delete_model_weights(self):
        raise NotImplementedError
    
    def remove_sentence_cache(self):
        """
        :return:
        """
        del self.sent_embeds
