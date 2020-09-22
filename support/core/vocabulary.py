"""
"""

__all__ = [
    "Vocabulary",
    "VocabularyOption",
]

from collections import Counter
from functools import partial
from functools import wraps

from ._logger import logger
from .dataset import DataSet
from .utils import Option
from .utils import _is_iterable


class VocabularyOption(Option):
    def __init__(self,
                 max_size=None,
                 min_freq=None,
                 padding='<pad>',
                 unknown='<unk>'):
        super().__init__(
            max_size=max_size,
            min_freq=min_freq,
            padding=padding,
            unknown=unknown
        )


def _check_build_vocab(func):
    """
    """

    @wraps(func)
    def _wrapper(self, *args, **kwargs):
        if self._word2idx is None or self.rebuild is True:
            self.build_vocab()
        return func(self, *args, **kwargs)
    
    return _wrapper


def _check_build_status(func):
    """A decorator to check whether the vocabulary updates after the last build.

    """
    
    @wraps(func)  # to solve missing docstring
    def _wrapper(self, *args, **kwargs):
        if self.rebuild is False:
            self.rebuild = True
            if self.max_size is not None and len(self.word_count) >= self.max_size:
                logger.info("[Warning] Vocabulary has reached the max size {} when calling {} method. "
                            "Adding more words may cause unexpected behaviour of Vocabulary. ".format(
                    self.max_size, func.__name__))
        return func(self, *args, **kwargs)
    
    return _wrapper


class Vocabulary(object):
    """
    """
    
    def __init__(self, max_size=None, min_freq=None, padding='<pad>', unknown='<unk>'):
        """
        """
        self.max_size = max_size
        self.min_freq = min_freq
        self.word_count = Counter()
        self.unknown = unknown
        self.padding = padding
        self._word2idx = None
        self._idx2word = None
        self.rebuild = True
        self._no_create_word = Counter()

    @property
    @_check_build_vocab
    def word2idx(self):
        return self._word2idx

    @word2idx.setter
    def word2idx(self, value):
        self._word2idx = value

    @property
    @_check_build_vocab
    def idx2word(self):
        return self._idx2word

    @idx2word.setter
    def idx2word(self, value):
        self._word2idx = value

    @_check_build_status
    def update(self, word_lst, no_create_entry=False):
        """
        """
        self._add_no_create_entry(word_lst, no_create_entry)
        self.word_count.update(word_lst)
        return self
    
    @_check_build_status
    def add(self, word, no_create_entry=False):
        """
        """
        self._add_no_create_entry(word, no_create_entry)
        self.word_count[word] += 1
        return self
    
    def _add_no_create_entry(self, word, no_create_entry):
        """
        """
        if isinstance(word, str) or not _is_iterable(word):
            word = [word]
        for w in word:
            if no_create_entry and self.word_count.get(w, 0) == self._no_create_word.get(w, 0):
                self._no_create_word[w] += 1
            elif not no_create_entry and w in self._no_create_word:
                self._no_create_word.pop(w)
    
    @_check_build_status
    def add_word(self, word, no_create_entry=False):
        """
        """
        self.add(word, no_create_entry=no_create_entry)
    
    @_check_build_status
    def add_word_lst(self, word_lst, no_create_entry=False):
        """
        """
        self.update(word_lst, no_create_entry=no_create_entry)
        return self
    
    def build_vocab(self):
        """
        """
        if self._word2idx is None:
            self._word2idx = {}
            if self.padding is not None:
                self._word2idx[self.padding] = len(self._word2idx)
            if self.unknown is not None:
                self._word2idx[self.unknown] = len(self._word2idx)
        
        max_size = min(self.max_size, len(self.word_count)) if self.max_size else None
        words = self.word_count.most_common(max_size)
        if self.min_freq is not None:
            words = filter(lambda kv: kv[1] >= self.min_freq, words)
        if self._word2idx is not None:
            words = filter(lambda kv: kv[0] not in self._word2idx, words)
        start_idx = len(self._word2idx)
        self._word2idx.update({w: i + start_idx for i, (w, _) in enumerate(words)})
        self.build_reverse_vocab()
        self.rebuild = False
        return self
    
    def build_reverse_vocab(self):
        """
        """
        self._idx2word = {i: w for w, i in self._word2idx.items()}
        return self
    
    @_check_build_vocab
    def __len__(self):
        return len(self._word2idx)
    
    @_check_build_vocab
    def __contains__(self, item):
        """
        """
        return item in self._word2idx

    def has_word(self, w):
        """
        """
        return self.__contains__(w)
    
    @_check_build_vocab
    def __getitem__(self, w):
        """
        """
        if w in self._word2idx:
            return self._word2idx[w]
        if self.unknown is not None:
            return self._word2idx[self.unknown]
        else:
            raise ValueError("word `{}` not in vocabulary".format(w))
    
    @_check_build_vocab
    def index_dataset(self, *datasets, field_name, new_field_name=None):
        """
        """
        
        def index_instance(field):
            """
            """
            if isinstance(field, str) or not _is_iterable(field):
                return self.to_index(field)
            else:
                if isinstance(field[0], str) or not _is_iterable(field[0]):
                    return [self.to_index(w) for w in field]
                else:
                    if not isinstance(field[0][0], str) and _is_iterable(field[0][0]):
                        raise RuntimeError("Only support field with 2 dimensions.")
                    return [[self.to_index(c) for c in w] for w in field]

        new_field_name = new_field_name or field_name

        if type(new_field_name) == type(field_name):
            if isinstance(new_field_name, list):
                assert len(new_field_name) == len(field_name), "new_field_name should have same number elements with " \
                                                               "field_name."
            elif isinstance(new_field_name, str):
                field_name = [field_name]
                new_field_name = [new_field_name]
            else:
                raise TypeError("field_name and new_field_name can only be str or List[str].")

        for idx, dataset in enumerate(datasets):
            if isinstance(dataset, DataSet):
                try:
                    for f_n, n_f_n in zip(field_name, new_field_name):
                        dataset.apply_field(index_instance, field_name=f_n, new_field_name=n_f_n)
                except Exception as e:
                    logger.info("When processing the `{}` dataset, the following error occurred.".format(idx))
                    raise e
            else:
                raise RuntimeError("Only DataSet type is allowed.")
        return self

    @property
    def _no_create_word_length(self):
        return len(self._no_create_word)

    def from_dataset(self, *datasets, field_name, no_create_entry_dataset=None):
        """

        """
        if isinstance(field_name, str):
            field_name = [field_name]
        elif not isinstance(field_name, list):
            raise TypeError('invalid argument field_name: {}'.format(field_name))
        
        def construct_vocab(ins, no_create_entry=False):
            for fn in field_name:
                field = ins[fn]
                if isinstance(field, str) or not _is_iterable(field):
                    self.add_word(field, no_create_entry=no_create_entry)
                else:
                    if isinstance(field[0], str) or not _is_iterable(field[0]):
                        for word in field:
                            self.add_word(word, no_create_entry=no_create_entry)
                    else:
                        if not isinstance(field[0][0], str) and _is_iterable(field[0][0]):
                            raise RuntimeError("Only support field with 2 dimensions.")
                        for words in field:
                            for word in words:
                                self.add_word(word, no_create_entry=no_create_entry)
        
        for idx, dataset in enumerate(datasets):
            if isinstance(dataset, DataSet):
                try:
                    dataset.apply(construct_vocab)
                except BaseException as e:
                    logger.error("When processing the `{}` dataset, the following error occurred:".format(idx))
                    raise e
            else:
                raise TypeError("Only DataSet type is allowed.")
        
        if no_create_entry_dataset is not None:
            partial_construct_vocab = partial(construct_vocab, no_create_entry=True)
            if isinstance(no_create_entry_dataset, DataSet):
                no_create_entry_dataset.apply(partial_construct_vocab)
            elif isinstance(no_create_entry_dataset, list):
                for dataset in no_create_entry_dataset:
                    if not isinstance(dataset, DataSet):
                        raise TypeError("Only DataSet type is allowed.")
                    dataset.apply(partial_construct_vocab)
        return self
    
    def _is_word_no_create_entry(self, word):
        """
        """
        return word in self._no_create_word
    
    def to_index(self, w):
        """
        """
        return self.__getitem__(w)
    
    @property
    @_check_build_vocab
    def unknown_idx(self):
        """
        """
        if self.unknown is None:
            return None
        return self._word2idx[self.unknown]
    
    @property
    @_check_build_vocab
    def padding_idx(self):
        """
        """
        if self.padding is None:
            return None
        return self._word2idx[self.padding]
    
    @_check_build_vocab
    def to_word(self, idx):
        """
        """
        return self._idx2word[idx]
    
    def clear(self):
        """
        :return:
        """
        self.word_count.clear()
        self._word2idx = None
        self._idx2word = None
        self.rebuild = True
        self._no_create_word.clear()
        return self
    
    def __getstate__(self):
        """Use to prepare data for pickle.

        """
        len(self)  # make sure vocab has been built
        state = self.__dict__.copy()
        # no need to pickle _idx2word as it can be constructed from _word2idx
        del state['_idx2word']
        return state
    
    def __setstate__(self, state):
        """Use to restore state from pickle.

        """
        self.__dict__.update(state)
        self.build_reverse_vocab()
    
    def __repr__(self):
        return "Vocabulary({}...)".format(list(self.word_count.keys())[:5])
    
    @_check_build_vocab
    def __iter__(self):
        for word, index in self._word2idx.items():
            yield word, index
