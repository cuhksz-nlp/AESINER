
from support.io import Pipe, ConllLoader
from support.io import DataBundle
from support.io.pipe.utils import _add_words_field, _indexize
from support.io.pipe.utils import iob2, iob2bioes
from support.io.pipe.utils import _add_chars_field
from support.io.utils import check_loader_paths

from support.io import Conll2003NERLoader, WNUT_17Loader
from support import Const

def word_shape(words):
    shapes = []
    for word in words:
        caps = []
        for char in word:
            caps.append(char.isupper())
        if all(caps):
            shapes.append(0)
        elif any(caps) is False:
            shapes.append(1)
        elif caps[0]:
            shapes.append(2)
        elif any(caps):
            shapes.append(3)
        else:
            shapes.append(4)
    return shapes


class Conll2003NERPipe(Pipe):
    def __init__(self, encoding_type: str = 'bio', lower: bool = False, word_shape: bool=False):
        if encoding_type == 'bio':
            self.convert_tag = iob2
        elif encoding_type == 'bioes':
            self.convert_tag = lambda words: iob2bioes(iob2(words))
        else:
            raise ValueError("encoding_type only supports `bio` and `bioes`.")
        self.lower = lower
        self.word_shape = word_shape

    def process(self, data_bundle: DataBundle) -> DataBundle:
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self.convert_tag, field_name=Const.TARGET, new_field_name=Const.TARGET)

        _add_words_field(data_bundle, lower=self.lower)

        if self.word_shape:
            data_bundle.apply_field(word_shape, field_name='raw_words', new_field_name='word_shapes')
            data_bundle.set_input('word_shapes')

        data_bundle.apply_field(lambda chars:[''.join(['0' if c.isdigit() else c for c in char]) for char in chars],
                field_name=Const.INPUT, new_field_name=Const.INPUT)

        _indexize(data_bundle)

        input_fields = [Const.TARGET, Const.INPUT, Const.INPUT_LEN]
        target_fields = [Const.TARGET, Const.INPUT_LEN]

        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_from_file(self, paths) -> DataBundle:
        data_bundle = Conll2003NERLoader().load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle


class ENNERPipe(Pipe):
    def __init__(self, encoding_type: str = 'bio', lower: bool = False, word_shape: bool=False):
        if encoding_type == 'bio':
            self.convert_tag = iob2
        elif encoding_type == 'bioes':
            self.convert_tag = lambda words: iob2bioes(iob2(words))
        else:
            raise ValueError("encoding_type only supports `bio` and `bioes`.")
        self.lower = lower
        self.word_shape = word_shape

    def process(self, data_bundle: DataBundle) -> DataBundle:
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self.convert_tag, field_name=Const.TARGET, new_field_name=Const.TARGET)

        _add_words_field(data_bundle, lower=self.lower)

        if self.word_shape:
            data_bundle.apply_field(word_shape, field_name='raw_words', new_field_name='word_shapes')
            data_bundle.set_input('word_shapes')

        data_bundle.apply_field(lambda chars:[''.join(['0' if c.isdigit() else c for c in char]) for char in chars],
                field_name=Const.INPUT, new_field_name=Const.INPUT)

        _indexize(data_bundle)

        input_fields = [Const.TARGET, Const.INPUT, Const.INPUT_LEN]
        target_fields = [Const.TARGET, Const.INPUT_LEN]

        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_from_file(self, paths) -> DataBundle:
        data_bundle = WNUT_17Loader().load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle



from support.io import OntoNotesNERLoader

class OntoNotesNERPipe(Pipe):
    def __init__(self, encoding_type: str = 'bio', lower: bool = False, word_shape: bool=False):
        if encoding_type == 'bio':
            self.convert_tag = iob2
        elif encoding_type == 'bioes':
            self.convert_tag = lambda words: iob2bioes(iob2(words))
        else:
            raise ValueError("encoding_type only supports `bio` and `bioes`.")
        self.lower = lower
        self.word_shape = word_shape

    def process(self, data_bundle: DataBundle) -> DataBundle:
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self.convert_tag, field_name=Const.TARGET, new_field_name=Const.TARGET)

        _add_words_field(data_bundle, lower=self.lower)

        if self.word_shape:
            data_bundle.apply_field(word_shape, field_name='raw_words', new_field_name='word_shapes')
            data_bundle.set_input('word_shapes')

        data_bundle.apply_field(lambda chars:[''.join(['0' if c.isdigit() else c for c in char]) for char in chars],
                field_name=Const.INPUT, new_field_name=Const.INPUT)

        _indexize(data_bundle)

        input_fields = [Const.TARGET, Const.INPUT, Const.INPUT_LEN]
        target_fields = [Const.TARGET, Const.INPUT_LEN]

        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_from_file(self, paths):
        data_bundle = OntoNotesNERLoader().load(paths)
        return self.process(data_bundle)


def bmeso2bio(tags):
    new_tags = []
    for tag in tags:
        tag = tag.lower()
        if tag.startswith('m') or tag.startswith('e'):
            tag = 'i' + tag[1:]
        if tag.startswith('s'):
            tag = 'b' + tag[1:]
        new_tags.append(tag)
    return new_tags


def bmeso2bioes(tags):
    new_tags = []
    for tag in tags:
        lowered_tag = tag.lower()
        if lowered_tag.startswith('m'):
            tag = 'i' + tag[1:]
        new_tags.append(tag)
    return new_tags


class CNNERPipe(Pipe):
    def __init__(self, bigrams=False, encoding_type='bmeso'):
        super().__init__()
        self.bigrams = bigrams
        if encoding_type=='bmeso':
            self.encoding_func = lambda x:x
        elif encoding_type=='bio':
            self.encoding_func = bmeso2bio
        elif encoding_type == 'bioes':
            self.encoding_func = bmeso2bioes
        else:
            raise RuntimeError("Only support bio, bmeso, bioes")

    def process(self, data_bundle: DataBundle):
        _add_chars_field(data_bundle, lower=False)

        data_bundle.apply_field(self.encoding_func, field_name=Const.TARGET, new_field_name=Const.TARGET)

        data_bundle.apply_field(lambda chars:[''.join(['0' if c.isdigit() else c for c in char]) for char in chars],
            field_name=Const.CHAR_INPUT, new_field_name=Const.CHAR_INPUT)

        input_field_names = [Const.CHAR_INPUT]
        if self.bigrams:
            data_bundle.apply_field(lambda chars:[c1+c2 for c1,c2 in zip(chars, chars[1:]+['<eos>'])],
                                    field_name=Const.CHAR_INPUT, new_field_name='bigrams')
            input_field_names.append('bigrams')

        _indexize(data_bundle, input_field_names=input_field_names, target_field_names=Const.TARGET)

        input_fields = [Const.TARGET, Const.INPUT_LEN] + input_field_names
        target_fields = [Const.TARGET, Const.INPUT_LEN]

        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.CHAR_INPUT)

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_from_file(self, paths):
        paths = check_loader_paths(paths)
        loader = ConllLoader(headers=['raw_chars', 'target'])
        data_bundle = loader.load(paths)
        return self.process(data_bundle)