r"""
"""

__all__ = [
    "Const"
]


class Const:
    """
    """
    INPUT = 'words'
    CHAR_INPUT = 'chars'
    INPUT_LEN = 'seq_len'
    OUTPUT = 'pred'
    TARGET = 'target'
    LOSS = 'loss'
    RAW_WORD = 'raw_words'
    RAW_CHAR = 'raw_chars'
    
    @staticmethod
    def INPUTS(i):
        i = int(i) + 1
        return Const.INPUT + str(i)
    
    @staticmethod
    def CHAR_INPUTS(i):
        i = int(i) + 1
        return Const.CHAR_INPUT + str(i)
    
    @staticmethod
    def RAW_WORDS(i):
        i = int(i) + 1
        return Const.RAW_WORD + str(i)
    
    @staticmethod
    def RAW_CHARS(i):
        i = int(i) + 1
        return Const.RAW_CHAR + str(i)
    
    @staticmethod
    def INPUT_LENS(i):
        i = int(i) + 1
        return Const.INPUT_LEN + str(i)
    
    @staticmethod
    def OUTPUTS(i):
        i = int(i) + 1
        return Const.OUTPUT + str(i)
    
    @staticmethod
    def TARGETS(i):
        i = int(i) + 1
        return Const.TARGET + str(i)
    
    @staticmethod
    def LOSSES(i):
        i = int(i) + 1
        return Const.LOSS + str(i)
