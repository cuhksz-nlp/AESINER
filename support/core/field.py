"""
"""

__all__ = [
    "Padder",
    "AutoPadder",
    "EngChar2DPadder",
]

from abc import abstractmethod
from collections import Counter
from copy import deepcopy
from numbers import Number
from typing import Any

import numpy as np
import torch

from ._logger import logger
from .utils import _is_iterable


class SetInputOrTargetException(Exception):
    def __init__(self, msg, index=None, field_name=None):
        super().__init__(msg)
        self.msg = msg
        self.index = index
        self.field_name = field_name


class AppendToTargetOrInputException(Exception):
    def __init__(self, msg, index=None, field_name=None):
        super().__init__(msg)
        self.msg = msg
        self.index = index
        self.field_name = field_name


class FieldArray:
    def __init__(self, name, content, is_target=False, is_input=False, padder=None, ignore_type=False,
                 use_1st_ins_infer_dim_type=True):
        if len(content) == 0:
            raise RuntimeError("Empty fieldarray is not allowed.")
        _content = content
        try:
            _content = list(_content)
        except BaseException as e:
            logger.error(f"Cannot convert content(of type:{type(content)}) into list.")
            raise e
        self.name = name
        self.content = _content
        self._ignore_type = ignore_type
        self._cell_ndim = None
        self.dtype = None
        self._use_1st_ins_infer_dim_type = bool(use_1st_ins_infer_dim_type)
        self._is_input = False
        self._is_target = False
        
        if is_input:
            self.is_input = is_input
        if is_target:
            self.is_target = is_target
        
        if padder is None:
            padder = AutoPadder(pad_val=0)
        else:
            assert isinstance(padder, Padder), "padder must be of type support.Padder."
            padder = deepcopy(padder)
        self.set_padder(padder)
    
    @property
    def ignore_type(self):
        return self._ignore_type
    
    @ignore_type.setter
    def ignore_type(self, value):
        if value:
            self._cell_ndim = None
            self.dtype = None
        self._ignore_type = value
    
    @property
    def is_input(self):
        return self._is_input
    
    @is_input.setter
    def is_input(self, value):
        """
            field_array.is_input
        """
        #
        if value is True and \
                self._is_target is False and \
                self._ignore_type is False:
            self._check_dtype_and_ndim(only_check_1st_ins_dim_type=self._use_1st_ins_infer_dim_type)
        if value is False and self._is_target is False:
            self.dtype = None
            self._cell_ndim = None
        self._is_input = value
    
    @property
    def is_target(self):
        return self._is_target
    
    @is_target.setter
    def is_target(self, value):
        """
        """
        if value is True and \
                self._is_input is False and \
                self._ignore_type is False:
            self._check_dtype_and_ndim(only_check_1st_ins_dim_type=self._use_1st_ins_infer_dim_type)
        if value is False and self._is_input is False:
            self.dtype = None
            self._cell_ndim = None
        self._is_target = value
    
    def _check_dtype_and_ndim(self, only_check_1st_ins_dim_type=True):
        """
        :return:
        """
        cell_0 = self.content[0]
        index = 0
        try:
            type_0, dim_0 = _get_ele_type_and_dim(cell_0)
            if not only_check_1st_ins_dim_type:
                for cell in self.content[1:]:
                    index += 1
                    type_i, dim_i = _get_ele_type_and_dim(cell)
                    if type_i != type_0:
                        raise SetInputOrTargetException(
                            "Type:{} in index {} is different from the first element with type:{}."
                            ".".format(type_i, index, type_0))
                    if dim_0 != dim_i:
                        raise SetInputOrTargetException(
                            "Dimension:{} in index {} is different from the first element with "
                            "dimension:{}.".format(dim_i, index, dim_0))
            self._cell_ndim = dim_0
            self.dtype = type_0
        except SetInputOrTargetException as e:
            e.index = index
            raise e
    
    def append(self, val: Any):
        """
        """
        if (self._is_target or self._is_input) and self._ignore_type is False and not self._use_1st_ins_infer_dim_type:
            type_, dim_ = _get_ele_type_and_dim(val)
            if self.dtype != type_:
                raise AppendToTargetOrInputException(f"Value(type:{type_}) are of different types with "
                                                     f"previous values(type:{self.dtype}).")
            if self._cell_ndim != dim_:
                raise AppendToTargetOrInputException(f"Value(dim:{dim_}) are of different dimensions with "
                                                     f"previous values(dim:{self._cell_ndim}).")
            self.content.append(val)
        else:
            self.content.append(val)
    
    def pop(self, index):
        """
        """
        self.content.pop(index)
    
    def __getitem__(self, indices):
        return self.get(indices, pad=False)
    
    def __setitem__(self, idx, val):
        assert isinstance(idx, int)
        if (self._is_target or self._is_input) and self.ignore_type is False:  # 需要检测类型
            type_, dim_ = _get_ele_type_and_dim(val)
            if self.dtype != type_:
                raise RuntimeError(f"Value(type:{type_}) are of different types with "
                                   f"other values(type:{self.dtype}).")
            if self._cell_ndim != dim_:
                raise RuntimeError(f"Value(dim:{dim_}) are of different dimensions with "
                                   f"previous values(dim:{self._cell_ndim}).")
        self.content[idx] = val
    
    def get(self, indices, pad=True):
        """
        """
        if isinstance(indices, int):
            return self.content[indices]
        if self.is_input is False and self.is_target is False:
            raise RuntimeError("Please specify either is_input or is_target to True for {}".format(self.name))
        
        contents = [self.content[i] for i in indices]
        if self.padder is None or pad is False:
            return np.array(contents)
        else:
            return self.pad(contents)
    
    def pad(self, contents):
        return self.padder(contents, field_name=self.name, field_ele_dtype=self.dtype, dim=self._cell_ndim)
    
    def set_padder(self, padder):
        """
        """
        if padder is not None:
            assert isinstance(padder, Padder), "padder must be of type Padder."
            self.padder = deepcopy(padder)
        else:
            self.padder = None
    
    def set_pad_val(self, pad_val):
        """
        """
        if self.padder is not None:
            self.padder.set_pad_val(pad_val)
        return self
    
    def __len__(self):
        """
        """
        return len(self.content)
    
    def to(self, other):
        """
        """
        assert isinstance(other, FieldArray), "Only supports support.FieldArray type, not {}.".format(type(other))
        
        self.ignore_type = other.ignore_type
        self.is_input = other.is_input
        self.is_target = other.is_target
        self.padder = other.padder
        
        return self
    
    def split(self, sep: str = None, inplace: bool = True):
        """
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                new_contents.append(cell.split(sep))
            except Exception as e:
                logger.error(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)
    
    def int(self, inplace: bool = True):
        """
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                if isinstance(cell, list):
                    new_contents.append([int(value) for value in cell])
                else:
                    new_contents.append(int(cell))
            except Exception as e:
                logger.error(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)
    
    def float(self, inplace=True):
        """
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                if isinstance(cell, list):
                    new_contents.append([float(value) for value in cell])
                else:
                    new_contents.append(float(cell))
            except Exception as e:
                logger.error(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)
    
    def bool(self, inplace=True):
        """
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                if isinstance(cell, list):
                    new_contents.append([bool(value) for value in cell])
                else:
                    new_contents.append(bool(cell))
            except Exception as e:
                logger.error(f"Exception happens when process value in index {index}.")
                raise e
        
        return self._after_process(new_contents, inplace=inplace)
    
    def lower(self, inplace=True):
        """
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                if isinstance(cell, list):
                    new_contents.append([value.lower() for value in cell])
                else:
                    new_contents.append(cell.lower())
            except Exception as e:
                logger.error(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)
    
    def upper(self, inplace=True):
        """
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                if isinstance(cell, list):
                    new_contents.append([value.upper() for value in cell])
                else:
                    new_contents.append(cell.upper())
            except Exception as e:
                logger.error(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)
    
    def value_count(self):
        """
        """
        count = Counter()
        
        def cum(cell):
            if _is_iterable(cell) and not isinstance(cell, str):
                for cell_ in cell:
                    cum(cell_)
            else:
                count[cell] += 1
        
        for cell in self.content:
            cum(cell)
        return count
    
    def _after_process(self, new_contents, inplace):
        """
        """
        if inplace:
            self.content = new_contents
            try:
                self.is_input = self.is_input
                self.is_target = self.is_input
            except SetInputOrTargetException as e:
                logger.error("The newly generated field cannot be set as input or target.")
                raise e
            return self
        else:
            return new_contents


def _get_ele_type_and_dim(cell: Any, dim=0):
    """
    """
    if isinstance(cell, (str, Number, np.bool_)):
        if hasattr(cell, 'dtype'):
            return cell.dtype.type, dim
        return type(cell), dim
    elif isinstance(cell, list):
        dim += 1
        res = [_get_ele_type_and_dim(cell_i, dim) for cell_i in cell]
        types = set([i for i, j in res])
        dims = set([j for i, j in res])
        if len(types) > 1:
            raise SetInputOrTargetException("Mixed types detected: {}.".format(list(types)))
        elif len(types) == 0:
            raise SetInputOrTargetException("Empty value encountered.")
        if len(dims) > 1:
            raise SetInputOrTargetException("Mixed dimension detected: {}.".format(list(dims)))
        return types.pop(), dims.pop()
    elif isinstance(cell, torch.Tensor):
        return cell.dtype, cell.dim() + dim
    elif isinstance(cell, np.ndarray):
        if cell.dtype != np.dtype('O'):
            return cell.dtype.type, cell.ndim + dim
        dim += 1
        res = [_get_ele_type_and_dim(cell_i, dim) for cell_i in cell]
        types = set([i for i, j in res])
        dims = set([j for i, j in res])
        if len(types) > 1:
            raise SetInputOrTargetException("Mixed types detected: {}.".format(list(types)))
        elif len(types) == 0:
            raise SetInputOrTargetException("Empty value encountered.")
        if len(dims) > 1:
            raise SetInputOrTargetException("Mixed dimension detected: {}.".format(list(dims)))
        return types.pop(), dims.pop()
    else:  # 包含tuple, set, dict以及其它的类型
        raise SetInputOrTargetException(f"Cannot process type:{type(cell)}.")


class Padder:
    """
    """
    
    def __init__(self, pad_val=0, **kwargs):
        """
        """
        self.pad_val = pad_val
    
    def set_pad_val(self, pad_val):
        self.pad_val = pad_val

    def get_pad_val(self):
        return self.pad_val

    @abstractmethod
    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        """
        """
        raise NotImplementedError


class AutoPadder(Padder):
    """
    """
    
    def __init__(self, pad_val=0):
        super().__init__(pad_val=pad_val)
    
    def __call__(self, contents, field_name, field_ele_dtype, dim):
        if field_ele_dtype:
            if dim > 3:
                return np.array(contents)
            if isinstance(field_ele_dtype, type) and \
                    (issubclass(field_ele_dtype, np.number) or issubclass(field_ele_dtype, Number)):
                if dim == 0:
                    array = np.array(contents, dtype=field_ele_dtype)
                elif dim == 1:
                    max_len = max(map(len, contents))
                    array = np.full((len(contents), max_len), self.pad_val, dtype=field_ele_dtype)
                    for i, content_i in enumerate(contents):
                        array[i, :len(content_i)] = content_i
                elif dim == 2:
                    max_len = max(map(len, contents))
                    max_word_len = max([max([len(content_ii) for content_ii in content_i]) for
                                        content_i in contents])
                    array = np.full((len(contents), max_len, max_word_len), self.pad_val, dtype=field_ele_dtype)
                    for i, content_i in enumerate(contents):
                        for j, content_ii in enumerate(content_i):
                            array[i, j, :len(content_ii)] = content_ii
                else:
                    shape = np.shape(contents)
                    if len(shape) == 4:
                        array = np.array(contents, dtype=field_ele_dtype)
                    else:
                        raise RuntimeError(
                            f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
                return array
            elif str(field_ele_dtype).startswith('torch'):
                if dim == 0:
                    tensor = torch.tensor(contents).to(field_ele_dtype)
                elif dim == 1:
                    max_len = max(map(len, contents))
                    tensor = torch.full((len(contents), max_len), fill_value=self.pad_val, dtype=field_ele_dtype)
                    for i, content_i in enumerate(contents):
                        tensor[i, :len(content_i)] = content_i.clone().detach()
                elif dim == 2:
                    max_len = max(map(len, contents))
                    max_word_len = max([max([len(content_ii) for content_ii in content_i]) for
                                        content_i in contents])
                    tensor = torch.full((len(contents), max_len, max_word_len), fill_value=self.pad_val,
                                        dtype=field_ele_dtype)
                    for i, content_i in enumerate(contents):
                        for j, content_ii in enumerate(content_i):
                            tensor[i, j, :len(content_ii)] = content_ii.clone().detach()
                else:
                    shapes = set([np.shape(content_i) for content_i in contents])
                    if len(shapes) > 1:
                        raise RuntimeError(
                            f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
                    shape = shapes.pop()
                    if len(shape) == 3:
                        tensor = torch.full([len(contents)] + list(shape), fill_value=self.pad_val,
                                            dtype=field_ele_dtype)
                        for i, content_i in enumerate(contents):
                            tensor[i] = content_i.clone().detach().to(field_ele_dtype)
                    else:
                        raise RuntimeError(
                            f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
                return tensor
            else:
                return np.array(contents)  # 不进行任何操作
        else:
            return np.array(contents)


class EngChar2DPadder(Padder):
    """
    """
    
    def __init__(self, pad_val=0, pad_length=0):
        """
        """
        super().__init__(pad_val=pad_val)
        
        self.pad_length = pad_length
    
    def __call__(self, contents, field_name, field_ele_dtype, dim):
        """
        """
        if field_ele_dtype not in (np.int64, np.float64, int, float):
            raise TypeError('dtype of Field:{} should be np.int64 or np.float64 to do 2D padding, get {}.'.format(
                field_name, field_ele_dtype
            ))
        assert dim == 2, f"Field:{field_name} has {dim}, EngChar2DPadder only supports input with 2 dimensions."
        if self.pad_length < 1:
            max_char_length = max([max(len(char_lst) for char_lst in word_lst) for word_lst in contents])
        else:
            max_char_length = self.pad_length
        max_sent_length = max(len(word_lst) for word_lst in contents)
        batch_size = len(contents)
        dtype = type(contents[0][0][0])
        
        padded_array = np.full((batch_size, max_sent_length, max_char_length), fill_value=self.pad_val,
                               dtype=dtype)
        for b_idx, word_lst in enumerate(contents):
            for c_idx, char_lst in enumerate(word_lst):
                chars = char_lst[:max_char_length]
                padded_array[b_idx, c_idx, :len(chars)] = chars
        
        return padded_array
