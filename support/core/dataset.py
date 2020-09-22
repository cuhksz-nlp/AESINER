"""
"""
__all__ = [
    "DataSet"
]

import _pickle as pickle
from copy import deepcopy

import numpy as np
from prettytable import PrettyTable

from ._logger import logger
from .const import Const
from .field import AppendToTargetOrInputException
from .field import AutoPadder
from .field import FieldArray
from .field import SetInputOrTargetException
from .instance import Instance
from .utils import _get_func_signature
from .utils import pretty_table_printer


class DataSet(object):
    """
    """

    def __init__(self, data=None):
        """
        """
        self.field_arrays = {}
        if data is not None:
            if isinstance(data, dict):
                length_set = set()
                for key, value in data.items():
                    length_set.add(len(value))
                assert len(length_set) == 1, "Arrays must all be same length."
                for key, value in data.items():
                    self.add_field(field_name=key, fields=value)
            elif isinstance(data, list):
                for ins in data:
                    assert isinstance(ins, Instance), "Must be Instance type, not {}.".format(type(ins))
                    self.append(ins)

            else:
                raise ValueError("data only be dict or list type.")

    def __contains__(self, item):
        return item in self.field_arrays

    def __iter__(self):
        def iter_func():
            for idx in range(len(self)):
                yield self[idx]

        return iter_func()

    def _inner_iter(self):
        class Iter_ptr:
            def __init__(self, dataset, idx):
                self.dataset = dataset
                self.idx = idx

            def __getitem__(self, item):
                assert item in self.dataset.field_arrays, "no such field:{} in Instance {}".format(item, self.dataset[
                    self.idx])
                assert self.idx < len(self.dataset.field_arrays[item]), "index:{} out of range".format(self.idx)
                return self.dataset.field_arrays[item][self.idx]

            def __setitem__(self, key, value):
                raise TypeError("You cannot modify value directly.")

            def items(self):
                ins = self.dataset[self.idx]
                return ins.items()

            def __repr__(self):
                return self.dataset[self.idx].__repr__()

        def inner_iter_func():
            for idx in range(len(self)):
                yield Iter_ptr(self, idx)

        return inner_iter_func()

    def __getitem__(self, idx):
        """
        """
        if isinstance(idx, int):
            return Instance(**{name: self.field_arrays[name][idx] for name in self.field_arrays})
        elif isinstance(idx, slice):
            if idx.start is not None and (idx.start >= len(self) or idx.start <= -len(self)):
                raise RuntimeError(f"Start index {idx.start} out of range 0-{len(self) - 1}")
            data_set = DataSet()
            for field in self.field_arrays.values():
                data_set.add_field(field_name=field.name, fields=field.content[idx], padder=field.padder,
                                   is_input=field.is_input, is_target=field.is_target, ignore_type=field.ignore_type)
            return data_set
        elif isinstance(idx, str):
            if idx not in self:
                raise KeyError("No such field called {} in DataSet.".format(idx))
            return self.field_arrays[idx]
        elif isinstance(idx, list):
            dataset = DataSet()
            for i in idx:
                assert isinstance(i, int), "Only int index allowed."
                instance = self[i]
                dataset.append(instance)
            for field_name, field in self.field_arrays.items():
                dataset.field_arrays[field_name].to(field)
            return dataset
        else:
            raise KeyError("Unrecognized type {} for idx in __getitem__ method".format(type(idx)))

    def __getattr__(self, item):
        if item == "field_arrays":
            raise AttributeError
        if isinstance(item, str) and item in self.field_arrays:
            return self.field_arrays[item]

    def __setstate__(self, state):
        self.__dict__ = state

    def __getstate__(self):
        return self.__dict__

    def __len__(self):
        """Fetch the length of the dataset.
        """
        if len(self.field_arrays) == 0:
            return 0
        field = iter(self.field_arrays.values()).__next__()
        return len(field)

    def __repr__(self):
        return str(pretty_table_printer(self))

    def print_field_meta(self):
        """
        """
        if len(self.field_arrays)>0:
            field_names = ['field_names']
            is_inputs = ['is_input']
            is_targets = ['is_target']
            pad_values = ['pad_value']
            ignore_types = ['ignore_type']

            for name, field_array in self.field_arrays.items():
                field_names.append(name)
                if field_array.is_input:
                    is_inputs.append(True)
                else:
                    is_inputs.append(False)
                if field_array.is_target:
                    is_targets.append(True)
                else:
                    is_targets.append(False)

                if (field_array.is_input or field_array.is_target) and field_array.padder is not None:
                    pad_values.append(field_array.padder.get_pad_val())
                else:
                    pad_values.append(' ')

                if field_array._ignore_type:
                    ignore_types.append(True)
                elif field_array.is_input or field_array.is_target:
                    ignore_types.append(False)
                else:
                    ignore_types.append(' ')
            table = PrettyTable(field_names=field_names)
            fields = [is_inputs, is_targets, ignore_types, pad_values]
            for field in fields:
                table.add_row(field)
            logger.info(table)
            return table

    def append(self, instance):
        """
        """
        if len(self.field_arrays) == 0:
            # DataSet has no field yet
            for name, field in instance.fields.items():
                # field = field.tolist() if isinstance(field, np.ndarray) else field
                self.field_arrays[name] = FieldArray(name, [field])  # 第一个样本，必须用list包装起来
        else:
            if len(self.field_arrays) != len(instance.fields):
                raise ValueError(
                    "DataSet object has {} fields, but attempt to append an Instance object with {} fields."
                        .format(len(self.field_arrays), len(instance.fields)))
            for name, field in instance.fields.items():
                assert name in self.field_arrays
                try:
                    self.field_arrays[name].append(field)
                except AppendToTargetOrInputException as e:
                    logger.error(f"Cannot append to field:{name}.")
                    raise e

    def add_fieldarray(self, field_name, fieldarray):
        """
        """
        if not isinstance(fieldarray, FieldArray):
            raise TypeError("Only support.FieldArray supported.")
        if len(self) != len(fieldarray):
            raise RuntimeError(f"The field to add must have the same size as dataset. "
                               f"Dataset size {len(self)} != field size {len(fieldarray)}")
        self.field_arrays[field_name] = fieldarray

    def add_field(self, field_name, fields, padder=AutoPadder(), is_input=False, is_target=False, ignore_type=False):
        """
        """

        if len(self.field_arrays) != 0:
            if len(self) != len(fields):
                raise RuntimeError(f"The field to add must have the same size as dataset. "
                                   f"Dataset size {len(self)} != field size {len(fields)}")
        self.field_arrays[field_name] = FieldArray(field_name, fields, is_target=is_target, is_input=is_input,
                                                   padder=padder, ignore_type=ignore_type)

    def delete_instance(self, index):
        """
        """
        assert isinstance(index, int), "Only integer supported."
        if len(self) <= index:
            raise IndexError("{} is too large for as DataSet with {} instances.".format(index, len(self)))
        if len(self) == 1:
            self.field_arrays.clear()
        else:
            for field in self.field_arrays.values():
                field.pop(index)
        return self

    def delete_field(self, field_name):
        """
        """
        self.field_arrays.pop(field_name)
        return self

    def copy_field(self, field_name, new_field_name):
        """
        """
        if not self.has_field(field_name):
            raise KeyError(f"Field:{field_name} not found in DataSet.")
        fieldarray = deepcopy(self.get_field(field_name))
        self.add_fieldarray(field_name=new_field_name, fieldarray=fieldarray)
        return self

    def has_field(self, field_name):
        """
        """
        if isinstance(field_name, str):
            return field_name in self.field_arrays
        return False

    def get_field(self, field_name):
        """
        """
        if field_name not in self.field_arrays:
            raise KeyError("Field name {} not found in DataSet".format(field_name))
        return self.field_arrays[field_name]

    def get_all_fields(self):
        """
        """
        return self.field_arrays

    def get_field_names(self) -> list:
        """
        """
        return sorted(self.field_arrays.keys())

    def get_length(self):
        """
        """
        return len(self)

    def rename_field(self, field_name, new_field_name):
        """
        """
        if field_name in self.field_arrays:
            self.field_arrays[new_field_name] = self.field_arrays.pop(field_name)
            self.field_arrays[new_field_name].name = new_field_name
        else:
            raise KeyError("DataSet has no field named {}.".format(field_name))
        return self

    def set_target(self, *field_names, flag=True, use_1st_ins_infer_dim_type=True):
        """
        """
        assert isinstance(flag, bool), "Only bool type supported."
        for name in field_names:
            if name in self.field_arrays:
                try:
                    self.field_arrays[name]._use_1st_ins_infer_dim_type = bool(use_1st_ins_infer_dim_type)
                    self.field_arrays[name].is_target = flag
                except SetInputOrTargetException as e:
                    logger.error(f"Cannot set field:{name} as target.")
                    raise e
            else:
                raise KeyError("{} is not a valid field name.".format(name))
        return self

    def set_input(self, *field_names, flag=True, use_1st_ins_infer_dim_type=True):
        """
        """
        for name in field_names:
            if name in self.field_arrays:
                try:
                    self.field_arrays[name]._use_1st_ins_infer_dim_type = bool(use_1st_ins_infer_dim_type)
                    self.field_arrays[name].is_input = flag
                except SetInputOrTargetException as e:
                    logger.error(f"Cannot set field:{name} as input, exception happens at the {e.index} value.")
                    raise e
            else:
                raise KeyError("{} is not a valid field name.".format(name))
        return self

    def set_ignore_type(self, *field_names, flag=True):
        """
        """
        assert isinstance(flag, bool), "Only bool type supported."
        for name in field_names:
            if name in self.field_arrays:
                self.field_arrays[name].ignore_type = flag
            else:
                raise KeyError("{} is not a valid field name.".format(name))
        return self

    def set_padder(self, field_name, padder):
        """
        """
        if field_name not in self.field_arrays:
            raise KeyError("There is no field named {}.".format(field_name))
        self.field_arrays[field_name].set_padder(padder)
        return self

    def set_pad_val(self, field_name, pad_val):
        """
        """
        if field_name not in self.field_arrays:
            raise KeyError("There is no field named {}.".format(field_name))
        self.field_arrays[field_name].set_pad_val(pad_val)
        return self

    def get_input_name(self):
        """
        """
        return [name for name, field in self.field_arrays.items() if field.is_input]

    def get_target_name(self):
        """
        """
        return [name for name, field in self.field_arrays.items() if field.is_target]

    def apply_field(self, func, field_name, new_field_name=None, **kwargs):
        """
        """
        assert len(self) != 0, "Null DataSet cannot use apply_field()."
        if field_name not in self:
            raise KeyError("DataSet has no field named `{}`.".format(field_name))
        results = []
        idx = -1
        try:
            for idx, ins in enumerate(self._inner_iter()):
                results.append(func(ins[field_name]))
        except Exception as e:
            if idx != -1:
                logger.error("Exception happens at the `{}`th(from 1) instance.".format(idx + 1))
            raise e
        if not (new_field_name is None) and len(list(filter(lambda x: x is not None, results))) == 0:  # all None
            raise ValueError("{} always return None.".format(_get_func_signature(func=func)))

        if new_field_name is not None:
            self._add_apply_field(results, new_field_name, kwargs)

        return results

    def _add_apply_field(self, results, new_field_name, kwargs):
        """
        """
        extra_param = {}
        if 'is_input' in kwargs:
            extra_param['is_input'] = kwargs['is_input']
        if 'is_target' in kwargs:
            extra_param['is_target'] = kwargs['is_target']
        if 'ignore_type' in kwargs:
            extra_param['ignore_type'] = kwargs['ignore_type']
        if new_field_name in self.field_arrays:
            old_field = self.field_arrays[new_field_name]
            if 'is_input' not in extra_param:
                extra_param['is_input'] = old_field.is_input
            if 'is_target' not in extra_param:
                extra_param['is_target'] = old_field.is_target
            if 'ignore_type' not in extra_param:
                extra_param['ignore_type'] = old_field.ignore_type
            self.add_field(field_name=new_field_name, fields=results, is_input=extra_param["is_input"],
                           is_target=extra_param["is_target"], ignore_type=extra_param['ignore_type'])
        else:
            self.add_field(field_name=new_field_name, fields=results, is_input=extra_param.get("is_input", None),
                           is_target=extra_param.get("is_target", None),
                           ignore_type=extra_param.get("ignore_type", False))

    def apply(self, func, new_field_name=None, **kwargs):
        """
        """
        assert len(self) != 0, "Null DataSet cannot use apply()."
        idx = -1
        try:
            results = []
            for idx, ins in enumerate(self._inner_iter()):
                results.append(func(ins))
        except BaseException as e:
            if idx != -1:
                logger.error("Exception happens at the `{}`th instance.".format(idx))
            raise e

        if not (new_field_name is None) and len(list(filter(lambda x: x is not None, results))) == 0:  # all None
            raise ValueError("{} always return None.".format(_get_func_signature(func=func)))

        if new_field_name is not None:
            self._add_apply_field(results, new_field_name, kwargs)

        return results

    def add_seq_len(self, field_name: str, new_field_name=Const.INPUT_LEN):
        """
        """
        if self.has_field(field_name=field_name):
            self.apply_field(len, field_name, new_field_name=new_field_name)
        else:
            raise KeyError(f"Field:{field_name} not found.")
        return self

    def drop(self, func, inplace=True):
        """
        """
        if inplace:
            results = [ins for ins in self._inner_iter() if not func(ins)]
            for name, old_field in self.field_arrays.items():
                self.field_arrays[name].content = [ins[name] for ins in results]
            return self
        else:
            results = [ins for ins in self if not func(ins)]
            if len(results) != 0:
                dataset = DataSet(results)
                for field_name, field in self.field_arrays.items():
                    dataset.field_arrays[field_name].to(field)
                return dataset
            else:
                return DataSet()

    def split(self, ratio, shuffle=True):
        """
        """
        assert isinstance(ratio, float)
        assert 0 < ratio < 1
        all_indices = [_ for _ in range(len(self))]
        if shuffle:
            np.random.shuffle(all_indices)
        split = int(ratio * len(self))
        dev_indices = all_indices[:split]
        train_indices = all_indices[split:]
        dev_set = DataSet()
        train_set = DataSet()
        for idx in dev_indices:
            dev_set.append(self[idx])
        for idx in train_indices:
            train_set.append(self[idx])
        for field_name in self.field_arrays:
            train_set.field_arrays[field_name].to(self.field_arrays[field_name])
            dev_set.field_arrays[field_name].to(self.field_arrays[field_name])

        return train_set, dev_set

    def save(self, path):
        """
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        r"""
        """
        with open(path, 'rb') as f:
            d = pickle.load(f)
            assert isinstance(d, DataSet), "The object is not DataSet, but {}.".format(type(d))
        return d
