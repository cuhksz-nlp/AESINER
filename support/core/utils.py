"""
"""

__all__ = [
    "cache_results",
    "seq_len_to_mask",
    "get_seq_len"
]

import _pickle
import inspect
import os
import warnings
from collections import Counter, namedtuple
import numpy as np
import torch
import torch.nn as nn
from typing import List
from ._logger import logger
from prettytable import PrettyTable
try:
    from apex import amp
except:
    amp = None

_CheckRes = namedtuple('_CheckRes', ['missing', 'unused', 'duplicated', 'required', 'all_needed',
                                     'varargs'])


class Option(dict):
    """a dict can treat keys as attributes"""

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        if key.startswith('__') and key.endswith('__'):
            raise AttributeError(key)
        self.__setitem__(key, value)

    def __delattr__(self, item):
        try:
            self.pop(item)
        except KeyError:
            raise AttributeError(item)

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)


def _prepare_cache_filepath(filepath):
    """
    """
    _cache_filepath = os.path.abspath(filepath)
    if os.path.isdir(_cache_filepath):
        raise RuntimeError("The cache_file_path must be a file, not a directory.")
    cache_dir = os.path.dirname(_cache_filepath)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)


def cache_results(_cache_fp, _refresh=False, _verbose=1):
    """
    """

    def wrapper_(func):
        signature = inspect.signature(func)
        for key, _ in signature.parameters.items():
            if key in ('_cache_fp', '_refresh', '_verbose'):
                raise RuntimeError("The function decorated by cache_results cannot have keyword `{}`.".format(key))

        def wrapper(*args, **kwargs):
            if '_cache_fp' in kwargs:
                cache_filepath = kwargs.pop('_cache_fp')
                assert isinstance(cache_filepath, str), "_cache_fp can only be str."
            else:
                cache_filepath = _cache_fp
            if '_refresh' in kwargs:
                refresh = kwargs.pop('_refresh')
                assert isinstance(refresh, bool), "_refresh can only be bool."
            else:
                refresh = _refresh
            if '_verbose' in kwargs:
                verbose = kwargs.pop('_verbose')
                assert isinstance(verbose, int), "_verbose can only be integer."
            else:
                verbose = _verbose
            refresh_flag = True

            if cache_filepath is not None and refresh is False:
                # load data
                if os.path.exists(cache_filepath):
                    with open(cache_filepath, 'rb') as f:
                        results = _pickle.load(f)
                    if verbose == 1:
                        logger.info("Read cache from {}.".format(cache_filepath))
                    refresh_flag = False

            if refresh_flag:
                results = func(*args, **kwargs)
                if cache_filepath is not None:
                    if results is None:
                        raise RuntimeError("The return value is None. Delete the decorator.")
                    _prepare_cache_filepath(cache_filepath)
                    with open(cache_filepath, 'wb') as f:
                        _pickle.dump(results, f)
                    logger.info("Save cache to {}.".format(cache_filepath))

            return results

        return wrapper

    return wrapper_


def _save_model(model, model_name, save_dir, only_param=False):
    """
    """
    model_path = os.path.join(save_dir, model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if isinstance(model, nn.DataParallel):
        model = model.module
    if only_param:
        state_dict = model.state_dict()
        for key in state_dict:
            state_dict[key] = state_dict[key].cpu()
        torch.save(state_dict, model_path)
    else:
        _model_device = _get_model_device(model)
        model.cpu()
        torch.save(model, model_path)
        model.to(_model_device)


def _move_model_to_device(model, device):
    """
    """

    if device is None:
        if isinstance(model, torch.nn.DataParallel):
            model.cuda()
        return model
    else:
        if not torch.cuda.is_available() and (
                device != 'cpu' or (isinstance(device, torch.device) and device.type != 'cpu')):
            raise ValueError("There is no usable gpu. set `device` as `cpu` or `None`.")

    if isinstance(model, torch.nn.DataParallel):
        raise RuntimeError("When model is `torch.nn.DataParallel`, the device has to be `None`.")

    if isinstance(device, int):
        assert device > -1, "device can only be non-negative integer"
        assert torch.cuda.device_count() > device, "Only has {} gpus, cannot use device {}.".format(
            torch.cuda.device_count(),
            device)
        device = torch.device('cuda:{}'.format(device))
    elif isinstance(device, str):
        device = torch.device(device)
        if device.type == 'cuda' and device.index is not None:
            assert device.index < torch.cuda.device_count(), "Only has {} gpus, cannot use device cuda:{}.".format(
                torch.cuda.device_count(),
                device)
    elif isinstance(device, torch.device):
        if device.type == 'cuda' and device.index is not None:
            assert device.index < torch.cuda.device_count(), "Only has {} gpus, cannot use device cuda:{}.".format(
                torch.cuda.device_count(),
                device)
    elif isinstance(device, list):
        types = set([type(d) for d in device])
        assert len(types) == 1, "Mixed type in device, only `int` allowed."
        assert list(types)[0] == int, "Only int supported for multiple devices."
        assert len(set(device)) == len(device), "Duplicated device id found in device."
        for d in device:
            assert d > -1, "Only non-negative device id allowed."
        if len(device) > 1:
            output_device = device[0]
            model = nn.DataParallel(model, device_ids=device, output_device=output_device)
        device = torch.device(device[0])
    else:
        raise TypeError("Unsupported device type.")
    model = model.to(device)
    return model


def _get_model_device(model):
    """
    """
    assert isinstance(model, nn.Module)

    parameters = list(model.parameters())
    if len(parameters) == 0:
        return None
    else:
        return parameters[0].device


def _build_args(func, **kwargs):
    """
    """
    spect = inspect.getfullargspec(func)
    if spect.varkw is not None:
        return kwargs
    needed_args = set(spect.args)
    defaults = []
    if spect.defaults is not None:
        defaults = [arg for arg in spect.defaults]
    start_idx = len(spect.args) - len(defaults)
    output = {name: default for name, default in zip(spect.args[start_idx:], defaults)}
    output.update({name: val for name, val in kwargs.items() if name in needed_args})
    return output


def _map_args(maps: dict, **kwargs):
    output = {}
    for name, val in kwargs.items():
        if name in maps:
            assert isinstance(maps[name], str)
            output.update({maps[name]: val})
        else:
            output.update({name: val})
    for keys in maps.keys():
        if keys not in output.keys():
            pass
    return output


def _get_arg_list(func):
    assert callable(func)
    spect = inspect.getfullargspec(func)
    if spect.defaults is not None:
        args = spect.args[: -len(spect.defaults)]
        defaults = spect.args[-len(spect.defaults):]
        defaults_val = spect.defaults
    else:
        args = spect.args
        defaults = None
        defaults_val = None
    varargs = spect.varargs
    kwargs = spect.varkw
    return args, defaults, defaults_val, varargs, kwargs


# check args
def _check_arg_dict_list(func, args):
    if isinstance(args, dict):
        arg_dict_list = [args]
    else:
        arg_dict_list = args
    assert callable(func) and isinstance(arg_dict_list, (list, tuple))
    assert len(arg_dict_list) > 0 and isinstance(arg_dict_list[0], dict)
    spect = inspect.getfullargspec(func)
    all_args = set([arg for arg in spect.args if arg != 'self'])
    defaults = []
    if spect.defaults is not None:
        defaults = [arg for arg in spect.defaults]
    start_idx = len(spect.args) - len(defaults)
    default_args = set(spect.args[start_idx:])
    require_args = all_args - default_args
    input_arg_count = Counter()
    for arg_dict in arg_dict_list:
        input_arg_count.update(arg_dict.keys())
    duplicated = [name for name, val in input_arg_count.items() if val > 1]
    input_args = set(input_arg_count.keys())
    missing = list(require_args - input_args)
    unused = list(input_args - all_args)
    varargs = [] if not spect.varargs else [spect.varargs]
    return _CheckRes(missing=missing,
                     unused=unused,
                     duplicated=duplicated,
                     required=list(require_args),
                     all_needed=list(all_args),
                     varargs=varargs)


def _get_func_signature(func):
    """
    """
    if inspect.ismethod(func):
        class_name = func.__self__.__class__.__name__
        signature = inspect.signature(func)
        signature_str = str(signature)
        if len(signature_str) > 2:
            _self = '(self, '
        else:
            _self = '(self'
        signature_str = class_name + '.' + func.__name__ + _self + signature_str[1:]
        return signature_str
    elif inspect.isfunction(func):
        signature = inspect.signature(func)
        signature_str = str(signature)
        signature_str = func.__name__ + signature_str
        return signature_str


def _is_function_or_method(func):
    """
    """
    if not inspect.ismethod(func) and not inspect.isfunction(func):
        return False
    return True


def _check_function_or_method(func):
    if not _is_function_or_method(func):
        raise TypeError(f"{type(func)} is not a method or function.")


def _move_dict_value_to_device(*args, device: torch.device, non_blocking=False):
    """
    """
    if not torch.cuda.is_available():
        return

    if not isinstance(device, torch.device):
        raise TypeError(f"device must be `torch.device`, got `{type(device)}`")

    for arg in args:
        if isinstance(arg, dict):
            for key, value in arg.items():
                if isinstance(value, torch.Tensor):
                    arg[key] = value.to(device, non_blocking=non_blocking)
        else:
            raise TypeError("Only support `dict` type right now.")


class _CheckError(Exception):
    """
    """

    def __init__(self, check_res: _CheckRes, func_signature: str):
        errs = [f'Problems occurred when calling `{func_signature}`']

        if check_res.varargs:
            errs.append(f"\tvarargs: {check_res.varargs}(Does not support pass positional arguments, please delete it)")
        if check_res.missing:
            errs.append(f"\tmissing param: {check_res.missing}")
        if check_res.duplicated:
            errs.append(f"\tduplicated param: {check_res.duplicated}")
        if check_res.unused:
            errs.append(f"\tunused param: {check_res.unused}")

        Exception.__init__(self, '\n'.join(errs))

        self.check_res = check_res
        self.func_signature = func_signature


IGNORE_CHECK_LEVEL = 0
WARNING_CHECK_LEVEL = 1
STRICT_CHECK_LEVEL = 2


def _check_loss_evaluate(prev_func_signature: str, func_signature: str, check_res: _CheckRes,
                         pred_dict: dict, target_dict: dict, dataset, check_level=0):
    errs = []
    unuseds = []
    _unused_field = []
    _unused_param = []
    suggestions = []

    if check_res.unused:
        for _unused in check_res.unused:
            if _unused in target_dict:
                _unused_field.append(_unused)
            else:
                _unused_param.append(_unused)
        if _unused_field:
            unuseds.append(f"\tunused field: {_unused_field}")
        if _unused_param:
            unuseds.append(f"\tunused param: {_unused_param}")  # output from predict or forward

    module_name = func_signature.split('.')[0]
    if check_res.missing:
        errs.append(f"\tmissing param: {check_res.missing}")
        import re
        mapped_missing = []
        unmapped_missing = []
        input_func_map = {}
        for _miss_ in check_res.missing:
            _miss = _miss_.split('(')[0]
            matches = re.findall("(?<=`)[a-zA-Z0-9]*?(?=`)", _miss_)
            if len(matches) == 2:
                fun_arg, module_name = matches
                input_func_map[_miss] = fun_arg
                if fun_arg == _miss:
                    unmapped_missing.append(_miss)
                else:
                    mapped_missing.append(_miss)
            else:
                unmapped_missing.append(_miss)

        for _miss in mapped_missing + unmapped_missing:
            if _miss in dataset:
                suggestions.append(f"Set `{_miss}` as target.")
            else:
                _tmp = ''
                if check_res.unused:
                    _tmp = f"Check key assignment for `{input_func_map.get(_miss,_miss)}` when initialize {module_name}."
                if _tmp:
                    _tmp += f' Or provide `{_miss}` in DataSet or output of {prev_func_signature}.'
                else:
                    _tmp = f'Provide `{_miss}` in DataSet or output of {prev_func_signature}.'
                suggestions.append(_tmp)

    if check_res.duplicated:
        errs.append(f"\tduplicated param: {check_res.duplicated}.")
        suggestions.append(f"Delete {check_res.duplicated} in the output of "
                           f"{prev_func_signature} or do not set {check_res.duplicated} as targets. ")

    if len(errs) > 0:
        errs.extend(unuseds)
    elif check_level == STRICT_CHECK_LEVEL:
        errs.extend(unuseds)

    if len(errs) > 0:
        errs.insert(0, f'Problems occurred when calling {func_signature}')
        sugg_str = ""
        if len(suggestions) > 1:
            for idx, sugg in enumerate(suggestions):
                if idx > 0:
                    sugg_str += '\t\t\t'
                sugg_str += f'({idx + 1}). {sugg}\n'
            sugg_str = sugg_str[:-1]
        else:
            sugg_str += suggestions[0]
        errs.append(f'\ttarget field: {list(target_dict.keys())}')
        errs.append(f'\tparam from {prev_func_signature}: {list(pred_dict.keys())}')
        err_str = '\n' + '\n'.join(errs) + '\n\tSuggestion: ' + sugg_str
        raise NameError(err_str)
    if check_res.unused:
        if check_level == WARNING_CHECK_LEVEL:
            if not module_name:
                module_name = func_signature.split('.')[0]
            _unused_warn = f'{check_res.unused} is not used by {module_name}.'
            warnings.warn(message=_unused_warn)


def _check_forward_error(forward_func, batch_x, dataset, check_level):
    check_res = _check_arg_dict_list(forward_func, batch_x)
    func_signature = _get_func_signature(forward_func)

    errs = []
    suggestions = []
    _unused = []

    if check_res.missing:
        errs.append(f"\tmissing param: {check_res.missing}")
        _miss_in_dataset = []
        _miss_out_dataset = []
        for _miss in check_res.missing:
            if _miss in dataset:
                _miss_in_dataset.append(_miss)
            else:
                _miss_out_dataset.append(_miss)
        if _miss_in_dataset:
            suggestions.append(f"You might need to set {_miss_in_dataset} as input. ")
        if _miss_out_dataset:
            _tmp = f"You need to provide {_miss_out_dataset} in DataSet and set it as input. "

            suggestions.append(_tmp)

    if check_res.unused:
        _unused = [f"\tunused field: {check_res.unused}"]
        if len(errs) > 0:
            errs.extend(_unused)
        elif check_level == STRICT_CHECK_LEVEL:
            errs.extend(_unused)

    if len(errs) > 0:
        errs.insert(0, f'Problems occurred when calling {func_signature}')
        sugg_str = ""
        if len(suggestions) > 1:
            for idx, sugg in enumerate(suggestions):
                sugg_str += f'({idx + 1}). {sugg}'
        else:
            sugg_str += suggestions[0]
        err_str = '\n' + '\n'.join(errs) + '\n\tSuggestion: ' + sugg_str
        raise NameError(err_str)
    if _unused:
        if check_level == WARNING_CHECK_LEVEL:
            _unused_warn = _unused[0] + f' in {func_signature}.'
            warnings.warn(message=_unused_warn)


def seq_len_to_mask(seq_len, max_len=None):
    """
    """
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask


class _pseudo_tqdm:
    """
    """

    def __init__(self, **kwargs):
        self.logger = logger

    def write(self, info):
        self.logger.info(info)

    def set_postfix_str(self, info):
        self.logger.info(info)

    def __getattr__(self, item):
        def pass_func(*args, **kwargs):
            pass

        return pass_func

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self


def iob2(tags: List[str]) -> List[str]:
    """
    """
    for i, tag in enumerate(tags):
        if tag == "O":
            continue
        split = tag.split("-")
        if len(split) != 2 or split[0] not in ["I", "B"]:
            raise TypeError("The encoding schema is not a valid IOB type.")
        if split[0] == "B":
            continue
        elif i == 0 or tags[i - 1] == "O":  # conversion IOB1 to IOB2
            tags[i] = "B" + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = "B" + tag[1:]
    return tags


def iob2bioes(tags: List[str]) -> List[str]:
    """
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        else:
            split = tag.split('-')[0]
            if split == 'B':
                if i + 1 != len(tags) and tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('B-', 'S-'))
            elif split == 'I':
                if i + 1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('I-', 'E-'))
            else:
                raise TypeError("Invalid IOB format.")
    return new_tags


def _is_iterable(value):
    try:
        iter(value)
        return True
    except BaseException as e:
        return False


def get_seq_len(words, pad_value=0):
    """
    """
    mask = words.ne(pad_value)
    return mask.sum(dim=-1)


def pretty_table_printer(dataset_or_ins) -> PrettyTable:
    """
    """
    x = PrettyTable()
    try:
        sz = os.get_terminal_size()
        column = sz.columns
        row = sz.lines
    except OSError:
        column = 144
        row = 11

    if type(dataset_or_ins).__name__ == "DataSet":
        x.field_names = list(dataset_or_ins.field_arrays.keys())
        c_size = len(x.field_names)
        for ins in dataset_or_ins:
            x.add_row([sub_column(ins[k], column, c_size, k) for k in x.field_names])
            row -= 1
            if row < 0:
                x.add_row(["..." for _ in range(c_size)])
                break
    elif type(dataset_or_ins).__name__ == "Instance":
        x.field_names = list(dataset_or_ins.fields.keys())
        c_size = len(x.field_names)
        x.add_row([sub_column(dataset_or_ins[k], column, c_size, k) for k in x.field_names])

    else:
        raise Exception("only accept  DataSet and Instance")
    x.align = "l"

    return x


def sub_column(string: str, c: int, c_size: int, title: str) -> str:
    """
    """
    avg = max(int(c / c_size / 2), len(title))
    string = str(string)
    res = ""
    counter = 0
    for char in string:
        if ord(char) > 255:
            counter += 2
        else:
            counter += 1
        res += char
        if counter > avg:
            res = res + "..."
            break
    return res


def _check_fp16():
    if amp is None:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    if not torch.backends.cudnn.enabled:
        raise RuntimeError("Amp requires cudnn backend to be enabled.")
