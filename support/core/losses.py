"""
"""
__all__ = [
    "LossBase",
    
    "LossFunc",
    "LossInForward",
    
    "CrossEntropyLoss",
    "BCELoss",
    "L1Loss",
    "NLLLoss",

    "CMRC2018Loss"

]

import inspect
from collections import defaultdict

import torch
import torch.nn.functional as F

from .utils import _CheckError
from .utils import _CheckRes
from .utils import _build_args
from .utils import _check_arg_dict_list
from .utils import _check_function_or_method
from .utils import _get_func_signature
from .utils import seq_len_to_mask
from ..core.const import Const


class LossBase(object):
    """
    """
    
    def __init__(self):
        self._param_map = {}
        self._checked = False

    @property
    def param_map(self):
        if len(self._param_map) == 0:
            func_spect = inspect.getfullargspec(self.get_loss)
            func_args = [arg for arg in func_spect.args if arg != 'self']
            for arg in func_args:
                self._param_map[arg] = arg
        return self._param_map

    def get_loss(self, *args, **kwargs):
        raise NotImplementedError
    
    def _init_param_map(self, key_map=None, **kwargs):
        """
        """
        value_counter = defaultdict(set)
        if key_map is not None:
            if not isinstance(key_map, dict):
                raise TypeError("key_map must be `dict`, got {}.".format(type(key_map)))
            for key, value in key_map.items():
                if value is None:
                    self._param_map[key] = key
                    continue
                if not isinstance(key, str):
                    raise TypeError(f"key in key_map must be `str`, not `{type(key)}`.")
                if not isinstance(value, str):
                    raise TypeError(f"value in key_map must be `str`, not `{type(value)}`.")
                self._param_map[key] = value
                value_counter[value].add(key)
        for key, value in kwargs.items():
            if value is None:
                self._param_map[key] = key
                continue
            if not isinstance(value, str):
                raise TypeError(f"in {key}={value}, value must be `str`, not `{type(value)}`.")
            self._param_map[key] = value
            value_counter[value].add(key)
        for value, key_set in value_counter.items():
            if len(key_set) > 1:
                raise ValueError(f"Several parameters:{key_set} are provided with one output {value}.")
        
        func_spect = inspect.getfullargspec(self.get_loss)
        func_args = [arg for arg in func_spect.args if arg != 'self']
        for func_param, input_param in self._param_map.items():
            if func_param not in func_args:
                raise NameError(
                    f"Parameter `{func_param}` is not in {_get_func_signature(self.get_loss)}. Please check the "
                    f"initialization parameters, or change its signature.")


    def __call__(self, pred_dict, target_dict, check=False):
        """
        """

        if not self._checked:
            func_spect = inspect.getfullargspec(self.get_loss)
            func_args = set([arg for arg in func_spect.args if arg != 'self'])
            for func_arg, input_arg in self._param_map.items():
                if func_arg not in func_args:
                    raise NameError(f"`{func_arg}` not in {_get_func_signature(self.get_loss)}.")
            
            for arg in func_args:
                if arg not in self._param_map:
                    self._param_map[arg] = arg
            self._evaluate_args = func_args
            self._reverse_param_map = {input_arg: func_arg for func_arg, input_arg in self._param_map.items()}

        mapped_pred_dict = {}
        mapped_target_dict = {}
        for input_arg, mapped_arg in self._reverse_param_map.items():
            if input_arg in pred_dict:
                mapped_pred_dict[mapped_arg] = pred_dict[input_arg]
            if input_arg in target_dict:
                mapped_target_dict[mapped_arg] = target_dict[input_arg]
        
        if not self._checked:
            duplicated = []
            for input_arg, mapped_arg in self._reverse_param_map.items():
                if input_arg in pred_dict and input_arg in target_dict:
                    duplicated.append(input_arg)
            check_res = _check_arg_dict_list(self.get_loss, [mapped_pred_dict, mapped_target_dict])
            # replace missing.
            missing = check_res.missing
            replaced_missing = list(missing)
            for idx, func_arg in enumerate(missing):
                # Don't delete `` in this information, nor add ``
                replaced_missing[idx] = f"{self._param_map[func_arg]}" + f"(assign to `{func_arg}` " \
                    f"in `{self.__class__.__name__}`)"
            
            check_res = _CheckRes(missing=replaced_missing,
                                  unused=check_res.unused,
                                  duplicated=duplicated,
                                  required=check_res.required,
                                  all_needed=check_res.all_needed,
                                  varargs=check_res.varargs)
            
            if check_res.missing or check_res.duplicated:
                raise _CheckError(check_res=check_res,
                                  func_signature=_get_func_signature(self.get_loss))
            self._checked = True

        refined_args = _build_args(self.get_loss, **mapped_pred_dict, **mapped_target_dict)
        
        loss = self.get_loss(**refined_args)
        self._checked = True
        
        return loss


class LossFunc(LossBase):
    """
    """
    
    def __init__(self, func, key_map=None, **kwargs):
        
        super(LossFunc, self).__init__()
        _check_function_or_method(func)
        self.get_loss = func
        if key_map is not None:
            if not isinstance(key_map, dict):
                raise RuntimeError(f"Loss error: key_map except a {type({})} but got a {type(key_map)}")
        self._init_param_map(key_map, **kwargs)


class CrossEntropyLoss(LossBase):
    """
    """
    
    def __init__(self, pred=None, target=None, seq_len=None, class_in_dim=-1, padding_idx=-100, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self.padding_idx = padding_idx
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction
        self.class_in_dim = class_in_dim
    
    def get_loss(self, pred, target, seq_len=None):
        if pred.dim() > 2:
            if self.class_in_dim == -1:
                if pred.size(1) != target.size(1):
                    pred = pred.transpose(1, 2)
            else:
                pred = pred.tranpose(-1, pred)
            pred = pred.reshape(-1, pred.size(-1))
            target = target.reshape(-1)
        if seq_len is not None and target.dim()>1:
            mask = seq_len_to_mask(seq_len, max_len=target.size(1)).reshape(-1).eq(0)
            target = target.masked_fill(mask, self.padding_idx)

        return F.cross_entropy(input=pred, target=target,
                               ignore_index=self.padding_idx, reduction=self.reduction)


class L1Loss(LossBase):
    """
    """
    
    def __init__(self, pred=None, target=None, reduction='mean'):
        super(L1Loss, self).__init__()
        self._init_param_map(pred=pred, target=target)
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction
    
    def get_loss(self, pred, target):
        return F.l1_loss(input=pred, target=target, reduction=self.reduction)


class BCELoss(LossBase):
    """
    """
    
    def __init__(self, pred=None, target=None, reduction='mean'):
        super(BCELoss, self).__init__()
        self._init_param_map(pred=pred, target=target)
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction
    
    def get_loss(self, pred, target):
        return F.binary_cross_entropy(input=pred, target=target, reduction=self.reduction)


class NLLLoss(LossBase):
    """
    """
    
    def __init__(self, pred=None, target=None, ignore_idx=-100, reduction='mean'):
        """
        """
        super(NLLLoss, self).__init__()
        self._init_param_map(pred=pred, target=target)
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction
        self.ignore_idx = ignore_idx
    
    def get_loss(self, pred, target):
        return F.nll_loss(input=pred, target=target, ignore_index=self.ignore_idx, reduction=self.reduction)


class LossInForward(LossBase):
    """
    """
    
    def __init__(self, loss_key=Const.LOSS):
        """
        """
        super().__init__()
        if not isinstance(loss_key, str):
            raise TypeError(f"Only str allowed for loss_key, got {type(loss_key)}.")
        self.loss_key = loss_key
    
    def get_loss(self, **kwargs):
        if self.loss_key not in kwargs:
            check_res = _CheckRes(
                missing=[self.loss_key + f"(assign to `{self.loss_key}` in `{self.__class__.__name__}`"],
                unused=[],
                duplicated=[],
                required=[],
                all_needed=[],
                varargs=[])
            raise _CheckError(check_res=check_res, func_signature=_get_func_signature(self.get_loss))
        return kwargs[self.loss_key]
    
    def __call__(self, pred_dict, target_dict, check=False):
        
        loss = self.get_loss(**pred_dict)
        
        if not (isinstance(loss, torch.Tensor) and len(loss.size()) == 0):
            if not isinstance(loss, torch.Tensor):
                raise TypeError(f"Loss excepted to be a torch.Tensor, got {type(loss)}")
            loss = torch.sum(loss) / (loss.view(-1)).size(0)

        return loss


class CMRC2018Loss(LossBase):
    """
    """
    def __init__(self, target_start=None, target_end=None, context_len=None, pred_start=None, pred_end=None,
                  reduction='mean'):
        super().__init__()

        assert reduction in ('mean', 'sum')

        self._init_param_map(target_start=target_start, target_end=target_end, context_len=context_len,
                             pred_start=pred_start, pred_end=pred_end)
        self.reduction = reduction

    def get_loss(self, target_start, target_end, context_len, pred_start, pred_end):
        """
        """
        batch_size, max_len = pred_end.size()
        mask = seq_len_to_mask(context_len, max_len).eq(0)

        pred_start = pred_start.masked_fill(mask, float('-inf'))
        pred_end = pred_end.masked_fill(mask, float('-inf'))

        start_loss = F.cross_entropy(pred_start, target_start, reduction='sum')
        end_loss = F.cross_entropy(pred_end, target_end, reduction='sum')

        loss = start_loss + end_loss

        if self.reduction == 'mean':
            loss = loss / batch_size

        return loss/2

def _prepare_losser(losser):
    if losser is None:
        losser = LossInForward()
        return losser
    elif isinstance(losser, LossBase):
        return losser
    else:
        raise TypeError(f"Type of loss should be `fastNLP.LossBase`, got {type(losser)}")
