"""
"""
__all__ = [
    "MetricBase",
    "AccuracyMetric",
    "SpanFPreRecMetric",
    "CMRC2018Metric"
]

import inspect
import warnings
from abc import abstractmethod
from collections import defaultdict
from typing import Union
import re

import numpy as np
import torch

from .utils import _CheckError
from .utils import _CheckRes
from .utils import _build_args
from .utils import _check_arg_dict_list
from .utils import _get_func_signature
from .utils import seq_len_to_mask
from .vocabulary import Vocabulary


class MetricBase(object):
    """
    """

    def __init__(self):
        self._param_map = {}  # key is param in function, value is input param.
        self._checked = False
        self._metric_name = self.__class__.__name__

    @property
    def param_map(self):
        if len(self._param_map) == 0:
            func_spect = inspect.getfullargspec(self.evaluate)
            func_args = [arg for arg in func_spect.args if arg != 'self']
            for arg in func_args:
                self._param_map[arg] = arg
        return self._param_map

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_metric(self, reset=True):
        raise NotImplemented

    def set_metric_name(self, name: str):
        """

        :param str name:
        :return: self
        """
        self._metric_name = name
        return self

    def get_metric_name(self):
        """

        :return:
        """
        return self._metric_name

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

        # check consistence between signature and _param_map
        func_spect = inspect.getfullargspec(self.evaluate)
        func_args = [arg for arg in func_spect.args if arg != 'self']
        for func_param, input_param in self._param_map.items():
            if func_param not in func_args:
                raise NameError(
                    f"Parameter `{func_param}` is not in {_get_func_signature(self.evaluate)}. Please check the "
                    f"initialization parameters, or change its signature.")

    def _fast_param_map(self, pred_dict, target_dict):
        """Only used as inner function. When the pred_dict, target is unequivocal. Don't need users to pass key_map.
            such as pred_dict has one element, target_dict has one element

        :param pred_dict:
        :param target_dict:
        :return: dict, if dict is not {}, pass it to self.evaluate. Otherwise do mapping.
        """
        fast_param = {}
        if len(self._param_map) == 2 and len(pred_dict) == 1 and len(target_dict) == 1:
            fast_param['pred'] = list(pred_dict.values())[0]
            fast_param['target'] = list(target_dict.values())[0]
            return fast_param
        return fast_param

    def __call__(self, pred_dict, target_dict):
        """
        """

        fast_param = self._fast_param_map(pred_dict, target_dict)
        if fast_param:
            self.evaluate(**fast_param)
            return

        if not self._checked:
            if not callable(self.evaluate):
                raise TypeError(f"{self.__class__.__name__}.evaluate has to be callable, not {type(self.evaluate)}.")
            # 1. check consistence between signature and _param_map
            func_spect = inspect.getfullargspec(self.evaluate)
            func_args = set([arg for arg in func_spect.args if arg != 'self'])
            for func_arg, input_arg in self._param_map.items():
                if func_arg not in func_args:
                    raise NameError(f"`{func_arg}` not in {_get_func_signature(self.evaluate)}.")

            # 2. only part of the _param_map are passed, left are not
            for arg in func_args:
                if arg not in self._param_map:
                    self._param_map[arg] = arg  # This param does not need mapping.
            self._evaluate_args = func_args
            self._reverse_param_map = {input_arg: func_arg for func_arg, input_arg in self._param_map.items()}

        # need to wrap inputs in dict.
        mapped_pred_dict = {}
        mapped_target_dict = {}
        for input_arg, mapped_arg in self._reverse_param_map.items():
            if input_arg in pred_dict:
                mapped_pred_dict[mapped_arg] = pred_dict[input_arg]
            if input_arg in target_dict:
                mapped_target_dict[mapped_arg] = target_dict[input_arg]

        # missing
        if not self._checked:
            duplicated = []
            for input_arg, mapped_arg in self._reverse_param_map.items():
                if input_arg in pred_dict and input_arg in target_dict:
                    duplicated.append(input_arg)
            check_res = _check_arg_dict_list(self.evaluate, [mapped_pred_dict, mapped_target_dict])
            # only check missing.
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
                                  func_signature=_get_func_signature(self.evaluate))
            self._checked = True
        refined_args = _build_args(self.evaluate, **mapped_pred_dict, **mapped_target_dict)

        self.evaluate(**refined_args)

        return


class AccuracyMetric(MetricBase):
    """
    """

    def __init__(self, pred=None, target=None, seq_len=None):
        """
        """

        super().__init__()

        self._init_param_map(pred=pred, target=target, seq_len=seq_len)

        self.total = 0
        self.acc_count = 0

    def evaluate(self, pred, target, seq_len=None):
        """
        """
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if seq_len is not None and not isinstance(seq_len, torch.Tensor):
            raise TypeError(f"`seq_lens` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(seq_len)}.")

        if seq_len is not None and target.dim() > 1:
            max_len = target.size(1)
            masks = seq_len_to_mask(seq_len=seq_len, max_len=max_len)
        else:
            masks = None

        if pred.dim() == target.dim():
            pass
        elif pred.dim() == target.dim() + 1:
            pred = pred.argmax(dim=-1)
            if seq_len is None and target.dim() > 1:
                warnings.warn("You are not passing `seq_len` to exclude pad when calculate accuracy.")
        else:
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        target = target.to(pred)
        if masks is not None:
            self.acc_count += torch.sum(torch.eq(pred, target).masked_fill(masks.eq(0), 0)).item()
            self.total += torch.sum(masks).item()
        else:
            self.acc_count += torch.sum(torch.eq(pred, target)).item()
            self.total += np.prod(list(pred.size()))

    def get_metric(self, reset=True):
        """
        """
        evaluate_result = {'acc': round(float(self.acc_count) / (self.total + 1e-12), 6)}
        if reset:
            self.acc_count = 0
            self.total = 0
        return evaluate_result


def _bmes_tag_to_spans(tags, ignore_labels=None):
    """
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bmes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bmes_tag, label = tag[:1], tag[2:]
        if bmes_tag in ('b', 's'):
            spans.append((label, [idx, idx]))
        elif bmes_tag in ('m', 'e') and prev_bmes_tag in ('b', 'm') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        else:
            spans.append((label, [idx, idx]))
        prev_bmes_tag = bmes_tag
    return [(span[0], (span[1][0], span[1][1] + 1))
            for span in spans
            if span[0] not in ignore_labels
            ]


def _bmeso_tag_to_spans(tags, ignore_labels=None):
    """
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bmes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bmes_tag, label = tag[:1], tag[2:]
        if bmes_tag in ('b', 's'):
            spans.append((label, [idx, idx]))
        elif bmes_tag in ('m', 'e') and prev_bmes_tag in ('b', 'm') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bmes_tag == 'o':
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bmes_tag = bmes_tag
    return [(span[0], (span[1][0], span[1][1] + 1))
            for span in spans
            if span[0] not in ignore_labels
            ]


def _bioes_tag_to_spans(tags, ignore_labels=None):
    """
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bioes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bioes_tag, label = tag[:1], tag[2:]
        if bioes_tag in ('b', 's'):
            spans.append((label, [idx, idx]))
        elif bioes_tag in ('i', 'e') and prev_bioes_tag in ('b', 'i') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bioes_tag == 'o':
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bioes_tag = bioes_tag
    return [(span[0], (span[1][0], span[1][1] + 1))
            for span in spans
            if span[0] not in ignore_labels
            ]


def _bio_tag_to_spans(tags, ignore_labels=None):
    """
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bio_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bio_tag, label = tag[:1], tag[2:]
        if bio_tag == 'b':
            spans.append((label, [idx, idx]))
        elif bio_tag == 'i' and prev_bio_tag in ('b', 'i') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bio_tag == 'o':  # o tag does not count
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bio_tag = bio_tag
    return [(span[0], (span[1][0], span[1][1] + 1)) for span in spans if span[0] not in ignore_labels]


def _get_encoding_type_from_tag_vocab(tag_vocab: Union[Vocabulary, dict]) -> str:
    """
    """
    tag_set = set()
    unk_token = '<unk>'
    pad_token = '<pad>'
    if isinstance(tag_vocab, Vocabulary):
        unk_token = tag_vocab.unknown
        pad_token = tag_vocab.padding
        tag_vocab = tag_vocab.idx2word
    for idx, tag in tag_vocab.items():
        if tag in (unk_token, pad_token):
            continue
        tag = tag[:1].lower()
        tag_set.add(tag)

    bmes_tag_set = set('bmes')
    if tag_set == bmes_tag_set:
        return 'bmes'
    bio_tag_set = set('bio')
    if tag_set == bio_tag_set:
        return 'bio'
    bmeso_tag_set = set('bmeso')
    if tag_set == bmeso_tag_set:
        return 'bmeso'
    bioes_tag_set = set('bioes')
    if tag_set == bioes_tag_set:
        return 'bioes'
    raise RuntimeError("encoding_type cannot be inferred automatically. Only support "
                       "'bio', 'bmes', 'bmeso', 'bioes' type.")


def _check_tag_vocab_and_encoding_type(tag_vocab: Union[Vocabulary, dict], encoding_type: str):
    """
    """
    tag_set = set()
    unk_token = '<unk>'
    pad_token = '<pad>'
    if isinstance(tag_vocab, Vocabulary):
        unk_token = tag_vocab.unknown
        pad_token = tag_vocab.padding
        tag_vocab = tag_vocab.idx2word
    for idx, tag in tag_vocab.items():
        if tag in (unk_token, pad_token):
            continue
        tag = tag[:1].lower()
        tag_set.add(tag)

    tags = encoding_type
    for tag in tag_set:
        assert tag in tags, f"{tag} is not a valid tag in encoding type:{encoding_type}. Please check your " \
                            f"encoding_type."
        tags = tags.replace(tag, '')
    if tags:
        warnings.warn(f"Tag:{tags} in encoding type:{encoding_type} is not presented in your Vocabulary. Check your "
                      "encoding_type.")


class SpanFPreRecMetric(MetricBase):
    r"""
    """

    def __init__(self, tag_vocab, pred=None, target=None, seq_len=None, encoding_type=None, ignore_labels=None,
                 only_gross=True, f_type='micro', beta=1):
        r"""
        """

        if not isinstance(tag_vocab, Vocabulary):
            raise TypeError("tag_vocab can only be support.Vocabulary, not {}.".format(type(tag_vocab)))
        if f_type not in ('micro', 'macro'):
            raise ValueError("f_type only supports `micro` or `macro`', got {}.".format(f_type))

        if encoding_type:
            encoding_type = encoding_type.lower()
            _check_tag_vocab_and_encoding_type(tag_vocab, encoding_type)
            self.encoding_type = encoding_type
        else:
            self.encoding_type = _get_encoding_type_from_tag_vocab(tag_vocab)

        if self.encoding_type == 'bmes':
            self.tag_to_span_func = _bmes_tag_to_spans
        elif self.encoding_type == 'bio':
            self.tag_to_span_func = _bio_tag_to_spans
        elif self.encoding_type == 'bmeso':
            self.tag_to_span_func = _bmeso_tag_to_spans
        elif self.encoding_type == 'bioes':
            self.tag_to_span_func = _bioes_tag_to_spans
        else:
            raise ValueError("Only support 'bio', 'bmes', 'bmeso', 'bioes' type.")

        self.ignore_labels = ignore_labels
        self.f_type = f_type
        self.beta = beta
        self.beta_square = self.beta ** 2
        self.only_gross = only_gross

        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)

        self.tag_vocab = tag_vocab

        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)

    def evaluate(self, pred, target, seq_len):
        """
        """
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if not isinstance(seq_len, torch.Tensor):
            raise TypeError(f"`seq_lens` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(seq_len)}.")

        if pred.size() == target.size() and len(target.size()) == 2:
            pass
        elif len(pred.size()) == len(target.size()) + 1 and len(target.size()) == 2:
            num_classes = pred.size(-1)
            pred = pred.argmax(dim=-1)
            if (target >= num_classes).any():
                raise ValueError("A gold label passed to SpanBasedF1Metric contains an "
                                 "id >= {}, the number of classes.".format(num_classes))
        else:
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        batch_size = pred.size(0)
        pred = pred.tolist()
        target = target.tolist()
        for i in range(batch_size):
            pred_tags = pred[i][:int(seq_len[i])]
            gold_tags = target[i][:int(seq_len[i])]

            pred_str_tags = [self.tag_vocab.to_word(tag) for tag in pred_tags]
            gold_str_tags = [self.tag_vocab.to_word(tag) for tag in gold_tags]

            pred_spans = self.tag_to_span_func(pred_str_tags, ignore_labels=self.ignore_labels)
            gold_spans = self.tag_to_span_func(gold_str_tags, ignore_labels=self.ignore_labels)

            for span in pred_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1
            for span in gold_spans:
                self._false_negatives[span[0]] += 1

    def get_metric(self, reset=True):
        evaluate_result = {}
        if not self.only_gross or self.f_type == 'macro':
            tags = set(self._false_negatives.keys())
            tags.update(set(self._false_positives.keys()))
            tags.update(set(self._true_positives.keys()))
            f_sum = 0
            pre_sum = 0
            rec_sum = 0
            for tag in tags:
                tp = self._true_positives[tag]
                fn = self._false_negatives[tag]
                fp = self._false_positives[tag]
                f, pre, rec = self._compute_f_pre_rec(tp, fn, fp)
                f_sum += f
                pre_sum += pre
                rec_sum += rec
                if not self.only_gross and tag != '':
                    f_key = 'f-{}'.format(tag)
                    pre_key = 'pre-{}'.format(tag)
                    rec_key = 'rec-{}'.format(tag)
                    evaluate_result[f_key] = f
                    evaluate_result[pre_key] = pre
                    evaluate_result[rec_key] = rec

            if self.f_type == 'macro':
                evaluate_result['f'] = f_sum / len(tags)
                evaluate_result['pre'] = pre_sum / len(tags)
                evaluate_result['rec'] = rec_sum / len(tags)

        if self.f_type == 'micro':
            f, pre, rec = self._compute_f_pre_rec(sum(self._true_positives.values()),
                                                  sum(self._false_negatives.values()),
                                                  sum(self._false_positives.values()))
            evaluate_result['f'] = f
            evaluate_result['pre'] = pre
            evaluate_result['rec'] = rec

        if reset:
            self._true_positives = defaultdict(int)
            self._false_positives = defaultdict(int)
            self._false_negatives = defaultdict(int)

        for key, value in evaluate_result.items():
            evaluate_result[key] = round(value, 6)

        return evaluate_result

    def _compute_f_pre_rec(self, tp, fn, fp):
        """
        """
        pre = tp / (fp + tp + 1e-13)
        rec = tp / (fn + tp + 1e-13)
        f = (1 + self.beta_square) * pre * rec / (self.beta_square * pre + rec + 1e-13)

        return f, pre, rec


def _prepare_metrics(metrics):
    """
    """
    _metrics = []
    if metrics:
        if isinstance(metrics, list):
            for metric in metrics:
                if isinstance(metric, type):
                    metric = metric()
                if isinstance(metric, MetricBase):
                    metric_name = metric.__class__.__name__
                    if not callable(metric.evaluate):
                        raise TypeError(f"{metric_name}.evaluate must be callable, got {type(metric.evaluate)}.")
                    if not callable(metric.get_metric):
                        raise TypeError(f"{metric_name}.get_metric must be callable, got {type(metric.get_metric)}.")
                    _metrics.append(metric)
                else:
                    raise TypeError(
                        f"The type of metric in metrics must be `support.MetricBase`, not `{type(metric)}`.")
        elif isinstance(metrics, MetricBase):
            _metrics = [metrics]
        else:
            raise TypeError(f"The type of metrics should be `list[support.MetricBase]` or `support.MetricBase`, "
                            f"got {type(metrics)}.")
    return _metrics


def _accuracy_topk(y_true, y_prob, k=1):
    """
    """
    y_pred_topk = np.argsort(y_prob, axis=-1)[:, -1:-k - 1:-1]
    y_true_tile = np.tile(np.expand_dims(y_true, axis=1), (1, k))
    y_match = np.any(y_pred_topk == y_true_tile, axis=-1)
    acc = np.sum(y_match) / y_match.shape[0]
    return acc


def _pred_topk(y_prob, k=1):
    """
    """
    y_pred_topk = np.argsort(y_prob, axis=-1)[:, -1:-k - 1:-1]
    x_axis_index = np.tile(
        np.arange(len(y_prob))[:, np.newaxis],
        (1, k))
    y_prob_topk = y_prob[x_axis_index, y_pred_topk]
    return y_pred_topk, y_prob_topk


class CMRC2018Metric(MetricBase):
    def __init__(self, answers=None, raw_chars=None, context_len=None, pred_start=None, pred_end=None):
        super().__init__()
        self._init_param_map(answers=answers, raw_chars=raw_chars, context_len=context_len, pred_start=pred_start,
                             pred_end=pred_end)
        self.em = 0
        self.total = 0
        self.f1 = 0

    def evaluate(self, answers, raw_chars, context_len, pred_start, pred_end):
        """
        """
        batch_size, max_len = pred_start.size()
        context_mask = seq_len_to_mask(context_len, max_len=max_len).eq(0)
        pred_start.masked_fill_(context_mask, float('-inf'))
        pred_end.masked_fill_(context_mask, float('-inf'))
        max_pred_start, pred_start_index = pred_start.max(dim=-1, keepdim=True)
        pred_start_mask = pred_start.eq(max_pred_start).cumsum(dim=-1).eq(0)
        pred_end.masked_fill_(pred_start_mask, float('-inf'))
        pred_end_index = pred_end.argmax(dim=-1) + 1
        pred_ans = []
        for index, (start, end) in enumerate(zip(pred_start_index.flatten().tolist(), pred_end_index.tolist())):
            pred_ans.append(''.join(raw_chars[index][start:end]))
        for answer, pred_an in zip(answers, pred_ans):
            pred_an = pred_an.strip()
            self.f1 += _calc_cmrc2018_f1_score(answer, pred_an)
            self.total += 1
            self.em += _calc_cmrc2018_em_score(answer, pred_an)

    def get_metric(self, reset=True):
        eval_res = {'f1': round(self.f1 / self.total*100, 2), 'em': round(self.em / self.total*100, 2)}
        if reset:
            self.em = 0
            self.total = 0
            self.f1 = 0
        return eval_res

def _cn_segmentation(in_str, rm_punc=False):
    in_str = str(in_str).lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = {'-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=', '，', '。', '：', '？', '！', '“', '”', '；', '’', '《',
               '》', '……', '·', '、', '「', '」', '（', '）', '－', '～', '『', '』'}
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                ss = list(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    # handling last part
    if temp_str != "":
        ss = list(temp_str)
        segs_out.extend(ss)

    return segs_out


# remove punctuation
def _remove_punctuation(in_str):
    in_str = str(in_str).lower().strip()
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


# find longest common string
def _find_lcs(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax


def _calc_cmrc2018_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = _cn_segmentation(ans, rm_punc=True)
        prediction_segs = _cn_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = _find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision = 1.0 * lcs_len / len(prediction_segs)
        recall = 1.0 * lcs_len / len(ans_segs)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return max(f1_scores)


def _calc_cmrc2018_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = _remove_punctuation(ans)
        prediction_ = _remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em
