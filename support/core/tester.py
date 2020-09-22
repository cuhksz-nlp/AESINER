"""
"""
import time

import torch
import torch.nn as nn
import numpy as np

try:
    from tqdm.auto import tqdm
except:
    from .utils import _pseudo_tqdm as tqdm

from .batch import BatchIter, DataSetIter
from .dataset import DataSet
from .metrics import _prepare_metrics
from .sampler import SequentialSampler
from .utils import _CheckError
from .utils import _build_args
from .utils import _check_loss_evaluate
from .utils import _move_dict_value_to_device
from .utils import _get_func_signature
from .utils import _get_model_device
from .utils import _move_model_to_device
from ._parallel_utils import _data_parallel_wrapper
from ._parallel_utils import _model_contains_inner_module
from functools import partial
from ._logger import logger
import json

__all__ = [
    "Tester"
]


class Tester(object):
    """
    """
    
    def __init__(self, data, model, metrics, batch_size=16, num_workers=0, device=None, verbose=1, use_tqdm=True,
                 use_knowledge=False,
                 knowledge_type=None,
                 pos_th=1,
                 dep_th=1,
                 chunk_th=1,
                 test_feature_data=None,
                 feature2count=None,
                 feature2id=None,
                 id2feature=None):
        """
        """
        super(Tester, self).__init__()

        if not isinstance(model, nn.Module):
            raise TypeError(f"The type of model must be `torch.nn.Module`, got `{type(model)}`.")

        self.metrics = _prepare_metrics(metrics)
        
        self.data = data
        self._model = _move_model_to_device(model, device=device)
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_tqdm = use_tqdm
        self.logger = logger

        self.use_knowledge = use_knowledge
        self.knowledge_type = knowledge_type
        self.pos_th = pos_th
        self.dep_th = dep_th
        self.chunk_th = chunk_th
        self.test_feature_data = test_feature_data
        self.feature2count = feature2count
        self.feature2id = feature2id
        self.id2feature = id2feature

        if isinstance(data, DataSet):
            self.data_iterator = DataSetIter(
                dataset=data, batch_size=batch_size, num_workers=num_workers, sampler=SequentialSampler())
        elif isinstance(data, BatchIter):
            self.data_iterator = data
        else:
            raise TypeError("data type {} not support".format(type(data)))

        if (hasattr(self._model, 'predict') and callable(self._model.predict)) or \
                (_model_contains_inner_module(self._model) and hasattr(self._model.module, 'predict') and
                 callable(self._model.module.predict)):
            if isinstance(self._model, nn.DataParallel):
                self._predict_func_wrapper = partial(_data_parallel_wrapper('predict',
                                                                    self._model.device_ids,
                                                                    self._model.output_device),
                                                     network=self._model.module)
                self._predict_func = self._model.module.predict
            elif isinstance(self._model, nn.parallel.DistributedDataParallel):
                self._predict_func = self._model.module.predict
                self._predict_func_wrapper = self._model.module.predict
            else:
                self._predict_func = self._model.predict
                self._predict_func_wrapper = self._model.predict
        else:
            if _model_contains_inner_module(model):
                self._predict_func_wrapper = self._model.forward
                self._predict_func = self._model.module.forward
            else:
                self._predict_func = self._model.forward
                self._predict_func_wrapper = self._model.forward

    def get_features(self, indices, seq_len):
        ret_list = [[], [], []]
        type_list = ["pos", "dep", "chunk"]
        th_list = [self.pos_th, self.dep_th, self.chunk_th]
        for i in range(3):
            for index in indices:
                feature_data = self.test_feature_data[i][index]
                ret = []
                for item in feature_data:
                    word = item["word"]
                    feature = item[type_list[i]]
                    word_feature = word + "_" + feature
                    if self.feature2count.get(word_feature, 0) >= th_list[i]:
                        ret.append(self.feature2id[word_feature])
                    else:
                        ret.append(self.feature2id[feature])
                ret += [0] * (seq_len - len(ret))
                ret_list[i].append(ret)
            ret_list[i] = torch.tensor(ret_list[i])
        return ret_list

    def get_pos_mask_matrix(self, chars, seq_len):
        chars = chars.tolist()
        ret_list = []
        for sent in chars:
            ret = []
            temp_len = len([i for i in sent if i != 0])
            for i, word in enumerate(sent):
                if word in [0, 1]:
                    ret.append([0] * seq_len)
                else:
                    if i == 0:
                        ret.append([1] * 2 + [0] * (seq_len - 2))
                    elif i == temp_len - 1:
                        ret.append([0] * (temp_len - 2) + [1] * 2 + [0] * (seq_len - temp_len))
                    else:
                        ret.append([0] * (i - 1) + [1] * 3 + [0] * (seq_len - i - 2))
            ret_list.append(ret)
        return torch.tensor(ret_list)

    def get_chunk_mask_matrix(self, indices, seq_len):
        ret_list = []
        for index in indices:
            ret = []
            feature_data = self.test_feature_data[2][index]
            for item in feature_data:
                feature = item["range"]
                start, end = feature
                ret.append([0] * start + [1] * (end - start + 1) + [0] * (seq_len - end - 1))
            for _ in range(seq_len - len(feature_data)):
                ret.append([0] * seq_len)
            ret_list.append(ret)
        return torch.tensor(ret_list)

    def get_dep_mask_matrix(self, indices, seq_len):
        ret_list = []
        for index in indices:
            ret = [[0] * seq_len for _ in range(seq_len)]
            feature_data = self.test_feature_data[1][index]
            for i, item in enumerate(feature_data):
                feature = item["range"]
                for j in range(len(feature)):
                    if feature[j] == 1:
                        ret[i][j] = 1
            ret_list.append(ret)
        return torch.tensor(ret_list)

    def generate_nan_mask(self, origin_mask):
        b, l, d = origin_mask.shape
        nan_mask = np.zeros((b, l, d), dtype=int)
        for i in range(b):
            for j in range(l):
                if not any(origin_mask[i, j]):
                    nan_mask[i, j] = 1

        return torch.from_numpy(nan_mask)

    def test(self, epoch=None):
        # turn on the testing mode; clean up the history
        self._model_device = _get_model_device(self._model)
        network = self._model
        self._mode(network, is_test=True)
        data_iterator = self.data_iterator
        eval_results = {}
        try:
            with torch.no_grad():
                if not self.use_tqdm:
                    from .utils import _pseudo_tqdm as inner_tqdm
                else:
                    inner_tqdm = tqdm
                with inner_tqdm(total=len(data_iterator), leave=False, dynamic_ncols=True) as pbar:
                    pbar.set_description_str(desc="Test")

                    start_time = time.time()

                    result = []

                    for indices, batch_x, batch_y in data_iterator:
                        features = self.get_features(indices, seq_len=torch.max(batch_x.get("seq_len")).item())
                        batch_x["pos_features"] = features[0]
                        batch_x["dep_features"] = features[1]
                        batch_x["chunk_features"] = features[2]
                        pos_matrix = self.get_pos_mask_matrix(batch_x.get("chars"),
                                                              seq_len=torch.max(batch_x.get("seq_len")).item())
                        dep_matrix = self.get_dep_mask_matrix(indices, seq_len=torch.max(batch_x.get("seq_len")).item())
                        chunk_matrix = self.get_chunk_mask_matrix(indices,
                                                                  seq_len=torch.max(batch_x.get("seq_len")).item())
                        batch_x["pos_matrix"] = pos_matrix
                        batch_x["dep_matrix"] = dep_matrix
                        batch_x["chunk_matrix"] = chunk_matrix
                        nan_matrix = None
                        batch_x["nan_matrix"] = nan_matrix
                        _move_dict_value_to_device(batch_x, batch_y, device=self._model_device)

                        pred_dict = self._data_forward(self._predict_func, batch_x)

                        pred = pred_dict["pred"].tolist()
                        target = batch_y["target"].tolist()

                        if epoch is not None:
                            for i, p, t in zip(indices, pred, target):
                                result.append({"index": i, "pred": p, "target": t})

                        if not isinstance(pred_dict, dict):
                            raise TypeError(f"The return value of {_get_func_signature(self._predict_func)} "
                                            f"must be `dict`, got {type(pred_dict)}.")
                        for metric in self.metrics:
                            metric(pred_dict, batch_y)

                        if self.use_tqdm:
                            pbar.update()

                    if epoch is not None:
                        with open("result/epoch_%d.json" % epoch, "w+", encoding="utf-8") as f:
                            f.write(json.dumps(result))

                    for metric in self.metrics:
                        eval_result = metric.get_metric()
                        if not isinstance(eval_result, dict):
                            raise TypeError(f"The return value of {_get_func_signature(metric.get_metric)} must be "
                                            f"`dict`, got {type(eval_result)}")
                        metric_name = metric.get_metric_name()
                        eval_results[metric_name] = eval_result
                    pbar.close()
                    end_time = time.time()
                    test_str = f'Evaluate data in {round(end_time - start_time, 2)} seconds!'
                    if self.verbose >= 0:
                        self.logger.info(test_str)
        except _CheckError as e:
            prev_func_signature = _get_func_signature(self._predict_func)
            _check_loss_evaluate(prev_func_signature=prev_func_signature, func_signature=e.func_signature,
                                 check_res=e.check_res, pred_dict=pred_dict, target_dict=batch_y,
                                 dataset=self.data, check_level=0)
        
        if self.verbose >= 1:
            logger.info("[tester] \n{}".format(self._format_eval_results(eval_results)))
        self._mode(network, is_test=False)
        return eval_results
    
    def _mode(self, model, is_test=False):
        """
        """
        if is_test:
            model.eval()
        else:
            model.train()
    
    def _data_forward(self, func, batch_x):
        """A forward pass of the model. """
        # x = _build_args(func, **x)
        y = self._predict_func_wrapper(
            batch_x.get("chars"),
            batch_x.get("bigrams", None),
            batch_x.get("pos_features"),
            batch_x.get("dep_features"),
            batch_x.get("chunk_features"),
            batch_x.get("pos_matrix"),
            batch_x.get("dep_matrix"),
            batch_x.get("chunk_matrix"),
            batch_x.get("nan_matrix")
        )
        return y
    
    def _format_eval_results(self, results):
        """
        """
        _str = ''
        for metric_name, metric_result in results.items():
            _str += metric_name + ': '
            _str += ", ".join([str(key) + "=" + str(value) for key, value in metric_result.items()])
            _str += '\n'
        return _str[:-1]
