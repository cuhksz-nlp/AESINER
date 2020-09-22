r"""
"""
__all__ = [
    "Trainer"
]

import os
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm
except:
    from .utils import _pseudo_tqdm as tqdm
import warnings

from .batch import DataSetIter, BatchIter
from .callback import CallbackManager, CallbackException, Callback
from .dataset import DataSet
from .losses import _prepare_losser
from .metrics import _prepare_metrics
from .optimizer import Optimizer
from .sampler import Sampler
from .sampler import RandomSampler
from .tester import Tester
from .utils import _CheckError
from .utils import _build_args
from .utils import _check_forward_error
from .utils import _check_loss_evaluate
from .utils import _move_dict_value_to_device
from .utils import _get_func_signature
from .utils import _get_model_device
from .utils import _move_model_to_device
from ._parallel_utils import _model_contains_inner_module
from ._logger import logger
import numpy as np


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embeds.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeds.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Trainer(object):
    """
    """

    def __init__(self, train_data, model, optimizer=None, loss=None,
                 batch_size=32, sampler=None, drop_last=False, update_every=1,
                 num_workers=0, n_epochs=10, print_every=5,
                 dev_data=None, metrics=None, metric_key=None,
                 validate_every=-1, save_path=None, use_tqdm=True, device=None,
                 callbacks=None, check_code_level=-1,
                 use_knowledge=False,
                 knowledge_type=None,
                 pos_th=1,
                 dep_th=1,
                 chunk_th=1,
                 train_feature_data=None,
                 test_feature_data=None,
                 feature2count=None,
                 feature2id=None,
                 id2feature=None,
                 logger_func=None,
                 **kwargs):
        """
        """
        super(Trainer, self).__init__()
        if not isinstance(model, nn.Module):
            raise TypeError(f"The type of model must be torch.nn.Module, got {type(model)}.")

        # check metrics and dev_data
        if (not metrics) and dev_data is not None:
            raise ValueError("No metric for dev_data evaluation.")
        if metrics and (dev_data is None):
            raise ValueError("No dev_data for evaluations, pass dev_data or set metrics to None. ")

        # check update every
        assert update_every >= 1, "update_every must be no less than 1."
        self.update_every = int(update_every)

        # check save_path
        if not (save_path is None or isinstance(save_path, str)):
            raise ValueError("save_path can only be None or `str`.")
        # prepare evaluate
        metrics = _prepare_metrics(metrics)

        # parse metric_key
        # increase_better is True. It means the exp result gets better if the indicator increases.
        # It is true by default.
        self.increase_better = True
        if metric_key is not None:
            self.increase_better = False if metric_key[0] == "-" else True
            self.metric_key = metric_key[1:] if metric_key[0] == "+" or metric_key[0] == "-" else metric_key
        else:
            self.metric_key = None
        # prepare loss
        losser = _prepare_losser(loss)

        self.logger_func = logger_func
        
        if isinstance(train_data, BatchIter):
            if sampler is not None:
                warnings.warn("sampler is ignored when train_data is a BatchIter.")
            if num_workers>0:
                warnings.warn("num_workers is ignored when train_data is BatchIter.")
            if drop_last:
                warnings.warn("drop_last is ignored when train_data is BatchIter.")

        if isinstance(model, nn.parallel.DistributedDataParallel):  # 如果是分布式的
            if device is not None:
                warnings.warn("device is ignored when model is nn.parallel.DistributedDataParallel.")
                device = None
            if sampler is None:
                sampler = torch.utils.data.DistributedSampler(train_data)
            elif not isinstance(sampler, torch.utils.data.DistributedSampler):
                raise TypeError("When using nn.parallel.DistributedDataParallel, "
                                "sampler must be None or torch.utils.data.DistributedSampler.")
            if save_path:
                raise RuntimeError("Saving model in Distributed situation is not allowed right now.")
        else:
            if sampler is not None and not isinstance(sampler, (Sampler, torch.utils.data.Sampler)):
                raise ValueError(f"The type of sampler should be fastNLP.BaseSampler or pytorch's Sampler, got {type(sampler)}")
            if sampler is None:
                sampler = RandomSampler()
            elif hasattr(sampler, 'set_batch_size'):
                sampler.set_batch_size(batch_size)
                
        # train_data: <support.core.dataset.DataSet>
        if isinstance(train_data, DataSet):
            self.data_iterator = DataSetIter(
                dataset=train_data, batch_size=batch_size, num_workers=num_workers, sampler=sampler, drop_last=drop_last)
        elif isinstance(train_data, BatchIter):
            self.data_iterator = train_data
            train_data = train_data.dataset
        else:
            raise TypeError("train_data type {} not support".format(type(train_data)))

        model.train()
        self.model = _move_model_to_device(model, device=device)
        if _model_contains_inner_module(self.model):
            self._forward_func = self.model.module.forward
        else:
            self._forward_func = self.model.forward
        if check_code_level > -1:
            dev_dataset = dev_data
            if isinstance(dev_data, BatchIter):
                dev_dataset = None
                warnings.warn("dev_data is of BatchIter type, ignore validation checking.")
            check_batch_size = min(batch_size, DEFAULT_CHECK_BATCH_SIZE)
            if isinstance(self.model, nn.DataParallel):
                _num_devices = len(self.model.device_ids)
                if batch_size//_num_devices>1:
                    check_batch_size = max(len(self.model.device_ids)*2, check_batch_size)
                else:
                    check_batch_size = max(len(self.model.device_ids), check_batch_size)
            _check_code(dataset=train_data, model=self.model, losser=losser, forward_func=self._forward_func, metrics=metrics,
                        dev_data=dev_dataset, metric_key=self.metric_key, check_level=check_code_level,
                        batch_size=check_batch_size)

        self.train_data = train_data
        self.dev_data = dev_data  # If None, No validation.
        self.losser = losser
        self.metrics = metrics
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.save_path = save_path
        self.print_every = int(print_every)
        self.validate_every = int(validate_every) if validate_every != 0 else -1
        self.best_metric_indicator = None
        self.best_dev_epoch = None
        self.best_dev_step = None
        self.best_dev_perf = None
        self.n_steps = len(self.data_iterator) * self.n_epochs

        # new_add
        self.use_knowledge = use_knowledge
        self.knowledge_type =knowledge_type
        self.pos_th = pos_th
        self.dep_th = dep_th
        self.chunk_th = chunk_th
        self.train_feature_data = train_feature_data
        self.test_feature_data = test_feature_data
        self.feature2count = feature2count
        self.feature2id = feature2id
        self.id2feature = id2feature


        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, Optimizer):
            self.optimizer = optimizer.construct_from_pytorch(self.model.parameters())
        elif optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=4e-3)
        else:
            raise TypeError("optimizer can only be torch.optim.Optimizer type, not {}.".format(type(optimizer)))

        self.logger = logger

        self.use_tqdm = use_tqdm
        if 'test_use_tqdm' in kwargs:
            self.test_use_tqdm = kwargs.get('test_use_tqdm')
        else:
            self.test_use_tqdm = self.use_tqdm
        self.pbar = None
        self.print_every = abs(self.print_every)
        self.kwargs = kwargs

        if self.dev_data is not None:
            self.tester = Tester(model=self.model,
                                 data=self.dev_data,
                                 metrics=self.metrics,
                                 batch_size=kwargs.get("dev_batch_size", self.batch_size),
                                 device=None,  # 由上面的部分处理device
                                 verbose=0,
                                 use_tqdm=self.test_use_tqdm,
                                 use_knowledge=use_knowledge,
                                 knowledge_type=knowledge_type,
                                 pos_th=self.pos_th,
                                 dep_th=self.dep_th,
                                 chunk_th=self.chunk_th,
                                 test_feature_data=test_feature_data,
                                 feature2count=feature2count,
                                 feature2id=feature2id,
                                 id2feature=id2feature
                                 )

        self.step = 0
        self.start_time = None  # start timestamp

        if isinstance(callbacks, Callback):
            callbacks = [callbacks]

        self.callback_manager = CallbackManager(env={"trainer": self},
                                                callbacks=callbacks)

    def train(self, load_best_model=True, on_exception='auto'):
        """
        """
        results = {}
        if self.n_epochs <= 0:
            self.logger.info(f"training epoch is {self.n_epochs}, nothing was done.")
            results['seconds'] = 0.
            return results
        try:
            self._model_device = _get_model_device(self.model)
            self._mode(self.model, is_test=False)
            self._load_best_model = load_best_model
            self.start_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
            start_time = time.time()
            self.logger.info("training epochs started " + self.start_time)

            try:
                self.callback_manager.on_train_begin()
                self._train()
                self.callback_manager.on_train_end()

            except BaseException as e:
                self.callback_manager.on_exception(e)
                if on_exception == 'auto':
                    if not isinstance(e, (CallbackException, KeyboardInterrupt)):
                        raise e
                elif on_exception == 'raise':
                    raise e

            if self.dev_data is not None and self.best_dev_perf is not None:
                self.logger.info(
                    "\nIn Epoch:{}/Step:{}, got best dev performance:".format(self.best_dev_epoch, self.best_dev_step))
                self.logger_func("\nIn Epoch:{}/Step:{}, got best dev performance:".format(self.best_dev_epoch, self.best_dev_step))
                self.logger.info(self.tester._format_eval_results(self.best_dev_perf))
                self.logger_func(self.tester._format_eval_results(self.best_dev_perf))
                results['best_eval'] = self.best_dev_perf
                results['best_epoch'] = self.best_dev_epoch
                results['best_step'] = self.best_dev_step
                if load_best_model:
                    model_name = "best_" + "_".join([self.model.__class__.__name__, self.metric_key, self.start_time])
                    load_succeed = self._load_model(self.model, model_name)
                    if load_succeed:
                        self.logger.info("Reloaded the best model.")
                    else:
                        self.logger.info("Fail to reload best model.")
        finally:
            pass
        results['seconds'] = round(time.time() - start_time, 2)

        return results

    def get_features(self, indices, seq_len):
        # print("seq_len: ", seq_len)
        ret_list = [[], [], []]
        type_list = ["pos", "dep", "chunk"]
        th_list = [self.pos_th, self.dep_th, self.chunk_th]
        for i in range(3):
            for index in indices:
                feature_data = self.train_feature_data[i][index]
                ret = []
                # print(index, len(feature_data), [i.get("word") for i in feature_data])
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
            feature_data = self.train_feature_data[2][index]
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
            feature_data = self.train_feature_data[1][index]
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

    def _train(self):
        if not self.use_tqdm:
            from .utils import _pseudo_tqdm as inner_tqdm
        else:
            inner_tqdm = tqdm
        self.step = 0
        self.epoch = 0
        start = time.time()
        self._mode(self.model, is_test=False)

        with inner_tqdm(total=self.n_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True) as pbar:
            self.pbar = pbar
            avg_loss = 0
            data_iterator = self.data_iterator
            self.batch_per_epoch = data_iterator.num_batches
            for epoch in range(1, self.n_epochs + 1):
                self.epoch = epoch
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.n_epochs))
                # early stopping
                self.callback_manager.on_epoch_begin()
                for indices, batch_x, batch_y in data_iterator:
                    features = self.get_features(indices, seq_len=torch.max(batch_x.get("seq_len")).item())
                    batch_x["pos_features"] = features[0]
                    batch_x["dep_features"] = features[1]
                    batch_x["chunk_features"] = features[2]
                    pos_matrix = self.get_pos_mask_matrix(batch_x.get("chars"), seq_len=torch.max(batch_x.get("seq_len")).item())
                    dep_matrix = self.get_dep_mask_matrix(indices, seq_len=torch.max(batch_x.get("seq_len")).item())
                    chunk_matrix = self.get_chunk_mask_matrix(indices, seq_len=torch.max(batch_x.get("seq_len")).item())
                    batch_x["pos_matrix"] = pos_matrix
                    batch_x["dep_matrix"] = dep_matrix
                    batch_x["chunk_matrix"] = chunk_matrix
                    nan_matrix = None
                    batch_x["nan_matrix"] = nan_matrix

                    self.step += 1
                    _move_dict_value_to_device(batch_x, batch_y, device=self._model_device)
                    indices = data_iterator.get_batch_indices()
                    self.callback_manager.on_batch_begin(batch_x, batch_y, indices)
                    prediction = self._data_forward(self.model, batch_x)
                    self.callback_manager.on_loss_begin(batch_y, prediction)
                    loss = self._compute_loss(prediction, batch_y).mean()
                    avg_loss += loss.item()
                    loss = loss / self.update_every

                    # Is loss NaN or inf? requires_grad = False
                    self.callback_manager.on_backward_begin(loss)
                    self._grad_backward(loss)

                    self.callback_manager.on_backward_end()

                    self._update()
                    self.callback_manager.on_step_end()

                    if self.step % self.print_every == 0:
                        avg_loss = float(avg_loss) / self.print_every
                        if self.use_tqdm:
                            print_output = "loss:{:<6.5f}".format(avg_loss)
                            pbar.update(self.print_every)
                        else:
                            end = time.time()
                            diff = timedelta(seconds=round(end - start))
                            print_output = "[epoch: {:>3} step: {:>4}] train loss: {:>4.6} time: {}".format(
                                epoch, self.step, avg_loss, diff)
                        pbar.set_postfix_str(print_output)
                        avg_loss = 0
                    self.callback_manager.on_batch_end()

                    if ((self.validate_every > 0 and self.step % self.validate_every == 0) or
                        (self.validate_every < 0 and self.step % len(data_iterator) == 0)) \
                            and self.dev_data is not None:

                        eval_res = self._do_validation(epoch=None, step=self.step)
                        eval_str = "Evaluation on dev at Epoch {}/{}. Step:{}/{}: ".format(epoch, self.n_epochs, self.step,
                                                                                    self.n_steps)
                        self.logger_func(eval_str)
                        self.logger_func(self.tester._format_eval_results(eval_res)+'\n')
                        # pbar.write(eval_str + '\n')
                        self.logger.info(eval_str)
                        self.logger.info(self.tester._format_eval_results(eval_res)+'\n')
                        # ================= mini-batch end ==================== #
                # lr decay; early stopping
                self.callback_manager.on_epoch_end()
            # =============== epochs end =================== #
            pbar.close()
            self.pbar = None
        # ============ tqdm end ============== #

    def _do_validation(self, epoch, step):
        self.callback_manager.on_valid_begin()
        res = self.tester.test(epoch)

        is_better_eval = False
        if self._better_eval_result(res):
            if self.save_path is not None:
                self._save_model(self.model,
                                 "best_" + "_".join([self.model.__class__.__name__, self.metric_key, self.start_time]))
            elif self._load_best_model:
                self._best_model_states = {name: param.cpu().clone() for name, param in self.model.state_dict().items()}
            self.best_dev_perf = res
            self.best_dev_epoch = epoch
            self.best_dev_step = step
            is_better_eval = True
        self.callback_manager.on_valid_end(res, self.metric_key, self.optimizer, is_better_eval)
        return res

    def _mode(self, model, is_test=False):
        """
        """
        if is_test:
            model.eval()
        else:
            model.train()

    def _update(self):
        """Perform weight update on a model.

        """
        if self.step % self.update_every == 0:
            self.optimizer.step()

    def _data_forward(self, network, batch_x):
        # x = _build_args(self._forward_func, **x)
        y = network(
            batch_x.get("chars"),
            batch_x.get("target"),
            batch_x.get("bigrams", None),
            batch_x.get("pos_features"),
            batch_x.get("dep_features"),
            batch_x.get("chunk_features"),
            batch_x.get("pos_matrix"),
            batch_x.get("dep_matrix"),
            batch_x.get("chunk_matrix"),
            batch_x.get("nan_matrix")
        )
        if not isinstance(y, dict):
            raise TypeError(
                f"The return value of {_get_func_signature(self._forward_func)} should be dict, got {type(y)}.")
        return y

    def _grad_backward(self, loss):
        """Compute gradient with link rules.

        :param loss: a scalar where back-prop starts

        For PyTorch, just do "loss.backward()"
        """
        if (self.step-1) % self.update_every == 0:
            self.model.zero_grad()
        loss.backward()

    def _compute_loss(self, predict, truth):
        """
        """
        return self.losser(predict, truth)

    def _save_model(self, model, model_name, only_param=False):
        """
        """
        if self.save_path is not None:
            model_path = os.path.join(self.save_path, model_name)
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path, exist_ok=True)
            if _model_contains_inner_module(model):
                model = model.module
            if only_param:
                state_dict = model.state_dict()
                for key in state_dict:
                    state_dict[key] = state_dict[key].cpu()
                torch.save(state_dict, model_path)
            else:
                model.cpu()
                torch.save(model, model_path)
                model.to(self._model_device)

    def _load_model(self, model, model_name, only_param=False):
        if self.save_path is not None:
            model_path = os.path.join(self.save_path, model_name)
            if only_param:
                states = torch.load(model_path)
            else:
                states = torch.load(model_path).state_dict()
            if _model_contains_inner_module(model):
                model.module.load_state_dict(states)
            else:
                model.load_state_dict(states)
        elif hasattr(self, "_best_model_states"):
            model.load_state_dict(self._best_model_states)
        else:
            return False
        return True

    def _better_eval_result(self, metrics):
        """Check if the current epoch yields better validation results.

        :return bool value: True means current results on dev set is the best.
        """
        indicator, indicator_val = _check_eval_results(metrics, self.metric_key, self.metrics)
        if self.metric_key is None:
            self.metric_key = indicator
        is_better = True
        if self.best_metric_indicator is None:
            # first-time validation
            self.best_metric_indicator = indicator_val
        else:
            if self.increase_better is True:
                if indicator_val > self.best_metric_indicator:
                    self.best_metric_indicator = indicator_val
                else:
                    is_better = False
            else:
                if indicator_val < self.best_metric_indicator:
                    self.best_metric_indicator = indicator_val
                else:
                    is_better = False
        return is_better

    @property
    def is_master(self):
        return True

DEFAULT_CHECK_BATCH_SIZE = 2
DEFAULT_CHECK_NUM_BATCH = 2


def _get_value_info(_dict):
    # given a dict value, return information about this dict's value. Return list of str
    strs = []
    for key, value in _dict.items():
        _str = ''
        if isinstance(value, torch.Tensor):
            _str += "\t{}: (1)type:torch.Tensor (2)dtype:{}, (3)shape:{} ".format(key,
                                                                                  value.dtype, value.size())
        elif isinstance(value, np.ndarray):
            _str += "\t{}: (1)type:numpy.ndarray (2)dtype:{}, (3)shape:{} ".format(key,
                                                                                   value.dtype, value.shape)
        else:
            _str += "\t{}: type:{}".format(key, type(value))
        strs.append(_str)
    return strs


from numbers import Number
from .batch import _to_tensor


def _check_code(dataset, model, losser, metrics, forward_func, batch_size=DEFAULT_CHECK_BATCH_SIZE,
                dev_data=None, metric_key=None, check_level=0):
    model_device = _get_model_device(model=model)
    def _iter():
        start_idx = 0
        while start_idx<len(dataset):
            batch_x = {}
            batch_y = {}
            for field_name, field in dataset.get_all_fields().items():
                indices = list(range(start_idx, min(start_idx+batch_size, len(dataset))))
                if field.is_target or field.is_input:
                    batch = field.get(indices)
                    if field.dtype is not None and \
                            issubclass(field.dtype, Number) and not isinstance(batch, torch.Tensor):
                        batch, _ = _to_tensor(batch, field.dtype)
                    if field.is_target:
                        batch_y[field_name] = batch
                    if field.is_input:
                        batch_x[field_name] = batch
            yield (batch_x, batch_y)
            start_idx += batch_size

    for batch_count, (batch_x, batch_y) in enumerate(_iter()):
        _move_dict_value_to_device(batch_x, batch_y, device=model_device)
        # forward check
        if batch_count == 0:
            info_str = ""
            input_fields = _get_value_info(batch_x)
            target_fields = _get_value_info(batch_y)
            if len(input_fields) > 0:
                info_str += "input fields after batch(if batch size is {}):\n".format(batch_size)
                info_str += "\n".join(input_fields)
                info_str += '\n'
            else:
                raise RuntimeError("There is no input field.")
            if len(target_fields) > 0:
                info_str += "target fields after batch(if batch size is {}):\n".format(batch_size)
                info_str += "\n".join(target_fields)
                info_str += '\n'
            else:
                info_str += 'There is no target field.'
            logger.info(info_str)
            _check_forward_error(forward_func=forward_func, dataset=dataset,
                                 batch_x=batch_x, check_level=check_level)
        refined_batch_x = _build_args(forward_func, **batch_x)
        pred_dict = model(**refined_batch_x)
        func_signature = _get_func_signature(forward_func)
        if not isinstance(pred_dict, dict):
            raise TypeError(f"The return value of {func_signature} should be `dict`, not `{type(pred_dict)}`.")
        
        # loss check
        try:
            loss = losser(pred_dict, batch_y)
            # check loss output
            if batch_count == 0:
                if not isinstance(loss, torch.Tensor):
                    raise TypeError(
                        f"The return value of {_get_func_signature(losser.get_loss)} should be `torch.Tensor`, "
                        f"but got `{type(loss)}`.")
                if len(loss.size()) != 0:
                    raise ValueError(
                        f"The size of return value of {_get_func_signature(losser.get_loss)} is {loss.size()}, "
                        f"should be torch.size([])")
            loss.backward()
        except _CheckError as e:
            # TODO: another error raised if _CheckError caught
            pre_func_signature = _get_func_signature(forward_func)
            _check_loss_evaluate(prev_func_signature=pre_func_signature, func_signature=e.func_signature,
                                 check_res=e.check_res, pred_dict=pred_dict, target_dict=batch_y,
                                 dataset=dataset, check_level=check_level)
        model.zero_grad()
        if batch_count + 1 >= DEFAULT_CHECK_NUM_BATCH:
            break
    
    if dev_data is not None:
        tester = Tester(data=dev_data[:batch_size * DEFAULT_CHECK_NUM_BATCH], model=model, metrics=metrics,
                        batch_size=batch_size, verbose=-1, use_tqdm=False)
        evaluate_results = tester.test()
        _check_eval_results(metrics=evaluate_results, metric_key=metric_key, metric_list=metrics)


def _check_eval_results(metrics, metric_key, metric_list):
    if isinstance(metrics, tuple):
        loss, metrics = metrics
    
    if isinstance(metrics, dict):
        metric_dict = list(metrics.values())[0]  # 取第一个metric
        
        if metric_key is None:
            indicator_val, indicator = list(metric_dict.values())[0], list(metric_dict.keys())[0]
        else:
            if metric_key not in metric_dict:
                raise RuntimeError(f"metric key {metric_key} not found in {metric_dict}")
            indicator_val = metric_dict[metric_key]
            indicator = metric_key
    else:
        raise RuntimeError("Invalid metrics type. Expect {}, got {}".format((tuple, dict), type(metrics)))
    return indicator, indicator_val
