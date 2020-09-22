r"""
"""
__all__ = [
    "Callback",

    "GradientClipCallback",
    "EarlyStopCallback",
    "FitlogCallback",
    "EvaluateCallback",
    "LRScheduler",
    "ControlC",
    "LRFinder",
    "TensorboardCallback",
    "WarmupCallback",
    "SaveModelCallback",
    
    "CallbackException",
    "EarlyStopError"
]

import os
import sys
from copy import deepcopy

import torch

from .utils import _save_model

try:
    from tensorboardX import SummaryWriter
    
    tensorboardX_flag = True
except:
    tensorboardX_flag = False

from .dataset import DataSet
from .tester import Tester
from ._logger import logger
from .utils import _check_fp16

try:
    import fitlog
except:
    pass

try:
    from apex import amp
except:
    amp = None


class Callback(object):
    """
    """
    
    def __init__(self):
        super(Callback, self).__init__()
        self._trainer = None
        self._disabled = False

    @property
    def trainer(self):
        """
        """
        return self._trainer
    
    @property
    def step(self):
        return self._trainer.step
    
    @property
    def n_steps(self):
        return self._trainer.n_steps
    
    @property
    def batch_size(self):
        return self._trainer.batch_size
    
    @property
    def epoch(self):
        return self._trainer.epoch
    
    @property
    def n_epochs(self):
        return self._trainer.n_epochs
    
    @property
    def optimizer(self):
        return self._trainer.optimizer
    
    @property
    def model(self):
        return self._trainer.model
    
    @property
    def pbar(self):
        return self._trainer.pbar
    
    @property
    def update_every(self):
        return self._trainer.update_every
    
    @property
    def batch_per_epoch(self):
        return self._trainer.batch_per_epoch

    @property
    def is_master(self):
        return self._trainer.is_master

    @property
    def disabled(self):
        return self._disabled

    @property
    def logger(self):
        return getattr(self._trainer, 'logger', logger)

    def on_train_begin(self):
        """
        """
        pass
    
    def on_epoch_begin(self):
        """
        """
        pass
    
    def on_batch_begin(self, batch_x, batch_y, indices):
        """
        """
        pass
    
    def on_loss_begin(self, batch_y, predict_y):
        """
        """
        pass
    
    def on_backward_begin(self, loss):
        """
        """
        pass
    
    def on_backward_end(self):
        """
        """
        pass
    
    def on_step_end(self):
        """

        """
        pass
    
    def on_batch_end(self):
        """
        """
        pass
    
    def on_valid_begin(self):
        """
        :return:
        """
        pass
    
    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        pass
    
    def on_epoch_end(self):
        """
        """
        pass
    
    def on_train_end(self):
        """
        """
        pass
    
    def on_exception(self, exception):
        """
        """
        pass


def _transfer(func):
    """
    """
    
    def wrapper(manager, *arg):
        returns = []
        for callback in manager.callbacks:
            if callback.disabled:
                continue
            returns.append(getattr(callback, func.__name__)(*arg))
        return returns
    
    return wrapper


class CallbackManager(Callback):
    """
    """
    def __init__(self, env, callbacks=None):
        """
        """
        super(CallbackManager, self).__init__()
        # set attribute of trainer environment
        self._env = env
        self.callbacks = []
        if callbacks:
            self.callbacks = self.prepare_callbacks(callbacks)

    def prepare_callbacks(self, callbacks):
        if not callbacks:
            return []
        if isinstance(callbacks, list):
            if all([isinstance(cb, Callback) for cb in callbacks]) is True:
                pass
            else:
                obj = [not isinstance(cb, Callback) for cb in callbacks][0]
                raise TypeError(f"Expect sub-classes of Callback. Got {type(obj)}")
        else:
            raise TypeError(f"Expect callbacks in CallbackManager(callbacks) to be list. Got {type(callbacks)}.")

        for env_name, env_val in self._env.items():
            for callback in callbacks:
                setattr(callback, '_' + env_name, env_val)  # Callback.trainer
        return callbacks

    @_transfer
    def on_train_begin(self):
        pass
    
    @_transfer
    def on_epoch_begin(self):
        pass
    
    @_transfer
    def on_batch_begin(self, batch_x, batch_y, indices):
        pass
    
    @_transfer
    def on_loss_begin(self, batch_y, predict_y):
        pass
    
    @_transfer
    def on_backward_begin(self, loss):
        pass
    
    @_transfer
    def on_backward_end(self):
        pass
    
    @_transfer
    def on_step_end(self):
        pass
    
    @_transfer
    def on_batch_end(self):
        pass
    
    @_transfer
    def on_valid_begin(self):
        pass
    
    @_transfer
    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        pass

    @_transfer
    def on_validation(self):
        pass
    
    @_transfer
    def on_epoch_end(self):
        pass
    
    @_transfer
    def on_train_end(self):
        pass
    
    @_transfer
    def on_exception(self, exception):
        pass


class DistCallbackManager(CallbackManager):
    def __init__(self, env, callbacks_all=None, callbacks_master=None):
        super(DistCallbackManager, self).__init__(env)
        assert 'trainer' in env
        self._trainer = env['trainer']
        self.callbacks_master = []
        self.callbacks_all = []
        self.add_callback(callbacks_all, master=False)
        self.add_callback(callbacks_master, master=True)

    def patch_callback(self, callbacks, disabled):
        if not callbacks:
            return
        if not isinstance(callbacks, (list, tuple)):
            callbacks = [callbacks]
        for cb in callbacks:
            cb._disabled = disabled

    def add_callback(self, cb, master=False):
        if master:
            self.patch_callback(cb, not self.is_master)
            self.callbacks_master += self.prepare_callbacks(cb)
        else:
            self.callbacks_all += self.prepare_callbacks(cb)
        self.callbacks = self.callbacks_all + self.callbacks_master


class GradientClipCallback(Callback):
    """
    """
    
    def __init__(self, parameters=None, clip_value=1, clip_type='norm'):
        """
        """
        super().__init__()
        
        from torch import nn
        if clip_type == 'norm':
            self.clip_fun = nn.utils.clip_grad_norm_
        elif clip_type == 'value':
            self.clip_fun = nn.utils.clip_grad_value_
        else:
            raise ValueError("Only supports `norm` or `value` right now.")
        self.parameters = parameters
        self.clip_value = clip_value
    
    def on_backward_end(self):
        if self.step%self.update_every==0:
            if self.parameters is None:
                if getattr(self.trainer, 'fp16', ''):
                    _check_fp16()
                    self.clip_fun(amp.master_params(self.optimizer), self.clip_value)
                else:
                    self.clip_fun(self.model.parameters(), self.clip_value)
            else:
                self.clip_fun(self.parameters, self.clip_value)


class EarlyStopCallback(Callback):
    """
    """
    
    def __init__(self, patience):
        """
        """
        super(EarlyStopCallback, self).__init__()
        self.patience = patience
        self.wait = 0
    
    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        if not is_better_eval:
            if self.wait == self.patience:
                raise EarlyStopError("Early stopping raised.")
            else:
                self.wait += 1
        else:
            self.wait = 0
    
    def on_exception(self, exception):
        if isinstance(exception, EarlyStopError):
            logger.info("Early Stopping triggered in epoch {}!".format(self.epoch))
        else:
            raise exception


class FitlogCallback(Callback):
    """
    """

    def __init__(self, data=None, tester=None, log_loss_every=0, verbose=0, log_exception=False):
        """
        """
        super().__init__()
        self.datasets = {}
        self.testers = {}
        self._log_exception = log_exception
        assert isinstance(log_loss_every, int) and log_loss_every>=0
        if tester is not None:
            if isinstance(tester, dict):
                for name, test in tester.items():
                    if not isinstance(test, Tester):
                        raise TypeError(f"{name} in tester is not a valid fastNLP.Tester.")
                    self.testers['tester-' + name] = test
            if isinstance(tester, Tester):
                self.testers['tester-test'] = tester
            for tester in self.testers.values():
                setattr(tester, 'verbose', 0)

        if isinstance(data, dict):
            for key, value in data.items():
                assert isinstance(value, DataSet), f"Only DataSet object is allowed, not {type(value)}."
            for key, value in data.items():
                self.datasets['data-' + key] = value
        elif isinstance(data, DataSet):
            self.datasets['data-test'] = data
        elif data is not None:
            raise TypeError("data receives dict[DataSet] or DataSet object.")
        
        self.verbose = verbose
        self._log_loss_every = log_loss_every
        self._avg_loss = 0

    def on_train_begin(self):
        if (len(self.datasets) > 0 or len(self.testers) > 0) and self.trainer.dev_data is None:
            raise RuntimeError("Trainer has no dev data, you cannot pass extra data to do evaluation.")
        
        if len(self.datasets) > 0:
            for key, data in self.datasets.items():
                tester = Tester(data=data, model=self.model,
                                batch_size=self.trainer.kwargs.get('dev_batch_size', self.batch_size),
                                metrics=self.trainer.metrics,
                                verbose=0,
                                use_tqdm=self.trainer.test_use_tqdm)
                self.testers[key] = tester
        fitlog.add_progress(total_steps=self.n_steps)
    
    def on_backward_begin(self, loss):
        if self._log_loss_every>0:
            self._avg_loss += loss.item()
            if self.step%self._log_loss_every==0:
                fitlog.add_loss(self._avg_loss/self._log_loss_every, name='loss', step=self.step, epoch=self.epoch)
                self._avg_loss = 0

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        if better_result:
            eval_result = deepcopy(eval_result)
            eval_result['step'] = self.step
            eval_result['epoch'] = self.epoch
            fitlog.add_best_metric(eval_result)
        fitlog.add_metric(eval_result, step=self.step, epoch=self.epoch)
        if len(self.testers) > 0:
            for key, tester in self.testers.items():
                try:
                    eval_result = tester.test()
                    if self.verbose != 0:
                        self.pbar.write("FitlogCallback evaluation on {}:".format(key))
                        self.pbar.write(tester._format_eval_results(eval_result))
                    fitlog.add_metric(eval_result, name=key, step=self.step, epoch=self.epoch)
                    if better_result:
                        fitlog.add_best_metric(eval_result, name=key)
                except Exception:
                    self.pbar.write("Exception happens when evaluate on DataSet named `{}`.".format(key))
    
    def on_train_end(self):
        fitlog.finish()
    
    def on_exception(self, exception):
        fitlog.finish(status=1)
        if self._log_exception:
            fitlog.add_other(repr(exception), name='except_info')


class EvaluateCallback(Callback):
    """
    """

    def __init__(self, data=None, tester=None):
        """
        """
        super().__init__()
        self.datasets = {}
        self.testers = {}
        if tester is not None:
            if isinstance(tester, dict):
                for name, test in tester.items():
                    if not isinstance(test, Tester):
                        raise TypeError(f"{name} in tester is not a valid fastNLP.Tester.")
                    self.testers['tester-' + name] = test
            if isinstance(tester, Tester):
                self.testers['tester-test'] = tester
            for tester in self.testers.values():
                setattr(tester, 'verbose', 0)

        if isinstance(data, dict):
            for key, value in data.items():
                assert isinstance(value, DataSet), f"Only DataSet object is allowed, not {type(value)}."
            for key, value in data.items():
                self.datasets['data-' + key] = value
        elif isinstance(data, DataSet):
            self.datasets['data-test'] = data
        elif data is not None:
            raise TypeError("data receives dict[DataSet] or DataSet object.")

    def on_train_begin(self):
        if len(self.datasets) > 0 and self.trainer.dev_data is None:
            raise RuntimeError("Trainer has no dev data, you cannot pass extra DataSet to do evaluation.")

        if len(self.datasets) > 0:
            for key, data in self.datasets.items():
                tester = Tester(data=data, model=self.model,
                                batch_size=self.trainer.kwargs.get('dev_batch_size', self.batch_size),
                                metrics=self.trainer.metrics, verbose=0,
                                use_tqdm=self.trainer.test_use_tqdm)
                self.testers[key] = tester

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        if len(self.testers) > 0:
            for key, tester in self.testers.items():
                try:
                    eval_result = tester.test()
                    self.logger.info("EvaluateCallback evaluation on {}:".format(key))
                    self.logger.info(tester._format_eval_results(eval_result))
                except Exception:
                    self.logger.error("Exception happens when evaluate on DataSet named `{}`.".format(key))


class LRScheduler(Callback):
    """
    """
    
    def __init__(self, lr_scheduler):
        """
        """
        super(LRScheduler, self).__init__()
        import torch.optim
        if isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
            self.scheduler = lr_scheduler
        else:
            raise ValueError(f"Expect torch.optim.lr_scheduler for LRScheduler. Got {type(lr_scheduler)}.")
    
    def on_epoch_end(self):
        self.scheduler.step(self.epoch)


class ControlC(Callback):
    """
    """
    
    def __init__(self, quit_all):
        """
        """
        super(ControlC, self).__init__()
        if type(quit_all) != bool:
            raise ValueError("In KeyBoardInterrupt, quit_all arguemnt must be a bool.")
        self.quit_all = quit_all
    
    def on_exception(self, exception):
        if isinstance(exception, KeyboardInterrupt):
            if self.quit_all is True:
                import sys
                sys.exit(0)
            else:
                pass
        else:
            raise exception


class SmoothValue(object):
    """work for LRFinder"""
    
    def __init__(self, beta: float):
        self.beta, self.n, self.mov_avg = beta, 0, 0
        self.smooth = None
    
    def add_value(self, val: float) -> None:
        """Add `val` to calculate updated smoothed value."""
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)


class LRFinder(Callback):
    """
    """
    
    def __init__(self, start_lr=1e-6, end_lr=10):
        """
        """
        super(LRFinder, self).__init__()
        self.start_lr, self.end_lr = start_lr, end_lr
        
        self.stop = False
        self.best_loss = 0.
        self.best_lr = None
        self.loss_history = []
        self.smooth_value = SmoothValue(0.8)
        self.opt = None
        self.find = None

    @property
    def lr_gen(self):
        scale = (self.end_lr - self.start_lr) / self.batch_per_epoch
        return (self.start_lr + scale * (step + 1) for step in range(self.batch_per_epoch))
    
    @property
    def num_it(self):
        return self.batch_per_epoch
    
    def on_epoch_begin(self):
        if self.epoch == 1:
            self.opt = self.trainer.optimizer
            self.opt.param_groups[0]["lr"] = self.start_lr
            torch.save(self.model.state_dict(), 'tmp')
            self.find = True
    
    def on_backward_begin(self, loss):
        if self.find:
            if torch.isnan(loss) or self.stop is True:
                self.stop = True
                return
            loss_val = loss.detach().mean().item()
            self.loss_history.append(loss_val)
            self.smooth_value.add_value(loss_val)
            if self.best_loss == 0. or self.smooth_value.smooth < self.best_loss:
                self.best_loss = self.smooth_value.smooth
                self.best_lr = self.opt.param_groups[0]["lr"]
    
    def on_batch_end(self, *args):
        if self.find:
            lr = next(self.lr_gen, None)
            if lr is None or self.stop is True or self.loss_history[-1] > 4 * self.best_loss:
                self.stop = True
                return
            self.opt.param_groups[0]["lr"] = lr

    def on_epoch_end(self):
        if self.epoch == 1:  # first epoch
            self.opt.param_groups[0]["lr"] = self.best_lr
            self.find = False
            states = torch.load('tmp')
            self.model.load_state_dict(states)
            os.remove('tmp')
            self.pbar.write("Model reset. \nFind best lr={}".format(self.best_lr))


class TensorboardCallback(Callback):
    """
    """
    
    def __init__(self, *options):
        super(TensorboardCallback, self).__init__()
        args = {"model", "loss", "metric"}
        for opt in options:
            if opt not in args:
                raise ValueError("Unrecognized argument {}. Expect one of {}".format(opt, args))
        self.options = options
        self._summary_writer = None
        self.graph_added = False
    
    def on_train_begin(self):
        save_dir = self.trainer.save_path
        if save_dir is None:
            path = os.path.join("./", 'tensorboard_logs_{}'.format(self.trainer.start_time))
        else:
            path = os.path.join(save_dir, 'tensorboard_logs_{}'.format(self.trainer.start_time))
        if tensorboardX_flag:
            self._summary_writer = SummaryWriter(path)
        else:
            self._summary_writer = None
    
    def on_batch_begin(self, batch_x, batch_y, indices):
        if "model" in self.options and self.graph_added is False:
            self.graph_added = True
    
    def on_backward_begin(self, loss):
        if "loss" in self.options and self._summary_writer:
            self._summary_writer.add_scalar("loss", loss.item(), global_step=self.trainer.step)
        
        if "model" in self.options and self._summary_writer:
            for name, param in self.trainer.model.named_parameters():
                if param.requires_grad:
                    self._summary_writer.add_scalar(name + "_mean", param.mean(), global_step=self.trainer.step)
                    self._summary_writer.add_scalar(name + "_grad_mean", param.grad.mean(),
                                                    global_step=self.trainer.step)
    
    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        if "metric" in self.options and self._summary_writer:
            for name, metric in eval_result.items():
                for metric_key, metric_val in metric.items():
                    self._summary_writer.add_scalar("valid_{}_{}".format(name, metric_key), metric_val,
                                                    global_step=self.trainer.step)
    
    def on_train_end(self):
        if self._summary_writer:
            self._summary_writer.close()
            del self._summary_writer
    
    def on_exception(self, exception):
        if hasattr(self, "_summary_writer"):
            self._summary_writer.close()
            del self._summary_writer


class WarmupCallback(Callback):
    """
    """
    def __init__(self, warmup=0.1, schedule='constant'):
        """
        """
        super().__init__()
        self.warmup = max(warmup, 0.)

        self.initial_lrs = []
        if schedule == 'constant':
            self.get_lr = self._get_constant_lr
        elif schedule == 'linear':
            self.get_lr = self._get_linear_lr
        else:
            raise RuntimeError("Only support 'linear', 'constant'.")

    def _get_constant_lr(self, progress):
        if progress<self.warmup:
            return progress/self.warmup
        return 1

    def _get_linear_lr(self, progress):
        if progress<self.warmup:
            return progress/self.warmup
        return max((progress - 1.) / (self.warmup - 1.), 0.)

    def on_train_begin(self):
        self.t_steps = (len(self.trainer.train_data) // (self.batch_size*self.update_every) +
                            int(len(self.trainer.train_data) % (self.batch_size*self.update_every)!= 0)) * self.n_epochs
        if self.warmup>1:
            self.warmup = self.warmup/self.t_steps
        self.t_steps = max(2, self.t_steps)
        for group in self.optimizer.param_groups:
            self.initial_lrs.append(group['lr'])

    def on_backward_end(self):
        if self.step%self.update_every==0:
            progress = (self.step/self.update_every)/self.t_steps
            for lr, group in zip(self.initial_lrs, self.optimizer.param_groups):
                group['lr'] = lr * self.get_lr(progress)


class SaveModelCallback(Callback):
    """
    """
    def __init__(self, save_dir, top=3, only_param=False, save_on_exception=False):
        """
        """
        super().__init__()

        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        if top < 0:
            self.top = sys.maxsize
        else:
            self.top = top
        self._ordered_save_models = []

        self.only_param = only_param
        self.save_on_exception = save_on_exception

    def on_train_begin(self):
        self.save_dir = os.path.join(self.save_dir, self.trainer.start_time)

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        metric_value = list(eval_result.values())[0][metric_key]
        self._save_this_model(metric_value)

    def _insert_into_ordered_save_models(self, pair):
        index = -1
        for _pair in self._ordered_save_models:
            if _pair[0]>=pair[0] and self.trainer.increase_better:
                break
            if not self.trainer.increase_better and _pair[0]<=pair[0]:
                break
            index += 1
        save_pair = None
        if len(self._ordered_save_models)<self.top or (len(self._ordered_save_models)>=self.top and index!=-1):
            save_pair = pair
            self._ordered_save_models.insert(index+1, pair)
        delete_pair = None
        if len(self._ordered_save_models)>self.top:
            delete_pair = self._ordered_save_models.pop(0)
        return save_pair, delete_pair

    def _save_this_model(self, metric_value):
        name = "epoch-{}_step-{}_{}-{:.6f}.pt".format(self.epoch, self.step, self.trainer.metric_key, metric_value)
        save_pair, delete_pair = self._insert_into_ordered_save_models((metric_value, name))
        if save_pair:
            try:
                _save_model(self.model, model_name=name, save_dir=self.save_dir, only_param=self.only_param)
            except Exception as e:
                logger.error(f"The following exception:{e} happens when save model to {self.save_dir}.")
        if delete_pair:
            try:
                delete_model_path = os.path.join(self.save_dir, delete_pair[1])
                if os.path.exists(delete_model_path):
                    os.remove(delete_model_path)
            except Exception as e:
                logger.error(f"Fail to delete model {name} at {self.save_dir} caused by exception:{e}.")

    def on_exception(self, exception):
        if self.save_on_exception:
            name = "epoch-{}_step-{}_Exception-{}.pt".format(self.epoch, self.step, exception.__class__.__name__)
            _save_model(self.model, model_name=name, save_dir=self.save_dir, only_param=self.only_param)


class CallbackException(BaseException):
    """
   """
    
    def __init__(self, msg):
        """
        """
        super(CallbackException, self).__init__(msg)


class EarlyStopError(CallbackException):
    """
    """
    
    def __init__(self, msg):
        super(EarlyStopError, self).__init__(msg)


class EchoCallback(Callback):
    """
    """
    def __init__(self, name, out=sys.stdout):
        super(EchoCallback, self).__init__()
        self.name = name
        self.out = out  # deprecated

    def __getattribute__(self, item):
        if item.startswith('on_'):
            logger.info('{}.{} has been called at pid: {}'.format(self.name, item, os.getpid()))
        return super(EchoCallback, self).__getattribute__(item)


class _TesterCallback(Callback):
    def __init__(self, data, model, metrics, metric_key=None, batch_size=16, num_workers=None):
        super(_TesterCallback, self).__init__()
        if hasattr(model, 'module'):
            # for data parallel model
            model = model.module
        self.tester = Tester(data, model,
                             metrics=metrics, batch_size=batch_size,
                             num_workers=num_workers, verbose=0)
        if metric_key is not None:
            self.metric_key, self.increase_better = self._parse_metric_key(metric_key)
        else:
            self.metric_key = None
            self.increase_better = True
        self.score = None

    def on_valid_begin(self):
        cur_score = self.tester.test()
        eval_str = "Evaluation at Epoch {}/{}. Step:{}/{}. - {}".format(
                    self.epoch, self.n_epochs, self.step, self.n_steps,
                    self.tester._format_eval_results(cur_score))
        self.logger.info(eval_str)
        is_better = self.compare_better(cur_score)
        if is_better:
            self.score = cur_score
        return cur_score, is_better

    @staticmethod
    def _get_score(metric_dict, key):
        for metric in metric_dict.items():
            if key in metric:
                return metric[key]
        return None

    @staticmethod
    def _parse_metric_key(metric_key):
        # parse metric_key
        # increase_better is True. It means the exp result gets better if the indicator increases.
        # It is true by default.
        increase_better = False if metric_key[0] == "-" else True
        metric_key = metric_key[1:] if metric_key[0] == "+" or metric_key[0] == "-" else metric_key
        return metric_key, increase_better

    def compare_better(self, a):
        if self.score is None:
            return True
        if self.metric_key is None:
            metric_key = list(list(self.score.values())[0].keys())[0]
            self.metric_key, self.increase_better = self._parse_metric_key(metric_key)
        k = self.metric_key
        score = self._get_score(self.score, k)
        new_score = self._get_score(a, k)
        if score is None or new_score is None:
            return False
        if self.increase_better:
            return score <= new_score
        else:
            return score >= new_score
