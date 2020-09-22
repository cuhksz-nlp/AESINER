"""undocumented"""

__all__ = []

import threading

import torch
from torch import nn
from torch.nn.parallel.parallel_apply import get_a_var
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather


def parallel_apply(modules, func_name, inputs, kwargs_tup=None, devices=None):
    r"""Applies each `module` in :attr:`modules` in parallel on arguments
    """
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()
    
    def _worker(i, module, input, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = getattr(module, func_name)(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e
    
    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device))
                   for i, (module, input, kwargs, device) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices))]
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])
    
    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


def _data_parallel_wrapper(func_name, device_ids, output_device):
    """
    """
    
    def wrapper(network, *inputs, **kwargs):
        inputs, kwargs = scatter_kwargs(inputs, kwargs, device_ids, dim=0)
        if len(device_ids) == 1:
            return getattr(network, func_name)(*inputs[0], **kwargs[0])
        replicas = replicate(network, device_ids[:len(inputs)])
        outputs = parallel_apply(replicas, func_name, inputs, kwargs, device_ids[:len(replicas)])
        return gather(outputs, output_device)
    
    return wrapper


def _model_contains_inner_module(model):
    """
    """
    if isinstance(model, nn.Module):
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return True
    return False
