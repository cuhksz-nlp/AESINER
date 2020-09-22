"""
support 由 :mod:`~support.core` 、 :mod:`~support.io` 、:mod:`~support.embeddings` 、 :mod:`~support.modules`、
:mod:`~support.models` 等子模块组成，你可以查看每个模块的文档。

- :mod:`~support.core` 是fastNLP 的核心模块，包括 DataSet、 Trainer、 Tester 等组件。详见文档 :mod:`support.core`
- :mod:`~support.io` 是实现输入输出的模块，包括了数据集的读取，模型的存取等功能。详见文档 :mod:`support.io`
- :mod:`~support.embeddings` 提供用于构建复杂网络模型所需的各种embedding。详见文档 :mod:`support.embeddings`
- :mod:`~support.modules`  包含了用于搭建神经网络模型的诸多组件，可以帮助用户快速搭建自己所需的网络。详见文档 :mod:`support.modules`
- :mod:`~support.models` 包含了一些使用 support 实现的完整网络模型，包括 :class:`~support.models.CNNText` 、 :class:`~support.models.SeqLabeling` 等常见模型。详见文档 :mod:`support.models`

support 中最常用的组件可以直接从 support 包中 import ，他们的文档如下：
"""
__all__ = [
    "Instance",
    "FieldArray",
    
    "DataSetIter",
    "BatchIter",
    "TorchLoaderIter",
    
    "Vocabulary",
    "DataSet",
    "Const",
    
    "Trainer",
    "Tester",
    
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
    'SaveModelCallback',
    "CallbackException",
    "EarlyStopError",
    
    "Padder",
    "AutoPadder",
    "EngChar2DPadder",
    
    "AccuracyMetric",
    "SpanFPreRecMetric",
    "ExtractiveQAMetric",
    
    "Optimizer",
    "SGD",
    "Adam",
    "AdamW",
    
    "Sampler",
    "SequentialSampler",
    "BucketSampler",
    "RandomSampler",
    
    "LossFunc",
    "CrossEntropyLoss",
    "L1Loss",
    "BCELoss",
    "NLLLoss",
    "LossInForward",
    
    "cache_results",
    
    'logger'
]
__version__ = '0.4.5'

import sys

from . import embeddings
from . import models
from . import modules
from .core import *
from .doc_utils import doc_process
from .io import loader, pipe

doc_process(sys.modules[__name__])