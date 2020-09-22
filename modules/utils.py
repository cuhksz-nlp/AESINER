def set_rng_seed(rng_seed:int = None, random:bool = True, numpy:bool = True,
                 pytorch:bool=True, deterministic:bool=True):
    if rng_seed is None:
        import time
        rng_seed = int(time.time()%1000000)
    if random:
        import random
        random.seed(rng_seed)
    if numpy:
        try:
            import numpy
            numpy.random.seed(rng_seed)
        except:
            pass
    if pytorch:
        try:
            import torch
            torch.manual_seed(rng_seed)
            torch.cuda.manual_seed_all(rng_seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
        except:
            pass
    return rng_seed