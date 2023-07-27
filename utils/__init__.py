from . import *
import torch
import numpy as np
import random
from datetime import timedelta
import accelerate

# import accelerate


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_accelerate(experiment, num_epochs, task_title, save_every):
    ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = accelerate.Accelerator(
        log_with=["wandb"],
        kwargs_handlers=[
            accelerate.InitProcessGroupKwargs(timeout=timedelta(hours=1.5)),
            ddp_kwargs,
        ],
    )
    accelerator.init_trackers(
        "experiment",
        config={
            "num_epochs": 1000,
            "task_title": task_title,
            "save_every": 20,
        },
    )

    return accelerator, ddp_kwargs
