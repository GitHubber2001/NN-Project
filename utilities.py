import random

import numpy as np
import torch


def set_deterministic_behaviour(random_seed):
    """Sets deterministic behaviour of the program"""

    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_accelerator_device() -> str:
    """Returns accelerator device for boosting performance"""

    current_accelerator = torch.accelerator.current_accelerator(True)
    if current_accelerator is not None:
        device = current_accelerator.type
    else:
        device = "cpu"

    print(f"Using device {device}")

    return device
