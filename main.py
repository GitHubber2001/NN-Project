"""
Kevin Kuipers (s5051150)
Federico Berdugo Morales (s5363268)
Chong Xin Le (S6865615)

Some utility functions like set_deterministic_behaviour and get_accelerator_device
reused from the 2 members, Kevin and Federico, their repository
for a project for the NLP course 2025-2026.
"""

import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from preprocess import load_dataset

RANDOM_SEED = 42


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


def main():
    start_time = time.perf_counter()

    set_deterministic_behaviour(RANDOM_SEED)
    device = get_accelerator_device()

    trajectories, dt, E0 = load_dataset()

    print(trajectories, dt, E0)

    end_time = time.perf_counter()
    time_elapsed = end_time - start_time

    print(f"Done (execution duration: {time_elapsed})")


if __name__ == "__main__":
    main()
