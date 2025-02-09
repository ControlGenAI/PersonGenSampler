import random

import numpy as np

import torch

from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_seed(prompt, seed=0):
    return sum(ord(c) for c in prompt) + seed


def fix_seed(prompt, s=0, device='cuda'):
    seed = get_seed(prompt, seed=s)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    generator = torch.Generator(device=device)
    generator = generator.manual_seed(seed)
    return generator
