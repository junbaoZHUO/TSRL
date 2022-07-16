import torch
import numpy as np


def uniform(shape, scale=0.05):
    """Uniform init."""
    initial = torch.Tensor(shape[0], shape[1]).uniform_(-scale, scale)
    return initial


def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = torch.Tensor(shape[0], shape[1]).uniform_(-init_range, init_range)
    return initial


def zeros(shape):
    """All zeros."""
    initial = torch.zeros(shape[0], shape[1])
    return initial


def ones():
    """All ones."""
    initial = torch.ones(shape[0], shape[1])
    return initial

