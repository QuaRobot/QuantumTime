from typing import Union

import torch

from .DeepTIMe import deeptime
from .QuantumTIMe import quantumtime


def get_model(model_type: str, **kwargs: Union[int, float]) -> torch.nn.Module:
    if model_type == 'deeptime':
        model = deeptime(datetime_feats=kwargs['datetime_feats'])
    elif model_type == 'quantumtime':
        model = quantumtime(in_feats=kwargs['datetime_feats'])
    else:
        raise ValueError(f"Unknown model type {model_type}")
    return model
