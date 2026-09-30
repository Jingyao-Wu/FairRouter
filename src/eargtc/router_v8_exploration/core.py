"""Array and target validation for routing estimators."""
import numpy as np
import torch
HEADS = ('trust', 'preference')

def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

def binary_target(state, head):
    z = array(state)
    if head not in HEADS:
        raise ValueError('unknown head')
    mask = np.ones(len(z), bool) if head == 'trust' else z != 2
    return (mask, (z[mask] != 2 if head == 'trust' else z[mask] == 0).astype(float))
