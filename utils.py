import torch
from zytlib import vector

def SOR(space1: torch.Tensor, space2: torch.Tensor) -> float:

    P = space1 @ space1.T
    Q = space2 @ space2.T

    return torch.trace(P @ Q).item() / torch.trace(P).item()
