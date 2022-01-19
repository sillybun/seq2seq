import torch
from zytlib import vector

def SOR(space1: torch.Tensor, space2: torch.Tensor) -> float:

    if len(space1.shape) == 1:
        space1 = torch.unsqueeze(space1, 1)

    if len(space2.shape) == 1:
        space2 = torch.unsqueeze(space2, 1)

    space1 = torch.nn.functional.normalize(space1, dim=0)
    space2 = torch.nn.functional.normalize(space2, dim=0)

    # P = space1 @ space1.T
    # Q = space2 @ space2.T

    trace_P_Q = torch.trace((space1.T @ space2) @ (space2.T @ space1))
    trace_P = torch.trace(space1.T @ space1)

    return trace_P_Q.item() / trace_P.item()
