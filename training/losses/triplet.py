import torch
import torch.nn.functional as F

# Define the cosine distance function
def cosine_distance(x: torch.Tensor, y: torch.Tensor, eps=1e-8):
    # x,y: (B,D)
    # The output is 1 - cos sim in [0, 2]
    return 1 - F.cosine_similarity(x, y, dim=-1, eps=eps)