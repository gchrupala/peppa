import torch
import torch.nn.functional as F
from itertools import groupby
import random

def identity(x):
    return x

def cosine_matrix(U, V):
    "Returns the matrix of cosine similarity between each row of U and each row of V."
    U_norm = U / U.norm(2, dim=1, keepdim=True)
    V_norm = V / V.norm(2, dim=1, keepdim=True)
    return torch.matmul(U_norm, V_norm.t())
        
def crop_audio_batch(audio):
    size = min(x.shape[1] for x in audio)
    return torch.stack([ x[:, :size] for x in audio ])

def pad_audio_batch(audio):
    size = max(x.shape[1] for x in audio)
    return torch.stack([ F.pad(x, (0, size-x.shape[1]), 'constant', 0) for x in audio ])

def crop_video_batch(video):
    size = min(x.shape[1] for x in video)
    return torch.stack([ x[:, :size, :, :] for x in video ])

def pad_video_batch(video):
    size = max(x.shape[1] for x in video)
    return torch.stack([ F.pad(x, (0,0, 0,0, 0,size-x.shape[1]), 'constant', 0) for x in video ])

def shuffled(xs):
    return sorted(xs, key=lambda _: random.random())

def grouped(xs, key=lambda x: x):
    return groupby(sorted(xs, key=key), key=key)
                 
            
def triu(x):
    "Extracts upper triangular part of a matrix, excluding the diagonal."
    ones  = torch.ones_like(x)
    return x[torch.triu(ones, diagonal=1) == 1]

def pearson_r(x, y, dim=0, eps=1e-8):
    "Returns Pearson's correlation coefficient."
    x1 = x - torch.mean(x, dim)
    x2 = y - torch.mean(y, dim)
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return w12 / (w1 * w2).clamp(min=eps)
