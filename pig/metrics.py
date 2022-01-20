import torch
from pig.util import cosine_matrix
import torch.nn.functional as F
import logging

def recall_at_n(candidates, references, correct, n=1):
    distances = 1-cosine_matrix(references, candidates)
    recall = []
    for j, row in enumerate(distances):
        # ids ordered by distance
        ranked = row.argsort()
        # top N ids
        topn = ranked[:n]
        # target ids
        target = torch.nonzero(correct[j])[:,0]
        # overlap between top N and target
        overlap = (topn.unsqueeze(dim=0) == target.unsqueeze(dim=1)).sum().item()
        # proportion of correctly retrieved to target
        recall.append(overlap/len(target))
    return torch.tensor(recall)

def batch_triplet_accuracy(batch):
    return triplet_accuracy(batch.anchor, batch.positive, batch.negative)

def triplet_accuracy(anchor, positive, negative, dim=1):
    sim_pos = F.cosine_similarity(anchor, positive, dim=dim)
    sim_neg = F.cosine_similarity(anchor, negative, dim=dim)
    return (torch.sign(sim_pos - sim_neg)+1) / 2
