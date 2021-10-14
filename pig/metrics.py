import torch
from pig.util import cosine_matrix
import torch.nn.functional as F

def recall_at_n(candidates, references, correct, n=1):
    distances = 1-cosine_matrix(references, candidates)
    recall = []
    for j, row in enumerate(distances):
        ranked = row.argsort()
        match = ranked[ torch.nonzero(correct[j][ranked])[:,0] ]
        topn = ranked[:n]
        overlap = (topn.unsqueeze(dim=0) == match.unsqueeze(dim=1)).sum().item()
        recall.append(overlap/len(match))
    return torch.tensor(recall)
        
    

def triplet_accuracy(anchor, positive, negative):
    sim_pos = F.cosine_similarity(anchor, positive)
    sim_neg = F.cosine_similarity(anchor, negative)
    return (torch.sign(sim_pos - sim_neg)+1) / 2
