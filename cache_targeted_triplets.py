import logging
import torch
import random

from pig.targeted_triplets import PeppaTargetedTripletDataset

random.seed(666)
torch.manual_seed(666)

logging.getLogger().setLevel(logging.INFO)

for pos in ["ADJ", "NOUN", "VERB"]:
    for fragment in ["dialog", "narration"]:
        PeppaTargetedTripletDataset.from_csv(f"data/out/val_{fragment}_targeted_triplets_{pos}", f"data/eval/eval_set_{fragment}_{pos}.csv")
