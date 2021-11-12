
import pig.data as D
import logging
import torch
import random

random.seed(666)
torch.manual_seed(666)

logging.getLogger().setLevel(logging.INFO)

ds._prepare_triplets()


# ds = D.PeppaPigDataset(cache=True, cache_dir="data/out/val_dialog_triplets", triplet=True, split=['val'], fragment_type='dialog', duration=None, jitter=False)
# for x in ds:
#     pass
# ds = D.PeppaPigDataset(cache=True, cache_dir="data/out/val_narration_triplets", triplet=True, split=['val'], fragment_type='narration', duration=None, jitter=False)
# for x in ds:
#     pass
