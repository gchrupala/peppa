
import pig.data as D
from pig.triplet import PeppaTripletDataset
import logging
import torch
import random

random.seed(666)
torch.manual_seed(666)

logging.getLogger().setLevel(logging.INFO)

ds = D.PeppaPigIterableDataset(split=['val'], fragment_type='dialog',    duration=None, jitter=False)
PeppaTripletDataset.from_dataset(ds, "data/out/val_dialog_triplets_v4")
ds = D.PeppaPigIterableDataset(split=['val'], fragment_type='narration', duration=None, jitter=False)
PeppaTripletDataset.from_dataset(ds, "data/out/val_narration_triplets_v4")
ds = D.PeppaPigIterableDataset(split=['test'], fragment_type='narration', duration=None, jitter=False)
PeppaTripletDataset.from_dataset(ds, "data/out/test_narration_triplets_v4")
