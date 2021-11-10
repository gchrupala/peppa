import torch
import pig.models as M
import pig.data as D
from pig.metrics import triplet_accuracy
import pytorch_lightning as pl
import random
random.seed(123)
torch.manual_seed(123)

ds = D.PeppaPigDataset(cache=True, triplet=True, split=['val'], fragment_type='dialog', duration=None, jitter=False)

net = M.PeppaPig.load_from_checkpoint("lightning_logs/version_1/checkpoints/epoch=40-step=7092.ckpt")
trainer = pl.Trainer(gpus=[0], logger=False)


# 1
loader = D.DataLoader(ds, collate_fn=D.collate_triplets, batch_size=8, shuffle=False)
encoded = trainer.predict(net, loader)
acc = torch.cat([ triplet_accuracy(b.anchor, b.positive, b.negative)
                  for b in encoded ])






        
