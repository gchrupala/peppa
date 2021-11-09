import torch
import pig.models as M
import pig.data as D
from pig.metrics import triplet_accuracy
import pytorch_lightning as pl

ds = D.PeppaPigIterableDataset(triplet=True, split=['val'],
                               fragment_type='dialog', duration=None)
loader = D.DataLoader(ds, collate_fn=D.collate_triplets, batch_size=8)

net = M.PeppaPig.load_from_checkpoint("lightning_logs/version_1/checkpoints/epoch=40-step=7092.ckpt")
trainer = pl.Trainer(gpus=[0])

acc = torch.cat([ triplet_accuracy(b.anchor, b.positive, b.negative)
                  for b in trainer.predict(net, loader) ])

print(f"Triplet accuracy on {len(acc)} dialog triplets: {acc.mean().item()}")
