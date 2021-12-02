import torch
import glob
from pig.models import PeppaPig
import pig.data
import pytorch_lightning as pl
import logging
from torch.utils.data import DataLoader
from dataclasses import dataclass
import pandas as pd

def load_best_model(dirname, higher_better=True):
    info = []
    for path in glob.glob(f"{dirname}/checkpoints/*.ckpt"):
       cp = torch.load(path, map_location='cpu')
       info.append(cp['callbacks'][pl.callbacks.model_checkpoint.ModelCheckpoint])
    best = sorted(info, key=lambda x: x['best_model_score'], reverse=higher_better)[0]
    logging.info(f"Best {best['monitor']}: {best['best_model_score']} at {best['best_model_path']}")
    net = PeppaPig.load_from_checkpoint(best['best_model_path'])
    return net, best['best_model_path']

def score(model, gpus):
    """Compute all standard scores for the given model. """
    trainer = pl.Trainer(gpus=gpus, logger=False)
    for fragment_type in ['dialog', 'narration']:
        yield dict(fragment_type=fragment_type,
                   triplet_acc=triplet_score(fragment_type, model, trainer),
                   recall_at_10=retrieval_score(fragment_type, model, trainer))
        
def retrieval_score(fragment_type, model, trainer):
        ds = pig.data.PeppaPigIterableDataset(
            target_size=(180, 100),
            split=['val'],
            fragment_type=fragment_type,
            duration=3.2)
        loader = DataLoader(ds, collate_fn=pig.data.collate, batch_size=8)

        V, A = zip(* [(batch.video, batch.audio) for batch
                  in trainer.predict(model, loader) ])
        V = torch.cat(V, dim=0)
        A = torch.cat(A, dim=0)
        correct = torch.eye(V.shape[0], device=A.device)
        rec10 = pig.metrics.recall_at_n(V, A, correct=correct, n=10).mean().item()
        return rec10
        
def triplet_score(fragment_type, model, trainer):
    ds = pig.data.PeppaTripletDataset.load(f"data/out/val_{fragment_type}_triplets_v2")
    loader = DataLoader(ds, collate_fn=pig.data.collate_triplets, batch_size=8)
    acc = torch.cat([ pig.metrics.batch_triplet_accuracy(batch)
                      for  batch in trainer.predict(model, loader) ]).mean().item()
    return acc



def main(gpu):
    logging.getLogger().setLevel(logging.INFO)
    rows = []
    for version in [38, 39, 41, 42]:
        logging.info(f"Evaluating version {version}")
        net, path = load_best_model(f"lightning_logs/version_{version}/")
        for row in score(net, gpus=[gpu]):
            row['version'] = version
            row['path']    = path
            print(row)
            rows.append(row)
    scores = pd.DataFrame.from_records(rows)
    scores.to_csv("results/scores.csv", index=False, header=True)
