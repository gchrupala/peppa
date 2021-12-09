import torch

from pig.evaluation import load_best_model, pretraining
import pig.data
import pytorch_lightning as pl
import logging
from torch.utils.data import DataLoader
import pandas as pd

from pig.targeted_triplets import PeppaTargetedTripletDataset

BATCH_SIZE = 8


def score(model):
    """Compute all standard scores for the given model. """
    auto_select_gpus = False
    if torch.cuda.is_available():
        auto_select_gpus = True
    trainer = pl.Trainer(logger=False, auto_select_gpus=auto_select_gpus)
    for fragment_type in ['dialog', 'narration']:
        for pos in ["ADJ", "VERB", "NOUN"]:
            yield dict(fragment_type=fragment_type,
                       pos=pos,
                       targeted_triplet_acc=targeted_triplet_score(fragment_type, pos, model, trainer)
                       )
        

def targeted_triplet_score(fragment_type, pos, model, trainer):
    ds = PeppaTargetedTripletDataset.load(f"data/out/val_{fragment_type}_targeted_triplets_{pos}")
    loader = DataLoader(ds, collate_fn=pig.data.collate_triplets, batch_size=BATCH_SIZE)
    acc = torch.cat([pig.metrics.batch_triplet_accuracy(batch)
                      for batch in trainer.predict(model, loader)]).mean().item()
    return acc


def format_results():
    data = pd.read_csv("results/scores_targeted_triplets.csv")
    for fragment_type in ['dialog', 'narration']:
        table = data.query(f"fragment_type=='{fragment_type}'")
        table['pretraining'] = table.apply(pretraining, axis=1)

        table[['version', 'pretraining',
               'pos', 'targeted_triplet_acc']]\
            .rename(columns=dict(version='ID',
                                 pretraining='Pretraining',
                                 targeted_triplet_acc='Targeted Triplet Acc',
                                 pos='POS'))\
            .to_latex(buf=f"results/scores_{fragment_type}.tex",
                      index=False,
                      float_format="%.3f")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    rows = []
    for version in [43]: # TODO: evaluate all versions: [43, 44, 45]:
        logging.info(f"Evaluating version {version}")
        net, path = load_best_model(f"lightning_logs/version_{version}/")
        
        for row in score(net):
            row['version'] = version
            row['path']    = path
            row['audio_pretrained'] = net.config['audio']['pretrained']
            row['video_pretrained'] = net.config['video']['pretrained']
            row['audio_pooling'] = net.config['audio']['pooling']
            row['video_pooling'] = net.config['video']['pooling']
            print(row)
            rows.append(row)
    scores = pd.DataFrame.from_records(rows)
    scores.to_csv("results/scores_targeted_triplets.csv", index=False, header=True)
    format_results()
