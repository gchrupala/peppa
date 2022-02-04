import torch
import glob
from pig.models import PeppaPig
import pig.data
import pytorch_lightning as pl
import logging
from torch.utils.data import DataLoader
from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch
import random

random.seed(666)
torch.manual_seed(666)

BATCH_SIZE=8
VERSIONS = (48, 50, 51, 52)

def data_statistics():
    rows = []
    for split in ['train', 'val', 'test']:
        for fragment_type in ['dialog', 'narration']:
            if pig.data.SPLIT_SPEC[fragment_type][split] is not None:
                ds = pig.data.PeppaPigIterableDataset(
                    target_size=(180, 100),
                    split=[split],
                    fragment_type=fragment_type,
                    duration=3.2)
                duration = np.array([clip.duration for clip in ds._raw_clips() ])
                rows.append({'Split': split, 'Type': fragment_type, 'Triplet': 'No',
                             'Size (h)': duration.sum() / 60 / 60, 'Items': len(duration),
                             'Mean length (s)': duration.mean() })
                if split != 'train':
                    ds = pig.data.PeppaTripletDataset.load(f"data/out/{split}_{fragment_type}_triplets_v4")
                    duration = np.array([ val['duration'] for key, val in ds._clip_info.items() ])
                    rows.append({'Split': split, 'Type': fragment_type, 'Triplet': 'Yes',
                                 'Size (h)': duration.sum() / 60 / 60, 'Items': len(duration),
                                 'Mean length (s)': duration.mean() })
    data = pd.DataFrame.from_records(rows).sort_values(by="Triplet")
    data.to_csv("results/data_statistics.csv", index=False, header=True)
    data.to_latex("results/data_statistics.tex", index=False, header=True, float_format="%.2f")
    

def load_best_model(dirname, higher_better=True):
    info = []
    for path in glob.glob(f"{dirname}/checkpoints/*.ckpt"):
       cp = torch.load(path, map_location='cpu')
       item = cp['callbacks'][pl.callbacks.model_checkpoint.ModelCheckpoint]
       if item['best_model_score'] is not None:
           info.append(item)
    best = sorted(info, key=lambda x: x['best_model_score'], reverse=higher_better)[0]
    logging.info(f"Best {best['monitor']}: {best['best_model_score']} at {best['best_model_path']}")
    local_model_path = best['best_model_path'].split("/peppa/")[1]
    net = PeppaPig.load_from_checkpoint(local_model_path)
    return net, best['best_model_path']

def score(model, gpus):
    """Compute all standard scores for the given model. """
    trainer = pl.Trainer(gpus=gpus, logger=False, precision=16)
    for fragment_type in ['dialog', 'narration']:
        logging.info(f"Evaluating: {fragment_type}, triplet")
        acc = triplet_score(fragment_type, model, trainer)
        logging.info(f"Evaluating: {fragment_type}, recall_fixed")
        rec_fixed = resampled_retrieval_score(fragment_type,
                                    model,
                                    trainer,
                                    duration=2.3,
                                    jitter=False,
                                    jitter_sd=None)
        logging.info(f"Evaluating: {fragment_type}, recall_jitter")
        rec_jitter = resampled_retrieval_score(fragment_type,
                                     model,
                                     trainer,
                                     duration=2.3,
                                     jitter=True,
                                     jitter_sd=0.5)
        yield dict(fragment_type=fragment_type,
                   triplet_acc=acc.mean().item(),
                   triplet_acc_std=acc.std().item(),
                   recall_at_10_fixed=rec_fixed.mean(dim=1).mean().item(),
                   recall_at_10_fixed_std=rec_fixed.mean(dim=1).std().item(),
                   recall_at_10_jitter=rec_jitter.mean(dim=1).mean().item(),
                   recall_at_10_jitter_std=rec_jitter.mean(dim=1).std().item())

def full_score(model, gpus):
    """Compute all standard scores for the given model. """
    trainer = pl.Trainer(gpus=gpus, logger=False, precision=16)
    data = []
    for fragment_type in ['dialog', 'narration']:
        logging.info(f"Evaluating: {fragment_type}, triplet")
        acc = triplet_score(fragment_type, model, trainer)
        logging.info(f"Evaluating: {fragment_type}, recall_fixed")
        rec_fixed = resampled_retrieval_score(fragment_type,
                                    model,
                                    trainer,
                                    duration=2.3,
                                    jitter=False,
                                    jitter_sd=None)
        logging.info(f"Evaluating: {fragment_type}, recall_jitter")
        rec_jitter = resampled_retrieval_score(fragment_type,
                                     model,
                                     trainer,
                                     duration=2.3,
                                     jitter=True,
                                     jitter_sd=0.5)
        data.append(dict(fragment_type=fragment_type,
                         triplet_acc=acc,
                         recall_at_10_fixed=rec_fixed,
                         recall_at_10_jitter=rec_jitter))
    return data
        
def retrieval_score(fragment_type, model, trainer, duration=2.3, jitter=False, jitter_sd=None, batch_size=BATCH_SIZE):
        base_ds = pig.data.PeppaPigDataset(
            target_size=model.config["data"]["target_size"],
            split=['val'],
            fragment_type=fragment_type,
            duration=duration,
            jitter=jitter,
            jitter_sd=jitter_sd
            )
        key = lambda x: x.audio_duration
        loader = pig.data.grouped_loader(base_ds, key, pig.data.collate, batch_size=batch_size)
        V, A = zip(* [(batch.video, batch.audio) for batch
                  in trainer.predict(model, loader) ])
        V = torch.cat(V, dim=0)
        A = torch.cat(A, dim=0)
        correct = torch.eye(V.shape[0], device=A.device)
        rec10 = pig.metrics.recall_at_n(V, A, correct=correct, n=10).mean().item()
        return rec10

def resampled_retrieval_score(fragment_type,
                              model,
                              trainer,
                              duration=2.3,
                              jitter=False,
                              jitter_sd=None,
                              batch_size=BATCH_SIZE):
        base_ds = pig.data.PeppaPigDataset(
            target_size=model.config["data"]["target_size"],
            split=['val'],
            fragment_type=fragment_type,
            duration=duration,
            jitter=jitter,
            jitter_sd=jitter_sd
            )
        key = lambda x: x.audio_duration
        loader = pig.data.grouped_loader(base_ds, key, pig.data.collate, batch_size=batch_size)
        V, A = zip(* [(batch.video, batch.audio) for batch
                  in trainer.predict(model, loader) ])
        V = torch.cat(V, dim=0)
        A = torch.cat(A, dim=0)
        rec10 = pig.metrics.resampled_recall(V, A, size=100, n_samples=100, n=10)
        return rec10

def triplet_score(fragment_type, model, trainer, batch_size=BATCH_SIZE):
    from pig.triplet import TripletScorer
    scorer = TripletScorer(fragment_type=fragment_type, split=['val'], target_size=model.config["data"]["target_size"])
    acc = scorer.evaluate(model, trainer=trainer, n_samples=500, batch_size=batch_size)
    return acc


def pretraining(row):
    return { (True, True): "AV",
             (True, False): "A",
             (False, True): "V",
             (False, False): "None"}[row['audio_pretrained'],
                                     row['video_pretrained']]

def format():
    data = add_condition(pd.read_csv("results/scores.csv"))
    for fragment_type in ['dialog', 'narration']:
        table = data.query(f"fragment_type=='{fragment_type}'")
        table['pretraining'] = table.apply(pretraining, axis=1)
        


        table[['version', 'static', 'jitter', 'pretraining', 'resolution',
               'recall_at_10_fixed', 'recall_at_10_jitter', 'triplet_acc']]\
            .replace(True, "Yes").replace(False, "")\
            .rename(columns=dict(version='ID',
                                 static='Static',
                                 jitter='Jitter',
                                 pretraining='Pretraining',
                                 resolution='Resolution',
                                 recall_at_10_fixed='R@10 (fixed)',
                                 recall_at_10_jitter='R@10 (jitter)',
                                 triplet_acc='Triplet Acc'))\
            .to_latex(buf=f"results/scores_{fragment_type}.tex",
                      index=False,
                      float_format="%.3f")
                                

def add_condition(data):
    records = data.to_dict(orient='records')
    for row in records:
        config = yaml.safe_load(open(row['hparams_path']))
        row['jitter'] = config['data']['train']['jitter']
        row['static'] = config['video'].get('static', False)
        row['audio_pretrained'] = config['audio']['pretrained']
        row['video_pretrained'] = config['video']['pretrained']
        row['resolution'] = 'x'.join(map(str, config['data']['target_size']))
    return pd.DataFrame.from_records(records)
    
def run(gpu=0, versions=VERSIONS):
    logging.getLogger().setLevel(logging.INFO)
    rows = []
    for version in versions:
        logging.info(f"Evaluating version {version}")
        net, path = load_best_model(f"lightning_logs/version_{version}/")
        for row in score(net, gpus=[gpu]):
            row['version']         = version
            row['checkpoint_path'] = path
            row['hparams_path']    = f"lightning_logs/version_{version}/hparams.yaml"
            rows.append(row)
    scores = pd.DataFrame.from_records(rows)
    return scores

def full_run(gpu=0, versions=VERSIONS):
    logging.getLogger().setLevel(logging.INFO)
    rows = []
    for version in versions:
        logging.info(f"Evaluating version {version}")
        net, path = load_best_model(f"lightning_logs/version_{version}/")
        for row in full_score(net, gpus=[gpu]):
            row['version']         = version
            row['checkpoint_path'] = path
            row['hparams_path']    = f"lightning_logs/version_{version}/hparams.yaml"
            rows.append(row)
    torch.save(rows, "results/full_scores.pt")
    
def main(gpu=0, versions=VERSIONS):
    scores = run(gpu=gpu, versions=versions)
    scores.to_csv("results/scores.csv", index=False, header=True)
    
    

