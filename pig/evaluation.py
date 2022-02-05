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
import yaml
from copy import deepcopy

random.seed(666)
torch.manual_seed(666)

BATCH_SIZE=8
VERSIONS = (206964, 68, 206974, 206975, 206976, 206977, 206978)

def data_statistics():
    rows = []
    for split in ['train', 'val', 'test']:
        for fragment_type in ['dialog', 'narration']:
            if pig.data.SPLIT_SPEC[fragment_type][split] is not None:
                ds = pig.data.PeppaPigIterableDataset(
                    target_size=(180, 100),
                    split=[split],
                    fragment_type=fragment_type,
                    duration=2.3)
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

def score_means(data):
    rows = []
    for item in data:
        row = deepcopy(item)
        row['triplet_acc_std'] = row['triplet_acc'].std().item()
        row['triplet_acc'] = row['triplet_acc'].mean().item()
        row['recall_at_10_fixed_std'] = row['recall_at_10_fixed'].mean(dim=1).std().item()
        row['recall_at_10_fixed'] = row['recall_at_10_fixed'].mean(dim=1).mean().item()
        row['recall_at_10_jitter_std'] = row['recall_at_10_jitter'].mean(dim=1).std().item()
        row['recall_at_10_jitter'] = row['recall_at_10_jitter'].mean(dim=1).mean().item()
        rows.append(row)
    return pd.DataFrame.from_records(rows)

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
        rec10 = pig.metrics.resampled_recall(V, A, size=100, n_samples=500, n=10)
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
    data = torch.load("results/full_scores.pt")
    data = add_condition(data)
    data = score_means(data)
    for fragment_type in ['dialog', 'narration']:
        table = data.query(f"fragment_type=='{fragment_type}'")
        table['pretraining'] = pd.Categorical(table.apply(pretraining, axis=1),
                                              categories=['AV', 'A', 'V', 'None'])


        table[['version', 'static', 'jitter', 'pretraining', 'resolution',
               'recall_at_10_fixed', 'recall_at_10_jitter', 'triplet_acc']]\
            .sort_values(by=['resolution', 'static', 'jitter', 'pretraining'])\
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
    rows = []
    for row in data:
        record = {k:v for k,v in row.items()}
        config = yaml.safe_load(open(row['hparams_path']))
        record['jitter'] = config['data']['train']['jitter']
        record['static'] = config['video'].get('static', False)
        record['audio_pretrained'] = config['audio']['pretrained']
        record['video_pretrained'] = config['video']['pretrained']
        record['resolution'] = 'x'.join(map(str, config['data']['target_size']))
        rows.append(record)
    return rows
    
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
    torch.save(add_condition(rows), "results/full_scores.pt")
    
def main(gpu=0, versions=VERSIONS):
    scores = run(gpu=gpu, versions=versions)
    scores.to_csv("results/scores.csv", index=False, header=True)
    
    

