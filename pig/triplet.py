import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import json
import moviepy.editor as m
import pig.util as U
from dataclasses import dataclass
import os
import random
from pig.util import grouped, shuffled
import pig.data
import glob
import pytorch_lightning as pl
from pig.metrics import triplet_accuracy


@dataclass
class Triplet:
    anchor: ...
    positive: ...
    negative: ...
    
    

@dataclass
class TripletBatch:
    anchor: ...
    positive: ...
    negative: ...

class TripletScorer:

    def __init__(self, fragment_type, split=['val']):
        self.dataset = pig.data.PeppaPigIterableDataset(
            target_size=(180, 100),
            split=split,
            fragment_type=fragment_type,
            duration=None,
        )


    def _encode(self, model, trainer, batch_size):
        key = lambda x: x.audio_duration
        loader = pig.data.grouped_loader(self.dataset, key, pig.data.collate, batch_size=batch_size)
        audio, video, duration =  zip(*[ (batch.audio, batch.video, batch.audio_duration) for batch
                                         in trainer.predict(model, loader) ])
        self._duration = torch.cat(duration)
        self._audio = torch.cat(audio)
        self._video = torch.cat(video)
 
        
    def _score(self, n_samples=100):
        return score_triplets(self._video, self._audio, self._duration, n_samples=n_samples)
    
    def evaluate(self, model, batch_size, n_samples=100, trainer=None):
        if trainer is None:
            trainer = pl.Trainer(gpus=1, logger=False)
        self._encode(model, trainer, batch_size)
        return self._score(n_samples=n_samples)

def score_triplets(video, audio, duration, n_samples=100):
    accuracy = []
    for i in range(n_samples):
        pos_idx, neg_idx = zip(*_triplets(range(len(duration)),
                                          lambda idx: duration[idx]))
        pos_idx = torch.tensor(pos_idx)
        neg_idx = torch.tensor(neg_idx)
        acc = triplet_accuracy(anchor=audio[pos_idx],
                                   positive=video[pos_idx],
                                   negative=video[neg_idx]).mean().item()
        accuracy.append(acc)
    return torch.tensor(accuracy)
    

def _triplets(clips, criterion): 
    for size, items in grouped(clips, key=criterion):
        paired = pairs(shuffled(items))
        for p in paired:
            target, distractor = random.sample(p, 2)
            yield (target, distractor)


def triplets(clips):
    """Generates triplets of (a, v1, v2) where a is an audio clip, v1
       matching video and v2 a distractor video, matched by duration."""
    items = _triplets(clips, lambda x: x.duration)
    for target, distractor in items:
        yield Triplet(anchor=target.audio, positive=target.video, negative=distractor.video)


def pairs(xs):
    p = []
    for i in range(0, len(xs), 2):
        x = xs[i:i+2]
        if len(x) == 2:
            p.append(x)
    return p
