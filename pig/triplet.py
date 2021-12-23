from torch.utils.data import Dataset
import pickle
import json
import moviepy.editor as m
import pig.util as U
from dataclasses import dataclass
import os
import random
from pig.util import grouped, shuffled

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


class PeppaTripletDataset(Dataset):

    def __init__(self, raw=False):
        self.raw = raw
        
    @classmethod
    def from_dataset(cls, dataset, directory, raw=False):
        self = cls(raw=raw)
        self.directory = directory
        self._dataset = dataset
        self._save_clip_info()
        self._sample = list(self.sample())
        self._save_sample()
        return self
    
    @classmethod
    def load(cls, directory, raw=False):
        self = cls(raw=raw)
        self.directory = directory
        self._dataset = pickle.load(open(f"{self.directory}/dataset.pkl", "rb"))
        self._clip_info = json.load(open(f"{self.directory}/clip_info.json"))
        self._sample = json.load(open(f"{self.directory}/sample.json"))
        return self
    
    def save(self):
        self._save_clip_info()
        self._save_sample()
        
    def _save_clip_info(self):
        os.makedirs(self.directory, exist_ok=True)
        pickle.dump(self._dataset, open(f"{self.directory}/dataset.pkl", "wb"))
        self._clip_info = {}
        for i, clip in enumerate(self._dataset._raw_clips()):
            if clip.duration > 0:
                self._clip_info[i] = dict(path=f"{self.directory}/{i}.mp4",
                                          filename=clip.filename,
                                          offset=clip.offset,
                                          duration=clip.duration)
                clip.write_videofile(f"{self.directory}/{i}.mp4")
        json.dump(self._clip_info, open(f"{self.directory}/clip_info.json", "w"), indent=2)

    def _save_sample(self):
        json.dump(self._sample, open(f"{self.directory}/sample.json", "w"), indent=2)
        
    def sample(self):
        for info in _triplets(self._clip_info.values(), lambda x: x['duration']):
            yield info

    def __getitem__(self, idx):
        target_info, distractor_info = self._sample[idx]
        with m.VideoFileClip(target_info['path']) as target:
            with m.VideoFileClip(distractor_info['path']) as distractor:
                if self.raw:
                    return Triplet(anchor=target.audio, positive=target, negative=distractor)
                else:
                    positive = self._dataset.featurize(target)
                    negative = self._dataset.featurize(distractor)
                    return Triplet(anchor=positive.audio, positive=positive.video, negative=negative.video)
                   
    def __len__(self):
        return len(self._sample)


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


def collate_triplets(data):
    anchor, pos, neg = zip(*[(x.anchor, x.positive, x.negative) for x in data])
    return TripletBatch(anchor=U.pad_audio_batch(anchor),
                        positive=U.pad_video_batch(pos),
                        negative=U.pad_video_batch(neg))


def pairs(xs):
    p = []
    for i in range(0, len(xs), 2):
        x = xs[i:i+2]
        if len(x) == 2:
            p.append(x)
    return p
