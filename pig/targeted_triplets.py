import glob
import json
import logging
import pickle
from dataclasses import dataclass

import moviepy.editor as m
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from pig.data import featurize
from pig.util import pad_audio_batch, pad_video_batch

VIDEO_DURATION = 3.2
FPS = 10


@dataclass
class Triplet:
    anchor: torch.tensor
    positive: torch.tensor
    negative: torch.tensor
    video_duration: float
    audio_duration: float


@dataclass
class TripletBatch:
    anchor: torch.tensor
    positive: torch.tensor
    negative: torch.tensor


class PeppaTargetedTripletCachedDataset(Dataset):

    def __init__(self, fragment, pos, target_size=(180, 100), force_cache=False):
        self.cache_dir = f"data/out/items-targeted-triplets-{target_size[0]}-{target_size[1]}-{fragment}-{pos}/"
        if force_cache or not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            ds = PeppaTargetedTripletDataset.from_csv(fragment, pos, target_size)

            for i, item in enumerate(ds):
                logging.info(f"Caching item {self.cache_dir}/{i}.pt")
                torch.save(item, f"{self.cache_dir}/{i}.pt")

        self.length = len(glob.glob(f"{self.cache_dir}/*.pt"))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.load(f"{self.cache_dir}/{idx}.pt")


class PeppaTargetedTripletDataset(Dataset):

    def __init__(self, directory, target_size=(180, 100)):
        super().__init__()

        self.directory = directory
        self.target_size = target_size

    @classmethod
    def load(cls, directory, raw=False):
        self = cls(directory, raw=raw)
        self._clip_info = json.load(open(f"{self.directory}/clip_info.json"))
        self._sample = json.load(open(f"{self.directory}/sample.json"))
        return self

    @classmethod
    def from_csv(cls, fragment, pos, target_size=(180, 100)):
        directory = f"data/out/val_{fragment}_targeted_triplets_{pos}"
        self = cls(directory=directory, target_size=target_size)
        eval_set_info = pd.read_csv(f"data/eval/eval_set_{fragment}_{pos}.csv")

        self._load_eval_set_and_save_clip_info(eval_set_info)
        self._sample = list(self.sample())
        self._save_sample()
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

    def __getitem__(self, idx):
        target_info, distractor_info = self._sample[idx]
        with m.VideoFileClip(target_info['path']) as target:
            with m.VideoFileClip(distractor_info['path']) as distractor:
                positive = featurize(target)
                negative = featurize(distractor)
                return Triplet(anchor=positive.audio, positive=positive.video, negative=negative.video,
                               audio_duration=target.audio.duration, video_duration=target.duration)

    def __len__(self):
        return len(self._sample)

    def _load_eval_set_and_save_clip_info(self, eval_set_info):
        os.makedirs(self.directory, exist_ok=True)
        self._clip_info = {}

        for _, sample in eval_set_info.iterrows():
            id = sample.id
            id_counterexample = sample.id_counterexample

            clip = m.VideoFileClip(sample.episode_filepath)

            path_example = f"{self.directory}/{id}.avi"
            path_counterxample = f"{self.directory}/{id_counterexample}.avi"

            sample["audio_start"] = sample["clipStart"]
            sample["audio_end"] = sample["clipEnd"]

            clip_trimmed = clip.subclip(sample["clipStart"], sample["clipEnd"])

            clip_trimmed = clip_trimmed.resize(self.target_size)

            clip_trimmed.write_videofile(path_example, fps=FPS, codec='mpeg4')

            self._clip_info[id] = dict(path=path_example,
                                       path_counterexample=path_counterxample,
                                       transcript=sample['transcript'],
                                       target_word=sample['target_word'],
                                       distractor_word=sample['distractor_word'],
                                       id_counterexample=id_counterexample,
                                       filename=clip_trimmed.filename,
                                       audio_start=sample["audio_start"],
                                       audio_end=sample["audio_end"],
                                       duration=clip_trimmed.duration)

        json.dump(self._clip_info, open(f"{self.directory}/clip_info.json", "w"), indent=2)

    def sample(self):
        for info in _targeted_triplets(self._clip_info):
            yield info


def _targeted_triplets(clips_dict):
    clips = clips_dict.values()
    for item in clips:
        target, distractor = item, clips_dict[item["id_counterexample"]]
        yield (target, distractor)


def collate_triplets(data):
    anchor, pos, neg = zip(*[(x.anchor, x.positive, x.negative) for x in data])
    return TripletBatch(anchor=pad_audio_batch(anchor),
                        positive=pad_video_batch(pos),
                        negative=pad_video_batch(neg))
