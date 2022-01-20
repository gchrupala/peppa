import json
import moviepy.editor as m
import os
import pandas as pd

from pig.data import featurize
from pig.triplet import PeppaTripletDataset, Triplet

VIDEO_DURATION = 3.2
FPS = 10


class PeppaTargetedTripletDataset(PeppaTripletDataset):

    def __init__(self, directory, min_samples=100, target_size=(180, 100), raw=False):
        super().__init__(raw)

        self.directory = directory
        self.target_size = target_size
        self.min_samples = min_samples

    @classmethod
    def load(cls, directory, raw=False):
        self = cls(directory, raw=raw)
        self._clip_info = json.load(open(f"{self.directory}/clip_info.json"))
        self._sample = json.load(open(f"{self.directory}/sample.json"))
        return self

    @classmethod
    def from_csv(cls, directory, targeted_eval_set_csv, min_samples=100, target_size=(180, 100), raw=False):
        self = cls(directory, raw=raw, target_size=target_size, min_samples=min_samples)
        eval_set_info = pd.read_csv(targeted_eval_set_csv)

        self._load_eval_set_and_save_clip_info(eval_set_info)
        self._sample = list(self.sample())
        self._save_sample()
        return self

    def _load_eval_set_and_save_clip_info(self, eval_set_info):
        os.makedirs(self.directory, exist_ok=True)

        # Filter examples by num samples
        counts = eval_set_info.target_word.value_counts()
        words_enough_samples = counts[counts > self.min_samples].keys().to_list()
        if len(words_enough_samples) == 0:
            print(f"No words with enough samples (>{self.min_samples}) found.")
            return

        eval_set_info = eval_set_info[eval_set_info.target_word.isin(words_enough_samples) | eval_set_info.distractor_word.isin(words_enough_samples)]

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

