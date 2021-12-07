import json
import moviepy.editor as m
import os
import pandas as pd

from pig.data import featurize
from pig.triplet import PeppaTripletDataset, Triplet

VIDEO_DURATION = 3.2
FPS = 10


class PeppaTargetedTripletDataset(PeppaTripletDataset):

    def __init__(self, long_videos=False, raw=False):
        super().__init__(raw)

        self.long_videos = long_videos

    @classmethod
    def load(cls, directory, raw=False):
        self = cls(raw=raw)
        self.directory = directory
        self._clip_info = json.load(open(f"{self.directory}/clip_info.json"))
        self._sample = json.load(open(f"{self.directory}/sample.json"))
        return self

    @classmethod
    def from_csv(cls, directory, targeted_eval_set_csv, raw=False):
        self = cls(raw=raw)
        eval_set_info = pd.read_csv(targeted_eval_set_csv)

        self.directory = directory
        self._load_eval_set_and_save_clip_info(eval_set_info)
        self._sample = list(self.sample())
        self._save_sample()
        return self

    def _load_eval_set_and_save_clip_info(self, eval_set_info, target_size=(180, 100)):
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

            if self.long_videos:
                clip_duration = sample["clipEnd"] - sample["clipStart"]
                clip_time_center = sample["clipStart"] + clip_duration / 2
                sample["video_start"] = clip_time_center - VIDEO_DURATION / 2
                sample["video_end"] = clip_time_center + VIDEO_DURATION / 2

                sample["audio_start"] = sample["clipStart"] - sample["video_start"]
                sample["audio_end"] = sample["clipEnd"] - sample["video_start"]

                clip_trimmed = clip.subclip(sample["video_start"], sample["video_end"])

            else:
                clip_trimmed = clip.subclip(sample["clipStart"], sample["clipEnd"])

            clip_trimmed = clip_trimmed.resize(target_size)

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

    def __getitem__(self, idx):
        target_info, distractor_info = self._sample[idx]
        with m.VideoFileClip(target_info['path']) as target:
            with m.VideoFileClip(distractor_info['path']) as distractor:
                if self.long_videos:
                    # Clip audio data to correct size
                    target.audio = target.audio.subclip(target_info["audio_start"], target_info["audio_end"]).copy()
                    distractor.audio = distractor.audio.subclip(distractor_info["audio_start"], distractor_info["audio_end"]).copy()

                if self.raw:
                    return Triplet(anchor=target.audio, positive=target, negative=distractor)
                else:
                    positive = featurize(target)
                    negative = featurize(distractor)
                    return Triplet(anchor=positive.audio, positive=positive.video, negative=negative.video)


def _targeted_triplets(clips_dict):
    clips = clips_dict.values()
    for item in clips:
        target, distractor = item, clips_dict[item["id_counterexample"]]
        yield (target, distractor)

