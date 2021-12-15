import time

import numpy as np
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from pig.targeted_triplets import PeppaTargetedTripletDataset, FPS
import pygame
import pandas as pd

AUDIO_SAMPLE_RATE = 44100  # Sampling rate in samples per second.


if __name__ == "__main__":
    fragment_type = "narration"
    pos = "NOUN"

    target_word = "glass"

    eval_info_file = f"data/eval/eval_set_{fragment_type}_{pos}.csv"
    eval_info = pd.read_csv(eval_info_file, index_col="id")

    eval_dataset = PeppaTargetedTripletDataset.load(f"data/out/val_{fragment_type}_targeted_triplets_{pos}")

    assert len(eval_info) == len(eval_dataset)

    for i, info in eval_info.iterrows():
        if info["target_word"] == target_word:
            print(eval_info.iloc[i]["transcript"], end=" | ")
            print("Target: ", eval_info.iloc[i]["target_word"], end=" | ")
            print("Distractor: ", eval_info.iloc[i]["distractor_word"])

            sample = eval_dataset.__getitem__(i)

            video_data_pos = sample.positive
            video_data_pos = video_data_pos.permute(1, 2, 3, 0) * 255
            video_data_pos = list(video_data_pos.numpy())
            video_clip_pos = ImageSequenceClip(video_data_pos, fps=FPS)

            video_data_neg = sample.negative
            video_data_neg = video_data_neg.permute(1, 2, 3, 0) * 255
            video_data_neg = list(video_data_neg.numpy())
            video_clip_neg = ImageSequenceClip(video_data_neg, fps=FPS)

            audio_data = sample.anchor
            audio_data = audio_data.permute(1, 0).numpy()

            audio_data = np.tile(audio_data, 2)     # Playing mono doesn't work

            audio_clip = AudioArrayClip(audio_data, fps=AUDIO_SAMPLE_RATE)

            video_clip_pos = video_clip_pos.set_audio(audio_clip)
            pygame.display.set_caption('Positive Sample')

            video_clip_pos.preview(fps=FPS)

            time.sleep(1)
