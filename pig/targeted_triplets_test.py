import time

import numpy as np
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.editor import VideoFileClip

from pig.targeted_triplets import FPS, PeppaTargetedTripletCachedDataset
import pygame
import pandas as pd

AUDIO_SAMPLE_RATE = 44100  # Sampling rate in samples per second.


if __name__ == "__main__":
    fragment_type = "narration"
    pos = "NOUN"

    word = "car"

    eval_info_file = f"data/eval/eval_set_{fragment_type}_{pos}.csv"
    eval_info = pd.read_csv(eval_info_file, index_col="id")

    eval_dataset = PeppaTargetedTripletCachedDataset(fragment_type, pos)

    # assert len(eval_info) == len(eval_dataset)

    for i, info in eval_info.iterrows():
        if info["distractor_word"] == word: #and info["distractor_word"] == "george":
            # if "house" in info["transcript"]:
            print(i)
            print(info.id_counterexample)
            print(info["transcript"], end=" | ")
            print("Target: ", info["target_word"], end=" | ")
            print("Distractor: ", info["distractor_word"])

            # sample = eval_dataset.__getitem__(i)
            example = VideoFileClip(f"data/out/val_narration_targeted_triplets_NOUN/{i}.avi")
            counterexample = VideoFileClip(f"data/out/val_narration_targeted_triplets_NOUN/{info.id_counterexample}.avi")
                # video_data_pos = sample.positive
                # video_data_pos = video_data_pos.permute(1, 2, 3, 0) * 255
                # video_data_pos = list(video_data_pos.numpy())
                # video_clip_pos = ImageSequenceClip(video_data_pos, fps=FPS)
                #
                # video_data_neg = sample.negative
                # video_data_neg = video_data_neg.permute(1, 2, 3, 0) * 255
                # video_data_neg = list(video_data_neg.numpy())
                # video_clip_neg = ImageSequenceClip(video_data_neg, fps=FPS)
                #
                # audio_data = sample.anchor
                # audio_data = audio_data.permute(1, 0).numpy()
                #
                # audio_data = np.tile(audio_data, 2)     # Playing mono doesn't work
                #
                # audio_clip = AudioArrayClip(audio_data, fps=AUDIO_SAMPLE_RATE)
                #
                # video_clip_pos = video_clip_pos.set_audio(audio_clip)
            pygame.display.set_caption('Positive Sample')
            counterexample.preview(fps=FPS)

            time.sleep(1)

                # pygame.display.set_caption('Counter Sample')
                # counterexample.preview(fps=FPS)
                #
                # time.sleep(1)

                # 1708.avi
                # 1309.avi