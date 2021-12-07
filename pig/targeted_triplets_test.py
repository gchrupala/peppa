import time

import numpy as np
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from torch.utils.data import DataLoader

from pig.targeted_triplets import PeppaTargetedTripletDataset, FPS
import pygame

from pig.triplet import collate_triplets

AUDIO_SAMPLE_RATE = 44100  # Sampling rate in samples per second.


if __name__ == "__main__":
    fragment_type = "narration"
    pos = "ADJ"

    eval_dataset = PeppaTargetedTripletDataset.load(f"data/out/val_{fragment_type}_targeted_triplets_{pos}")
    loader = DataLoader(eval_dataset, collate_fn=collate_triplets, batch_size=1)

    for batch in loader:
        video_data_pos = batch.positive
        video_data_pos = video_data_pos.squeeze(0).permute(1, 2, 3, 0) * 255
        video_data_pos = list(video_data_pos.numpy())
        video_clip_pos = ImageSequenceClip(video_data_pos, fps=FPS)

        video_data_neg = batch.negative
        video_data_neg = video_data_neg.squeeze(0).permute(1, 2, 3, 0) * 255
        video_data_neg = list(video_data_neg.numpy())
        video_clip_neg = ImageSequenceClip(video_data_neg, fps=FPS)

        audio_data = batch.anchor
        audio_data = audio_data.squeeze(0).permute(1, 0).numpy()

        audio_data = np.tile(audio_data, 2)     # Playing mono doesn't work

        audio_clip = AudioArrayClip(audio_data, fps=AUDIO_SAMPLE_RATE)

        video_clip_pos = video_clip_pos.set_audio(audio_clip)
        pygame.display.set_caption('Positive Sample')

        video_clip_pos.preview(fps=FPS)

        time.sleep(1)
