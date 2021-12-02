import numpy as np
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from torch.utils.data import DataLoader

import pig.data
import pygame

from pig.triplet import PeppaTripletDataset
from pig.util import identity

FPS = 10
AUDIO_SAMPLE_RATE = 44100  # Sampling rate in samples per second.

if __name__ == "__main__":
    fragment_type = "narration"

    ds = PeppaTripletDataset.load("data/out/val_narration_triplets_v2")

    # Do not normalize image data to keep it readable
    ds.transform = identity

    loader = DataLoader(ds, collate_fn=pig.data.collate_triplets, batch_size=1)

    for batch in loader:
        video_data_pos = batch.positive
        video_data_pos = video_data_pos.squeeze(0).permute(1, 2, 3, 0) * 255
        video_data_pos = list(video_data_pos.numpy())
        video_clip_pos = ImageSequenceClip(video_data_pos, fps=FPS)

        audio_data = batch.anchor
        audio_data = audio_data.squeeze(0).permute(1, 0).numpy()

        # Playing mono doesn't work for some reason, convert data to stereo instead:
        audio_data = np.tile(audio_data, 2)

        audio_clip = AudioArrayClip(audio_data, fps=AUDIO_SAMPLE_RATE)

        video_clip_pos = video_clip_pos.set_audio(audio_clip)

        video_clip_pos.preview(fps=FPS)
        pygame.quit()
