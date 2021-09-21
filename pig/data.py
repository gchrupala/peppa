import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision import datasets, transforms
from dataclasses import dataclass
import glob
import pig.preprocess 
import moviepy.editor as m
import pytorch_lightning as pl
import logging
from itertools import groupby

@dataclass
class Clip:
    """Video clip with associated audio."""
    video: torch.tensor
    audio: torch.tensor
    filepath: str

@dataclass
class Pair:
    """Positive video-audio example."""
    video: torch.tensor
    audio: torch.tensor
    video_idx: int
    audio_idx: int
    filepath: str

@dataclass
class ClipBatch:
    """Batch of video clips with associated audio."""
    video: torch.tensor
    audio: torch.tensor
    
    
def batch_audio(audio):
    size = min(x.shape[1] for x in audio)
    return torch.stack([ x[:, :size] for x in audio ])

def batch_video(video):
    size = min(x.shape[1] for x in video)
    return torch.stack([ x[:, :size, :, :] for x in video ])

                       
def collate(data):
    video, audio = zip(*[(x.video, x.audio) for x in data])
    return ClipBatch(video=batch_video(video), audio=batch_audio(audio))

    
class PeppaPigIDataset(Dataset):
    def __init__(self):
        raise NotImplemented

    def __len__(self):
        raise NotImplemented

    def __getitem__(self, idx):
        raise NotImplemented


class PeppaPigIterableDataset(IterableDataset):
    def __init__(self, split='val', fragment_type='dialog', window=2):
        self.split = split
        self.fragment_type = fragment_type
        self.window = window
        self.splits = dict(train = range(1, 197),
                           val  = range(197, 203),
                           test = range(203, 210))
        
    def _clips(self):
        for episode_id in self.splits[self.split]:
            for path in sorted(glob.glob(f"data/out/{self.fragment_type}/{episode_id}/*.avi")):
                with m.VideoFileClip(path) as video:
                    logging.info(f"Path: {path}, size: {video.size}")
                    for clip in pig.preprocess.segment(video, duration=3.2):
                        v = torch.stack([ torch.tensor(frame/255).float()
                                          for frame in clip.iter_frames() ])
                        a = torch.tensor(clip.audio.to_soundarray()).float()
                        yield Clip(video = v.permute(3, 0, 1, 2),
                                   audio = a.mean(dim=1, keepdim=True).permute(1,0),
                                   filepath = path)        
        
    def _positives(self, items):
        clips  = list(enumerate(items))
        for i, a in clips:
            for j, b in clips:
                if abs(j - i) <= self.window: 
                    yield Pair(video = a.video,
                               audio = b.audio,
                               video_idx = i,
                               audio_idx = j,
                               filepath = a.filepath)
                    
    def __iter__(self):
        for _path, items in groupby(self._clips(), key=lambda x: x.filepath):
            yield from self._positives(items)

@dataclass
class Stats:
    """Mean and standard deviation of a data sample."""
    video_mean : torch.Tensor
    video_std  : torch.Tensor
    audio_mean : torch.Tensor
    audio_std  : torch.Tensor
            
def get_stats(loader):
    """Compute means and standard deviations over data points from `loader`."""
    # Mean pass
    video_sum = torch.zeros(1,3,1,1,1).float()
    video_count = torch.zeros(1,3,1,1,1).float()
    audio_sum = torch.zeros(1,1,1).float()
    audio_count = torch.zeros(1,1,1).float()
    for batch in loader:
         video_sum   += batch.video.sum(dim=(0,2,3,4), keepdim=True)
         video_count += torch.ones_like(batch.video).sum(dim=(0,2,3,4), keepdim=True)
         audio_sum   += batch.audio.sum(dim=(0,2), keepdim=True) 
         audio_count += torch.ones_like(batch.audio).sum(dim=(0,2), keepdim=True)
    video_mean = video_sum/video_count
    audio_mean = audio_sum/audio_count
    logging.info(f"Mean video: {video_mean}\nMean audio: {audio_mean}")
    # STD pass
    video_sse = torch.zeros(1,3,1,1,1).float()
    audio_sse = torch.zeros(1,1,1).float()
    for batch in loader:
        video_sse += ((batch.video - video_mean)**2).sum(dim=(0,2,3,4), keepdim=True)
        audio_sse += ((batch.audio - audio_mean)**2).sum(dim=(0,2), keepdim=True)
    return Stats(video_mean = video_mean.squeeze(),
                 video_std  = ((video_sse/video_count) **0.5).squeeze(),
                 audio_mean = audio_mean.squeeze(),
                 audio_std  = ((audio_sse/audio_count) **0.5).squeeze())

def worker_init_fn(worker_id):
    raise NotImplemented

    
class PeppaPigData(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        

    def prepare_data(self):
        # called only on 1 GPU
        pig.preprocess.extract()
        

    def setup(self, stage = None):
        # called on every GPU
        pass


    def train_dataloader(self):
        transforms = ...
        return DataLoader(self.train)

    def val_dataloader(self):
        transforms = ...
        return DataLoader(self.val)

    def test_dataloader(self):
        transforms = ...
        return DataLoader(self.test, batch_size=64)
