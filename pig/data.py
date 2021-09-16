import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision import datasets, transforms
from dataclasses import dataclass
import glob
import pig.preprocess 
import moviepy.editor as m
import pytorch_lightning as pl
import logging

@dataclass
class Clip:
    """Video clip with associated audio."""
    video: torch.tensor
    audio: torch.tensor

@dataclass
class ClipBatch:
    """Batch of video clips with associated audio."""
    video: torch.tensor
    audio: torch.tensor
    
def batch_audio(audios):
    raise NotImplemented

def batch_video(videos):
    raise NotImplemented

def collate_fn(data):
    videos, audios = zip(* [(clip.video, clip.audio) for clip in data])
    videos = batch_videos(videos)
    audios = batch_audios(audios)
    return ClipBatch(video=videos, audio=audios)
    

class PeppaPigIDataset(Dataset):
    def __init__(self):
        raise NotImplemented

    def __len__(self):
        raise NotImplemented

    def __getitem__(self, idx):
        raise NotImplemented


class PeppaPigIterableDataset(IterableDataset):
    def __init__(self, split='val', fragment_type='dialog'):
        self.split = split
        self.fragment_type = fragment_type
        self.splits = dict(train = range(1, 197),
                           val  = range(197, 203),
                           test = range(203, 210))

    def _clips(self):
        for episode_id in self.splits[self.split]:
            for path in sorted(glob.glob(f"data/out/{self.fragment_type}/{episode_id}/*.avi")):
                with m.VideoFileClip(path) as video:
                    logging.info(f"Path: {path}")
                    for clip in pig.preprocess.segment(video, duration=3.2):
                        v = torch.stack([ torch.tensor(frame/255).float()
                                          for frame in clip.iter_frames() ])
                        a = torch.tensor(clip.audio.to_soundarray()).float()
                        yield Clip(video = v.permute(3, 0, 1, 2),
                                   audio = a.mean(dim=1, keepdim=True).permute(1,0))
    def __iter__(self):
        return iter(self._clips())

def worker_init_fn(worker_id):
    raise NotImplemented

    
class PeppaPigData(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_dims = None
        self.vocab_size = 0

    def prepare_data(self):
        # called only on 1 GPU
        pig.preprocess.extract()
        

    def setup(self, stage = None):
        # called on every GPU
        vocab = load_vocab()
        self.vocab_size = len(vocab)

        self.train, self.val, self.test = load_datasets()
        self.train_dims = self.train.next_batch.size()

    def train_dataloader(self):
        transforms = ...
        return DataLoader(self.train, batch_size=64)

    def val_dataloader(self):
        transforms = ...
        return DataLoader(self.val, batch_size=64)

    def test_dataloader(self):
        transforms = ...
        return DataLoader(self.test, batch_size=64)
