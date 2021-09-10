import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from dataclasses import dataclass

@dataclass
class Clip:
    """Video clip with associated audio."""
    video: torch.tensor
    audio: torch.tensor

@dataclass
class ClipBatch
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
    
class PeppaPigDataset(Dataset):
    def __init__(self):
        raise NotImplemented

    def __len__(self):
        raise NotImplemented

    def __getitem__(self, idx):
        raise NotImplemented

class PeppaPigData(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_dims = None
        self.vocab_size = 0

    def prepare_data(self):
        # called only on 1 GPU
        download_dataset()
        tokenize()
        build_vocab()

    def setup(self, stage: Optional[str] = None):
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
