import torch
import torch.utils
from torch.utils.data import Dataset, IterableDataset, DataLoader


from dataclasses import dataclass
import glob
import os.path
import pig.preprocess 
import moviepy.editor as m
import pytorch_lightning as pl
import logging
import pig.util
import json
import pickle
from typing import Union
import os.path
import math

SPLIT_SPEC = {'dialog': {'train': range(1, 197),
                         'val': range(197, 210),
                         'test': None},
              'narration': {'val': range(1, 105),
                            'test': range(105, 210),
                            'train': None}}


@dataclass
class Clip:
    """Video clip with associated audio."""
    video: torch.tensor
    audio: torch.tensor
    video_duration: float
    audio_duration: float
    filename: str
    offset: Union[float, None] = None
    index: Union[int, None] = None
    

@dataclass
class RawPair:
    """Positive raw video-audio example."""
    video: m.VideoFileClip
    audio: m.AudioFileClip
    video_idx: int
    audio_idx: int

    
@dataclass
class ClipBatch:
    """Batch of video clips with associated audio."""
    video: torch.tensor
    audio: torch.tensor
    video_duration: torch.tensor
    audio_duration: torch.tensor

def collate_audio(data):
    return pig.util.pad_audio_batch(data)

def collate(data):
    video, audio, vlen, alen = zip(*[(x.video, x.audio, x.video_duration, x.audio_duration) for x in data])
    return ClipBatch(video=pig.util.pad_video_batch(video),
                     audio=pig.util.pad_audio_batch(audio),
                     video_duration = torch.tensor(vlen),
                     audio_duration = torch.tensor(alen))

def featurize(clip, audio_sample_rate):
    frames = [ torch.tensor(frame/255).float()
               for frame in clip.iter_frames() ]
    if len(frames) > 0:
        v = torch.stack(frames)
        return Clip(video = v.permute(3, 0, 1, 2),
                    audio = featurize_audio(clip.audio, audio_sample_rate),
                    video_duration = clip.duration,
                    audio_duration = clip.audio.duration,
                    filename = clip.filename)
    else:
        raise ValueError("Clip has zero frames.")

def featurize_audio(clip, samplerate):
    # .to_soundarray extracts corrupted audio from small clips, 
    # but calling the function twice seems to fix the issue.
    clip.to_soundarray(fps=samplerate, buffersize=5000)
    a = torch.tensor(clip.to_soundarray(fps=samplerate, buffersize=5000)).float()
    return a.mean(dim=1, keepdim=True).permute(1,0)
  
class AudioFileDataset(IterableDataset):

    def __init__(self, paths, audio_sample_rate):
        self.paths = paths
        self.audio_sample_rate = audio_sample_rate

    def __iter__(self):
        for path in self.paths:
            with m.AudioFileClip(path) as clip:
                yield featurize_audio(clip, self.audio_sample_rate)

class AudioClipDataset(IterableDataset):

    def __init__(self, clips, audio_sample_rate):
        self.clips = clips
        self.audio_sample_rate = audio_sample_rate

    def __iter__(self):
        for clip in self.clips:
            yield featurize_audio(clip, self.audio_sample_rate)


class ArrayDataset(IterableDataset):

    def __init__(self, arrays):
        self.arrays = arrays

    def __iter__(self):
        yield from self.arrays
        
class VideoFileDataset(IterableDataset):

    def __init__(self, paths, audio_sample_rate):
        self.paths = paths
        self.audio_sample_rate = audio_sample_rate

    def __iter__(self):
        for path in self.paths:
            with m.VideoFileClip(path) as clip:
                yield featurize(clip, self.audio_sample_rate)


class VideoClipDataset(IterableDataset):

    def __init__(self, clips, audio_sample_rate):
        self.clips = clips
        self.audio_sample_rate = audio_sample_rate

    def __iter__(self):
        for clip in self.clips:
            yield featurize(clip, self.audio_sample_rate)
        
class GenericIterableDataset(IterableDataset):
    def __init__(self, items):
        self.items = items

    def __iter__(self):
        yield from self.items

def audiofile_loader(paths, batch_size=32, audio_sample_rate=44100):
    dataset = AudioFileDataset(paths, audio_sample_rate)
    return DataLoader(dataset, collate_fn=collate_audio, batch_size=batch_size)

def grouped_audiofile_loader(paths, batch_size=32, audio_sample_rate=44100):
    dataset = AudioFileDataset(paths, audio_sample_rate)
    loader = grouped_loader(dataset,
                            lambda x: x.shape[1],
                            collate_audio,
                            batch_size)
    return loader
    

def audioarray_loader(arrays, batch_size=32):
    dataset = ArrayDataset(arrays)
    return DataLoader(dataset, collate_fn=collate_audio, batch_size=batch_size)

def audioclip_loader(clips, batch_size=32, audio_sample_rate=44100):
    dataset = AudioClipDataset(clips, audio_sample_rate)
    return DataLoader(dataset, collate_fn=collate_audio, batch_size=batch_size)

def grouped_audioclip_loader(paths, batch_size=32, audio_sample_rate=44100):
    dataset = AudioClipDataset(paths, audio_sample_rate)
    loader = grouped_loader(dataset,
                            lambda x: x.shape[1],
                            collate_audio,
                            batch_size)
    return loader

def grouped_audioarray_loader(arrays, batch_size=32):
    dataset = ArrayDataset(arrays)
    loader = grouped_loader(dataset,
                            lambda x: x.shape[1],
                            collate_audio,
                            batch_size)
    return loader

class GroupedDataset(IterableDataset):
    """Returns batches of within groups."""
    def __init__(self, dataset, key, collate_fn, batch_size):
        self.dataset = dataset
        self.key = key
        self.collate_fn = collate_fn
        self.batch_size = batch_size

    def __iter__(self):
        for _, items in pig.util.grouped(sorted(self.dataset, key=self.key), key=self.key):
            yield from DataLoader(GenericIterableDataset(items),
                                  collate_fn=self.collate_fn,
                                  batch_size=self.batch_size)

def grouped_loader(dataset, key, collate_fn, batch_size=8):
    gds = GroupedDataset(dataset, key, collate_fn, batch_size=batch_size)
    return DataLoader(gds, batch_size=None, batch_sampler=None) 
            
class PeppaPigDataset(Dataset):
    def __init__(self, force_cache=False, cache_dir=None, scrambled_video=False, **kwargs):
        dataset = PeppaPigIterableDataset(**kwargs)
        
        if cache_dir is None:
            self.cache_dir = f"data/out/items-{dataset.config_id()}/"
        else:
            self.cache_dir = cache_dir
        if force_cache or not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            pickle.dump(kwargs, open(f"{self.cache_dir}/settings.pkl", "wb"))
            for i, item in enumerate(dataset):
                logging.info(f"Caching item {self.cache_dir}/{i}.pt")
                torch.save(item, f"{self.cache_dir}/{i}.pt")
        self.length = len(glob.glob(f"{self.cache_dir}/*.pt"))
        self.scrambled_video = scrambled_video
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError("Index out of range")
        else:
            item = torch.load(f"{self.cache_dir}/{idx}.pt")
            if self.scrambled_video:
                # Shuffle along temporal dimension
                idx = torch.randperm(item.video.shape[1])
                item.video = item.video[:, idx]
            return item

    @classmethod
    def load(cls, directory):
        return PeppaPigDataset(force_cache=False, cache_dir=directory)
    
class PeppaPigIterableDataset(IterableDataset):
    def __init__(self,
                 split=['val'],
                 target_size=(180, 100),
                 fragment_type='dialog',
                 duration=3.2,
                 audio_sample_rate=44100,
                 jitter=False,
                 jitter_sd=None
                 ):
        if type(split) is str:
            raise ValueError("`split` should be a list of strings")
        self.split = split
        self.target_size = target_size
        self.fragment_type = fragment_type
        self.duration = duration
        self.audio_sample_rate = audio_sample_rate
        self.jitter = jitter
        self.jitter_sd = jitter_sd
        self.split_spec = SPLIT_SPEC

    def config_id(self):
        return "-".join([','.join(self.split),
                         f"{self.target_size[0]}x{self.target_size[1]}",
                         self.fragment_type,
                         f"{self.duration}",
                         f"{self.audio_sample_rate}",
                         f"{self.jitter},{self.jitter_sd}" if self.jitter else ""])

        
    def featurize(self, clip):
        return featurize(clip, self.audio_sample_rate)
        
    def _clips(self):
        for clip in self._raw_clips():
            try:
                yield self.featurize(clip)
            except ValueError as e:
                logging.warning(f"{e}")

    def _raw_clips(self):
        width,  height = self.target_size
        paths = [ path for split in self.split \
                       for episode_id in self.split_spec[self.fragment_type][split] \
                       for path in glob.glob(f"data/out/{width}x{height}/{self.fragment_type}/{episode_id}/*.avi") ]
        if len(paths) == 0:
            raise RuntimeError(f"No clips found in data/out/{width}x{height}/{self.fragment_type}/ . Extract the data first.")

        # Split data between workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            first = 0
            last = len(paths)
        else:
            per_worker = int(math.ceil(len(paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            first = worker_id * per_worker
            last = min(first + per_worker, len(paths))
            logging.info(f"Workerid: {worker_id}; [{first}:{last}]")
        for path in paths[first:last]:
            with m.VideoFileClip(path) as video:
            #logging.info(f"Path: {path}, size: {video.size}")
                if self.duration is None:
                    i = os.path.splitext(os.path.basename(path))[0]
                    meta = json.load(open(f"{os.path.dirname(path)}/{i}.json"))
                    clips = pig.preprocess.lines(video, meta)
                else:
                    clips = pig.preprocess.segment(video, duration=self.duration, jitter=self.jitter, jitter_sd=self.jitter_sd)
                for clip in clips:
                    yield clip

    def __iter__(self):
        yield from self._clips()

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

    
class PigData(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loader_args = ['batch_size', 'shuffle']
        if self.config['iterable']:
            self.Dataset = lambda *args, **kwargs: PeppaPigIterableDataset(*args, **kwargs)
        else:
            self.Dataset = lambda *args, **kwargs: PeppaPigDataset(*args, **kwargs)
    
    def prepare_data(self):
        if self.config['extract']:
            logging.info(f"Extracting data for target size {self.config['target_size']}")
            pig.preprocess.extract(self.config['target_size'])
        if self.config['prepare']:    
            logging.info("Collecting stats on training data.")
            
            train = self.Dataset(target_size=self.config['target_size'],
                                 audio_sample_rate=self.config['audio_sample_rate'],
                                 split=['train'], fragment_type='dialog', 
                                  **{k:v for k,v in self.config['train'].items()
                                     if k not in self.loader_args})
            logging.info("Saving stats")
            stats = get_stats(DataLoader(train, collate_fn=collate, batch_size=32))
            torch.save(stats, "data/out/stats.pt")

    def setup(self, **kwargs):
        logging.info("Creating train/val/test datasets")
        self.train = self.Dataset(target_size=self.config['target_size'],
                                  audio_sample_rate=self.config['audio_sample_rate'],
                                  split=['train'],
                                  fragment_type='dialog',
                                  **{k:v for k,v in self.config['train'].items()
                                     if k not in self.loader_args})

        self.val_dia   = PeppaPigDataset(force_cache=self.config['val']['force_cache'],
                                         target_size=self.config['target_size'],
                                         split=['val'], fragment_type='dialog',
                                         duration=self.config['val']['duration'],
                                         audio_sample_rate=self.config['audio_sample_rate'],
                                         jitter=self.config['val']['jitter'],
                                         jitter_sd=self.config['val'].get('jitter_sd'))

        self.val_narr = PeppaPigDataset(force_cache=self.config['val']['force_cache'],
                                        target_size=self.config['target_size'],
                                        split=['val'],
                                        fragment_type='narration',
                                        duration=self.config['val']['duration'],
                                        audio_sample_rate=self.config['audio_sample_rate'],
                                        jitter=self.config['val']['jitter'],
                                        jitter_sd=self.config['val'].get('jitter_sd'))
        self.val_dia3 = pig.data.PeppaPigDataset(
            force_cache=self.config['val']['force_cache'],
            target_size=self.config['target_size'],
            split=['val'], fragment_type='dialog',
            duration=None,
            audio_sample_rate=self.config['audio_sample_rate'],
            jitter=None)
        
        self.val_narr3 = pig.data.PeppaPigDataset(
            force_cache=self.config['val']['force_cache'],
            target_size=self.config['target_size'],
            split=['val'], fragment_type='narration',
            audio_sample_rate=self.config['audio_sample_rate'],
            duration=None,
            jitter=None)
        
        
    def train_dataloader(self):
        return DataLoader(self.train, collate_fn=collate, num_workers=self.config['num_workers'],
                          batch_size=self.config['train']['batch_size'],
                          shuffle=self.config['train']['shuffle'])

    def val_dataloader(self):
        
        dia = DataLoader(self.val_dia, collate_fn=collate, num_workers=self.config['num_workers'],
                          batch_size=self.config['val']['batch_size'])
        narr = DataLoader(self.val_narr, collate_fn=collate,
                               num_workers=self.config['num_workers'],
                          batch_size=self.config['val']['batch_size'])
        key = lambda x: x.audio_duration
        dia3 = grouped_loader(self.val_dia3,   key, collate,
                              batch_size=self.config['val']['batch_size'])
        narr3 = grouped_loader(self.val_narr3, key, collate,
                              batch_size=self.config['val']['batch_size'])
        
        return [ dia, narr, dia3, narr3 ]
    
    def test_dataloader(self):
        raise NotImplementedError
        #return DataLoader(self.test, collate_fn=collate, num_workers=self.config['num_workers'],
        #                  batch_size=self.config['test']['batch_size'])

