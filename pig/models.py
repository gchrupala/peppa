import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torchvision.models.video as V
import torchaudio.models as A
from torchaudio.models.wav2vec2.utils import import_fairseq_model
import fairseq
from torchvision.transforms import Compose
from pig.loss import TripletLoss
import pig.data
import logging
import sys
import pig.util
import pig.metrics

## Audio encoders

class Wav2LetterEncoder(nn.Module):
    
    def __init__(self, project=False):
        super().__init__()
        self.audio = A.Wav2Letter(input_type='waveform', num_features=1, num_classes=512)
        self.audiopool = torch.nn.AdaptiveAvgPool2d((512,1))
        if project:
            self.project = nn.Linear(512, 512)
        else:
            self.project = pig.util.identity

    def forward(self, x):
        return Compose([self.audio.acoustic_model,
                        self.audiopool,
                        lambda x: x.squeeze(dim=2),
                        self.project,
                        lambda x: nn.functional.normalize(x, p=2, dim=1)
        ])(x)


class Wav2VecEncoder(nn.Module):
    def __init__(self, path, freeze_feature_extractor=False, freeze_encoder_layers=None):
        super().__init__()
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([path])
        self.audio = import_fairseq_model(model[0], num_out=28)
        if freeze_feature_extractor:
            for param in self.audio.feature_extractor.parameters():
                param.requires_grad = False
        if freeze_encoder_layers is not None:
            for index in range(0, freeze_encoder_layers+1):
                for param in self.audio.encoder.layers[index]:
                    param.requires_grad = False
        self.audiopool = torch.nn.AdaptiveAvgPool2d((512,1))
        self.project = nn.Linear(512, 512)

        
    def forward(self, x):
        features, _ = self.audio.extract_features(x.squeeze(dim=1))
        return Compose([self.audiopool,
                        lambda x: x.squeeze(dim=2),
                        self.project,
                        lambda x: nn.functional.normalize(x, p=2, dim=1)
        ])(features)

        
## Video encoders
class R3DEncoder(nn.Module):
    
    def __init__(self, pretrained=False, project=False):
        super().__init__()
        self.pretrained = pretrained
        self.video = V.r3d_18(pretrained=pretrained, progress=False)
        if project:
            self.project = nn.Linear(512, 512)
        else:
            self.project = identity

        
    def forward(self, x):
        return Compose([self.video.stem,
                        self.video.layer1,
                        self.video.layer2,
                        self.video.layer3,
                        self.video.layer4,
                        self.video.avgpool,
                        lambda x: x.flatten(1),
                        self.project,
                        lambda x: nn.functional.normalize(x, p=2, dim=1)
        ])(x)
    
    
class PeppaPig(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.loss = TripletLoss(margin=self.config['margin'])
        self.video_encoder = R3DEncoder(**self.config['video'])
        self.audio_encoder = get_class(config['audio_class'])(**config['audio'])
        
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        raise NotImplemented

    def encode_video(self, x):
        return self.video_encoder(x)
    
    def encode_audio(self, x):
        return self.audio_encoder(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        
        V = self.encode_video(batch.video)
        A = self.encode_audio(batch.audio)
        loss = self.loss(V, A)
        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        if dataloader_idx == 0:
            V = self.encode_video(batch.video)
            A = self.encode_audio(batch.audio)
            loss = self.loss(V, A)
            # Logging to TensorBoard by default
            self.log("val_loss", loss, prog_bar=True)
            return (V, A)
        elif dataloader_idx == 1:
            a = self.encode_audio(batch.anchor)
            p = self.encode_video(batch.positive)
            n = self.encode_video(batch.negative)
            acc3 = pig.metrics.triplet_accuracy(a, p, n)
            self.log("val_acc3", acc3, prog_bar=True)
            return None
        else:
            raise ValueError(f"Invalid dataloader index {dataloader_idx}")
        
        
    def validation_epoch_end(self, outputs):
        out_main, _out_triplet = outputs
        V, A = zip(*out_main)
        V = torch.cat(V, dim=0)
        A = torch.cat(A, dim=0)
        correct = torch.eye(V.shape[0], device=A.device)
        rec10 = pig.metrics.recall_at_n(V, A, correct=correct, n=10)
        self.log("val_rec10", rec10, prog_bar=True)

        
    #def test_step(self, batch, batch_idx):    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer

def get_class(name):
    return getattr(sys.modules[__name__], name)

    
    
def main():

    logging.getLogger().setLevel(logging.WARNING)
    
    video_pretrained = False
    
    config = dict(lr=1e-4,
                  margin=0.2,
                  data=dict(normalization='kinetics' if video_pretrained else 'peppa',
                            target_size=(180, 100),
                            transform=None,
                            train=dict(split='train', fragment_type='dialog',
                                       window=0, duration=3.2, batch_size=8),
                            val=dict(split='val', fragment_type='narration',
                                     window=0, duration=None, batch_size=8,
                                     hard_triplet=True),
                            test=dict(split='test', fragment_type='narration',
                                      window=0, duration=None, batch_size=8)),
                  video=dict(pretrained=video_pretrained, project=True),
                  audio_class='Wav2VecEncoder',
                  audio = dict(path = 'data/in/wav2vec/wav2vec_small.pt', freeze_feature_extractor=True, freeze_encoder_layers=None)
                  #audio_class='Wav2LetterEncoder',
                  #audio=dict(project=True)

    )
                                      
    data = pig.data.PigData(config['data'], extract=False, prepare=False)
    net = PeppaPig(config)
    

    trainer = pl.Trainer(gpus=1, val_check_interval=100, accumulate_grad_batches=8)
    trainer.fit(net, data)

if __name__ == '__main__':
    main()
    
