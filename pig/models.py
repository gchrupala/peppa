import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torchvision.models.video as V
import torchaudio.models as A

from torchvision.transforms import Compose
from pig.loss import TripletLoss
import pig.data
import logging


def identity(x):
    return x
## Audio encoders

class Wav2LetterEncoder(nn.Module):
    
    def __init__(self, project=False):
        super().__init__()
        self.audio = A.Wav2Letter(input_type='waveform', num_features=1, num_classes=512)
        self.audiopool = torch.nn.AdaptiveAvgPool2d((512,1))
        if project:
            self.project = nn.Linear(512, 512)
        else:
            self.project = identity

    def forward(self, x):
        return Compose([self.audio.acoustic_model,
                        self.audiopool,
                        lambda x: x.squeeze(),
                        self.project,
                        lambda x: nn.functional.normalize(x, p=2, dim=1)
        ])(x)



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
        self.audio_encoder=Wav2LetterEncoder(project=self.config['audio']['project'])
        self.video_encoder=R3DEncoder(pretrained=self.config['video']['pretrained'],
                                      project=self.config['video']['project'])
        
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
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        V = self.encode_video(batch.video)
        A = self.encode_audio(batch.audio)
        loss = self.loss(V, A)
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
        return loss

        
    #def test_step(self, batch, batch_idx):    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer


    
    
def main():

    logging.getLogger().setLevel(logging.INFO)
                                      
    config = dict(lr=1e-5,
                  margin=0.1,
                  video=dict(pretrained=False, project=True),
                  audio=dict(project=True))
                                      
    data = pig.data.PigData(extract=False,
                            prepare=False,
                            normalization='kinetics' if config['video']['pretrained'] else 'peppa')
    net = PeppaPig(config)

    
    trainer = pl.Trainer(gpus=3, overfit_batches=10, log_every_n_steps=10, limit_val_batches=0)
    trainer.fit(net, data)

if __name__ == '__main__':
    main()
    
