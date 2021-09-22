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

class PeppaPig(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.video = V.r3d_18(pretrained=False, progress=False)
        self.audio = A.Wav2Letter(input_type='waveform', num_features=1, num_classes=512)
        self.audiopool = torch.nn.AdaptiveAvgPool2d((512,1))
        self.loss = TripletLoss(margin=0.2)
        
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        raise NotImplemented

    def encode_video(self, x):
        return Compose([self.video.stem,
                        self.video.layer1,
                        self.video.layer2,
                        self.video.layer3,
                        self.video.layer4,
                        self.video.avgpool,
                        lambda x: x.flatten(1)])(x)
    
    def encode_audio(self, x):
        return self.audiopool(self.audio.acoustic_model(x)).squeeze()
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        
        V = self.encode_video(batch.video)
        logging.info(f"Video encoded: {V.shape}")
        A = self.encode_audio(batch.audio)
        logging.info(f"Audio encoded: {A.shape}")
        loss = self.loss(V, A)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        V = self.encode_video(batch.video)
        logging.info(f"Video encoded: {V.shape}")
        A = self.encode_audio(batch.audio)
        logging.info(f"Audio encoded: {A.shape}")
        loss = self.loss(V, A)
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
        return loss

        
    #def test_step(self, batch, batch_idx):    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


    
    
def main():

    logging.getLogger().setLevel(logging.INFO)
    data = pig.data.PigData()
    
    net = PeppaPig()

    trainer = pl.Trainer(gpus=1, limit_train_batches=100, max_epochs=2, val_check_interval=10)
    
    trainer.fit(net, data)

if __name__ == '__main__':
    main()
    
