import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torchvision.models.video as V
import torchaudio.models as A
import platalea.encoders as P
from torchvision.transforms import Compose


def 

class PeppaPig(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.video = V.r3d_18(pretrained=False, progress=False)
        self.audio = A.Wav2Letter()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        raise NotImplemented

    def loss(self, video, audio):
        return NotImplemented
    
    def encode_video(self, x):
        return Compose([self.video.stem,
                        self.video.layer1,
                        self.video.layer2,
                        self.video.layer3,
                        self.video.layer4,
                        self.video.avgpool,
                        lambda x: x.flatten(1)])(x)
    
    def encode_audio(self, x):
        return self.audio.acoustic_model(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        video, audio = batch
        
        video_enc = self.encode_video(video)
        audio_enc = self.encode_audio(audio)

        loss = self.loss(video_enc, audio_enc)
        
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    
def main():
    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset, num_workers=12)
    autoencoder = LitAutoEncoder()

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    trainer = pl.Trainer(gpus=1, limit_train_batches=0.1, max_epochs=2)
    
    trainer.fit(autoencoder, train_loader)

if __name__ == '__main__':
    main()
    
