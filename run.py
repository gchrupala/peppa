import logging
import pig.data
import pig.models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
import json

video_pretrained = False
default_config = dict(lr=1e-5,
                      margin=0.2,
                      data=dict(num_workers=24,
                                extract=False,
                                prepare=False,
                                iterable=False,
                                cache=False,
                                normalization='kinetics' if video_pretrained else 'peppa',
                                target_size=(180, 100),
                                transform=None,
                                train=dict(batch_size=8),
                                val=dict(batch_size=8),
                                test=dict(batch_size=8)),
                      video=dict(pretrained=video_pretrained, project=True),
                      audio_class='Wav2VecEncoder',
                      audio = dict(path = 'data/in/wav2vec/wav2vec_small.pt',
                                   freeze_feature_extractor=True,
                                   freeze_encoder_layers=None),
                      training = dict(trainer_args=dict(gpus=[0],
                                                        accumulate_grad_batches=8),
                                      monitor='val_loss/dataloader_idx_0')
                                      
)



def main(args):
    logging.getLogger().setLevel(logging.INFO)
    if args.config_file is None:
        config = default_config
    else:
        config = json.load(open(args.config_file))

    # Override config
    for key, value in vars(args).items():
        if key in config:
            config[key] = value
    
    data = pig.data.PigData(config['data'])
    net = pig.models.PeppaPig(config)

    
    trainer = pl.Trainer(callbacks=[ModelCheckpoint(monitor=config['training']['monitor'])],
                         **config['training']['trainer_args'])
    trainer.fit(net, data)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--config_file", help="Configuration file (JSON)", default=None)
    parser.add_argument("--lr", type=float, help="Initial learning rate", default=1e-5)
    
    args = parser.parse_args()
    main(args)

