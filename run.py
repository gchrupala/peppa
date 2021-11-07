import logging
import pig.data
import pig.models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
import yaml

video_pretrained = False
default_config = dict(
                      margin=0.2,
                      data=dict(num_workers=24,
                                extract=False,
                                prepare=False,
                                iterable=False,
                                cache=False,
                                normalization='kinetics' if video_pretrained else 'peppa',
                                target_size=(180, 100),
                                transform=None,
                                train=dict(batch_size=8, shuffle=True),
                                val=dict(batch_size=8),
                                test=dict(batch_size=8)),
                      video=dict(pretrained=video_pretrained, project=True),
                      audio_class='Wav2VecEncoder',
                      audio = dict(path = 'data/in/wav2vec/wav2vec_small.pt',
                                   freeze_feature_extractor=True,
                                   freeze_encoder_layers=None),
                      training = dict(trainer_args=dict(gpus=1, auto_select_gpus=True,
                                                        accumulate_grad_batches=8),
                                      monitor='val_loss/dataloader_idx_0'),
                      optimizer = dict(lr=0.1e-4, warmup=0.1, schedule='warmup_linear', t_total=32640)
                                      
)



def main(args):
    logging.getLogger().setLevel(logging.INFO)
    if args.config_file is None:
        config = default_config
    else:
        config = yaml.safe_load(open(args.config_file))

    # Override config
    for key, value in vars(args).items():
        if key in config:
            config[key] = value
    
    data = pig.data.PigData(config['data'])
    net = pig.models.PeppaPig(config)

    
    trainer = pl.Trainer(callbacks=[ModelCheckpoint(monitor='val_loss/dataloader_idx_0', mode='min'),
                                    ModelCheckpoint(monitor='val_acc3/dataloader_idx_1', mode='max'),
                                    ModelCheckpoint(monitor='valnarr_loss/dataloader_idx_2', mode='min'),
                                    ModelCheckpoint(monitor='valnarr_acc3/dataloader_idx_3', mode='max'),
                                    ModelCheckpoint(monitor='val_rec10', mode='max'),
                                    ModelCheckpoint(monitor='valnarr_rec10', mode='max')
    ],
                         **config['training']['trainer_args'])
    trainer.fit(net, data)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--config_file", help="Configuration file (YAML)", default=None)

    
    args = parser.parse_args()
    main(args)

