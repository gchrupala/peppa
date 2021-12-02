import logging
import pig.data
import pig.models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
import yaml


default_config = {'margin': 0.2,
                  'data': {'num_workers': 12,
                           'extract': False,
                           'prepare': False,
                           'iterable': False,
                           'normalization': 'kinetics',
                           'fixed_triplet': True,
                           'cache': False,
                           'target_size': [180, 100],
                           'train': {'batch_size': 8, 'jitter': False, 'shuffle': True},
                           'val': {'batch_size': 8},
                           'test': {'batch_size': 8}},
                  'video': {'pretrained': True, 'project': True},
                  'audio_class': 'Wav2VecEncoder',
                  'audio': {'path': 'data/in/wav2vec/wav2vec_small.pt',
                            'pretrained': True,
                            'freeze_feature_extractor': True,
                            'freeze_encoder_layers': 3,
                            'pooling': 'attention'},
                  'training': {'trainer_args': {'gpus': [1],
                                                'auto_select_gpus': False,
                                                'accumulate_grad_batches': 8}},
                  'optimizer': {'lr': 0.0001,
                                'warmup': 0.1,
                                'schedule': 'warmup_linear',
                                't_total': 32640}}

def get_git_commit():
    import git
    import os
    repo = git.Repo(os.getcwd())
    master = repo.head.reference
    return master.commit.hexsha

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
    config['git_commit'] = get_git_commit()
    data = pig.data.PigData(config['data'])
    net = pig.models.PeppaPig(config)

    
    trainer = pl.Trainer(callbacks=[ModelCheckpoint(monitor='valnarr_rec10',
                                                    mode='max',
                                                    every_n_epochs=1,
                                                    auto_insert_metric_name=True)],
                         max_time="02:00:00:00",
                         **config['training']['trainer_args'])
    trainer.fit(net, data)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--config_file", help="Configuration file (YAML)", default=None)

    
    args = parser.parse_args()
    main(args)

