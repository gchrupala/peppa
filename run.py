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
                           'target_size': [180, 100],
                           'audio_sample_rate': 44100,
                           'train': {'force_cache': False,
                                     'batch_size': 8,
                                     'jitter': True,
                                     'jitter_sd': 0.5,
                                     'duration': 2.3,
                                     'shuffle': True},
                           'val': {'force_cache': False,
                                   'batch_size': 8,
                                   'jitter': False,
                                   'duration': 2.3},
                           'test': {'force_cache': False,
                                    'batch_size': 8,
                                    'jitter': False,
                                    'duration': 2.3}},
                  'video': {'pretrained': True,
                            'project': True,
                            'version': 'r2plus1d_18',
                            'pooling': 'attention'},
                  'audio': {'path': 'data/in/wav2vec/wav2vec_small.pt',
                            'pretrained': True,
                            'freeze_feature_extractor': False,
                            'freeze_encoder_layers': None,
                            'pooling': 'attention',
                            'full': True},
                  'training': {'trainer_args': {'gpus': 1,
                                                'auto_select_gpus': False,
                                                'accumulate_grad_batches': 8,
                                                'precision': 16}},
                  'optimizer': {'lr': 0.0001,
                                'warmup': 0.1,
                                'schedule': 'warmup_linear',
                                't_total': 15000}}

def conditions(base=default_config):
    from copy import deepcopy
    config = {}
    config['base'] = base
    
    freeze_wav2vec = deepcopy(base)
    freeze_wav2vec['audio']['freeze_feature_extractor'] = True
    freeze_wav2vec['audio']['freeze_encoder_layers'] = 12
    config['freeze_wav2vec'] = freeze_wav2vec

    jitter = deepcopy(base)
    jitter['data']['train']['jitter'] = False
    jitter['data']['train']['jitter_sd'] = None
    config['jitter'] = jitter

    pretraining_v = deepcopy(base)
    pretraining_v['audio']['pretrained'] = False
    config['pretraining_v'] = pretraining_v

    pretraining_a = deepcopy(base)
    pretraining_a['video']['pretrained'] = False
    config['pretraining_a'] = pretraining_a

    pretraining_none = deepcopy(base)
    pretraining_none['video']['pretrained'] = False
    pretraining_none['audio']['pretrained'] = False
    config['pretraining_none'] = pretraining_none

    static = deepcopy(base)
    static['video']['static'] = True
    del static['video']['version']
    config['static'] = static

    for name, hparams in config.items():
        yaml.dump(hparams, open(f"hparams_{name}.yaml", "w"))

    
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
    
    checkpoint_rec10 = ModelCheckpoint(monitor='valnarr_rec_fixed',
                                       mode='max',
                                       every_n_epochs=1,
                                       auto_insert_metric_name=True,
                                       verbose=False,
                                       save_last=True,
                                       save_top_k=1,
                                       save_weights_only=False,
                                       period=1,
                                       dirpath=None,
                                       filename="{epoch}-{valnarr_rec_fixed:.2f}"
    )
    checkpoint_triplet = ModelCheckpoint(monitor='valnarr_triplet',
                                       mode='max',
                                       every_n_epochs=1,
                                       auto_insert_metric_name=True,
                                       verbose=False,
                                       save_last=True,
                                       save_top_k=1,
                                       save_weights_only=False,
                                       period=1,
                                       dirpath=None,
                                       filename="{epoch}-{valnarr_triplet:.2f}"
    )
    trainer = pl.Trainer(callbacks=[checkpoint_rec10, checkpoint_triplet],
                         max_time="02:00:00:00",
                         num_sanity_val_steps=15,
                         limit_train_batches=args.limit_train_batches,
                         limit_val_batches=args.limit_val_batches,
                         **config['training']['trainer_args'])
    trainer.fit(net, data)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--config_file", help="Configuration file (YAML)", default=None)

    
    args = parser.parse_args()
    main(args)

