import logging
import pig.data
import pig.models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
import yaml
from pig.execution import default_config

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

