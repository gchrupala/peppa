import yaml
import glob

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

    return config

def dump_conditions():
    config = conditions()
    for name, hparams in config.items():
        yaml.dump(hparams, open(f"hparams_{name}.yaml", "w"))

def clean(item):
    from copy import deepcopy
    out = deepcopy(item)
    out['data']['audio_sample_rate'] = out['data'].get('audio_sample_rate', 44100)
    del out['training']['trainer_args']['gpus']
    if 'git_commit' in out:
        del out['git_commit']
    return out

def match_conditions():
    configs = conditions()
    prev = [ 335, 336, 351, 375, 376, 378, 376, 384 ]
    paths = set(glob.glob("lightning_logs/version_[456]*/hparams.yaml") + [ f"lightning_logs/version_{j}/hparams.yaml" for j in prev ])
    runs = {}
    versions = [ (f, yaml.safe_load(open(f))) for f in paths ] 
    for name, conf in configs.items():
        runs[name] = []
        conf = clean(conf)
        for path, version in versions:
            i = int(path.split('/')[1].split('_')[1])
            if conf == clean(version):
                runs[name].append(i)
    return runs

def select_runs(conditions):
    # Keep 4 runs
    for k,v in conditions.items():
        conditions[k] = sorted(v)[:4]
    output = dict(pretraining=conditions['base'] + conditions['pretraining_v'] + \
                  conditions['pretraining_a'] + conditions['pretraining_none'],
                  freeze_wav2vec=conditions['base'] + conditions['freeze_wav2vec'],
                  jitter=conditions['base'] + conditions['jitter'],
                  static=conditions['base'] + conditions['static'])
    return output

def save_conditions():
    configs = match_conditions()
    conditions = select_runs(configs)
    yaml.dump(conditions, open("conditions.yaml", "w"))
