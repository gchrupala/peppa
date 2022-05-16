import pig.data as D
import yaml
import random
import torchaudio

def sample(k=50):
    hparams = yaml.safe_load(open("hparams_base.yaml"))
    train = D.PeppaPigDataset(target_size=hparams['data']['target_size'],
                              audio_sample_rate=hparams['data']['audio_sample_rate'],
                              split=['train'],
                              fragment_type='dialog',
                              **{k:v for k,v in hparams['data']['train'].items()
                                 if k not in ['batch_size', 'shuffle']})
    idx = random.sample(range(len(train)), k)
    for i in idx:
        torchaudio.save(f"data/out/audio_sample_to_check/{i}.wav",
                        src=train[i].audio,
                        sample_rate=hparams['data']['audio_sample_rate'],
                        channels_first=True)
        
if __name__ == '__main__':
    sample()
    
