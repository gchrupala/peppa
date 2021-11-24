import yaml
from dataclasses import dataclass
import pandas as pd
import logging
import glob
import moviepy.editor as m
import time
import json
import os
import os.path
import torch
import torch.nn

def speakerize(data):
    for part in data['narrator_splits']:
        for sub in part['context']['subtitles']:
            sub['speaker'] = None

@dataclass            
class Interval:
    begin: pd.Timedelta
    end: pd.Timedelta
    payload: ... = None

    def within(self, larger):
        return larger.begin <= self.begin and larger.end >= self.end
    
            
def speakerize_tokens(context):
    passages = [ Interval(begin=pd.Timedelta(x['begin']), end=pd.Timedelta(x['end']),
                          payload=x['speaker']) for x in context['subtitles']
                 if x['speaker'] is not None ]
    logging.info(f"Found {len(passages)} passages")
    for token in context['tokenized']:
        #logging.info(f"Token {token}")
        for passage in passages:
            if Interval(pd.Timedelta(token['begin']), pd.Timedelta(token['end'])).within(passage):
                token['speaker'] = passage.payload
                logging.info(f"Found {token} within {passage}")
           


def speakerize_ep(path):
    data = yaml.safe_load(open(path))
    for part in data['narrator_splits']:
        speakerize_tokens(part['context'])
    return data

def clean(text):
    import re
    pattern = r'\[[^()]*\]'
    return re.sub(pattern, '', text)

    

def meta(path):
    base = os.path.basename(path)
    bare = os.path.splitext(base)[0]
    return f"{os.path.dirname(path)}/{bare}.json" 

def episode_id(path):
    return int(path.split("/")[-3].split('_')[1])

def phonemes(phones):
    from pig.ipa import arpa2ipa
    ipa = [ arpa2ipa(p['phone'].split('_')[0]) for p in phones ]
    if None in ipa:
        raise ValueError(f"Unknown ARPA transcription {[p['phone'] for p in phones]}")
    else:
        return ''.join(ipa)
    
@dataclass
class Word:
    spelling: str
    phonemes: str
    duration: float
    speaker: str
    episode: int = None
    audio: m.AudioFileClip = None
    charngram: torch.Tensor = None
    fasttext: torch.Tensor = None
    glove: torch.Tensor = None

class WordData():

    def __init__(self, audio_paths, alignment_paths, min_duration=0.1):
        self.min_duration = min_duration
        self.items = list(zip(audio_paths, alignment_paths))

    def valid_alignment(self, word):
        return  word['case'] == 'success' \
            and word['alignedWord'] != '<unk>' \
            and word['end']-word['start'] >= self.min_duration \
            and 'sil' not in [ p['phone'] for p in word['phones'] ]

        
    def words(self, read_audio=True,
              charngram=None,
              fasttext=None,
              glove=None):
        for audio_path, alignment_path in self.items:
            meta = json.load(open(alignment_path))
            if read_audio:
                audio = m.AudioFileClip(audio_path)
            for word in meta['words']:
                if self.valid_alignment(word):
                    if read_audio:
                        logging.info(f"Extracting <{word['alignedWord']}> from {audio_path}")
                        sub = audio.subclip(word['start'], word['end'])
                    else:
                        sub = None
                    yield Word(spelling= word['alignedWord'],
                               phonemes= phonemes(word['phones']),
                               duration= word['end']-word['start'],
                               speaker= meta['speaker'],
                               episode= episode_id(audio_path),
                               audio= sub,
                               charngram= charngram[word['alignedWord']].squeeze(dim=0) \
                                 if charngram is not None else None,
                               fasttext= fasttext[word['alignedWord']] \
                                 if fasttext is not None else None,
                               glove= glove[word['alignedWord']] \
                                 if glove is not None else None)
                               

def normalized_distance(a, b):
    from Levenshtein import distance
    return distance(a, b) / max(len(a), len(b))
                    
def pairwise(fragment_type='dialog'):
    from pig.models import PeppaPig
    from pig.data import audioclip_loader
    from pig.util import cosine_matrix
    from torchtext.vocab import CharNGram, FastText, GloVe
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    audio_paths = glob.glob(f"data/out/realign/{fragment_type}/ep_*/*/*.wav")
    anno_paths  = [ meta(path) for path in audio_paths ]

    word_data = WordData(audio_paths, anno_paths, min_duration=0.1)
    
    net = PeppaPig.load_from_checkpoint("lightning_logs/version_31/checkpoints/epoch=101-step=17645-v1.ckpt")
    net.eval()
    net.cuda()
    with torch.no_grad():
        loader = audioclip_loader(word.audio for word in word_data.words(read_audio=True))
        emb = torch.cat([ net.encode_audio(batch.to(net.device)).squeeze(dim=1)
                          for batch in loader ])
    sim = cosine_matrix(emb, emb).cpu()
    logging.info(f"Computed similarities: {sim.shape}")
    words = [ word for word in word_data.words(read_audio=False,
                                               #charngram=CharNGram(),
                                               fasttext=FastText(),
                                               glove=GloVe(name='840B', dim=300)  ) ]
    for i, word1 in enumerate(words):
        logging.info(f"Processing word {i}")
        for j, word2 in enumerate(words):
            if i < j:
                yield dict(spelling1=word1.spelling,
                           phonemes1=word1.phonemes,
                           duration1=word1.duration,
                           speaker1=word1.speaker,
                           episode1=word1.episode,
                           spelling2=word2.spelling,
                           phonemes2=word2.phonemes,
                           duration2=word2.duration,
                           speaker2=word2.speaker,
                           episode2=word2.episode,
                           distance=normalized_distance(word1.phonemes, word2.phonemes),
                           #charngramsim=cos(word1.charngram, word2.charngram).item(),
                           fasttextsim=cos(word1.fasttext, word2.fasttext).item(),
                           glovesim=cos(word1.glove, word2.glove).item(),
                           sametype=word1.phonemes==word2.phonemes,
                           samespeaker=word1.speaker==word2.speaker,
                           sameepisode=word1.episode==word2.episode,
                           dialog=fragment_type=='dialog',
                           durationdiff=abs(word1.duration-word2.duration),
                           similarity=sim[i, j].item())
    
                    
def dump_data():
    for fragment_type in ['dialog', 'narration']:
        import pandas
        logging.getLogger().setLevel(level=logging.INFO)
        data = pandas.DataFrame.from_records(pairwise(fragment_type))
        data.to_csv(f"pairwise_similarities_{fragment_type}.csv", index=False, header=True)
