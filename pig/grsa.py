import yaml
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging
import glob
import moviepy.editor as m
import time
import json
import os
import os.path
import torch
import torch.nn
import pig.evaluation

VERSIONS=[48, 61]

def checkpoint_path(version):
    return f"lightning_logs/version_{version}/"

def as_yaml(episodes):
    for episode in episodes:
        data = json.load(open(f"data/in/peppa/episodes/ep_{episode}.json"))
        speakerize(data)
        yaml.dump(data, open(f"data/out/speaker_id/ep_{episode}.yaml", "w"))

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
class Utt:
    spelling: str
    duration: float
    speaker: str
    phonemes: str = None
    episode: int = None
    audio: m.AudioFileClip = None
    embedding_1: torch.tensor = None
    embedding_2: torch.tensor = None
    embedding_t: torch.tensor = None





class UttData():

    def __init__(self, audio_paths, alignment_paths, multiword=False):
        self.items = list(zip(audio_paths, alignment_paths))
        self.multiword = multiword
        self.min_duration = 0.0

    def valid_word_alignment(self, word):
        return  word['case'] == 'success' \
            and word['end']-word['start'] >= self.min_duration 

    def valid_multiword_alignment(self, words):
        return np.all([ word['case'] == 'success' for word in words]) \
              and words[-1]['end']-words[0]['start'] >= self.min_duration
    
    def words(self, read_audio=True,  embed=None):
        for audio_path, alignment_path in self.items:
            meta = json.load(open(alignment_path))
            if read_audio:
                audio = m.AudioFileClip(audio_path)
            for word in meta['words']:
                if self.valid_word_alignment(word):
                    if read_audio:
                        logging.info(f"Extracting <{word['word']}> from {audio_path}")
                        sub = audio.subclip(word['start'], word['end'])
                    else:
                        sub = None
                    yield Utt(spelling= word['word'],
                              duration= word['end']-word['start'],
                              speaker= meta['speaker'],
                              episode= episode_id(audio_path),
                              audio= sub,
                              embedding_t= embed[word['word']] \
                                 if embed is not None else None)
                                       
    def multiwords(self, read_audio=True,  embed=None):
        for audio_path, alignment_path in self.items:
            meta = json.load(open(alignment_path))
            if self.valid_multiword_alignment(meta['words']):
                if read_audio:
                    logging.info(f"Extracting utterance from {audio_path}")
                    audio = m.AudioFileClip(audio_path).subclip(meta['words'][0]['start'],
                                                                meta['words'][-1]['end'])
                else:
                    audio = None
                text = " ".join((word['word'] for word in meta['words']))
                logging.info(f"Embedding extracted utterance")
                embedding_t  = embed(text) if embed is not None else None
                yield Utt(spelling= text,
                          duration= meta['words'][-1]['end']-meta['words'][0]['start'],
                          speaker = meta['speaker'],
                          episode = episode_id(audio_path),
                          audio   = audio,
                          embedding_t = embedding_t)


    def utterances(self, **kwargs):
        if self.multiword:
            yield from self.multiwords(**kwargs)
        else:
            yield from self.words(**kwargs)
                
def normalized_distance(a, b):
    from Levenshtein import distance
    return distance(a, b) / max(len(a), len(b))

def pairwise(version, fragment_type='dialog', multiword=False):
    from pig.models import PeppaPig
    from pig.data import audioclip_loader
    from pig.util import cosine_matrix
    from torchtext.vocab import GloVe
    from sentence_transformers import SentenceTransformer
    from copy import deepcopy
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    audio_paths = glob.glob(f"data/out/realign/{fragment_type}/ep_*/*/*.wav")
    anno_paths  = [ meta(path) for path in audio_paths ]

    data = UttData(audio_paths, anno_paths, multiword=multiword)

        
    net_2, net_path = pig.evaluation.load_best_model(checkpoint_path(version))
    net_1 = PeppaPig(net_2.config)
    net_2.eval(); net_2.cuda()
    net_1.eval(); net_1.cuda()
    with torch.no_grad():
        loader = audioclip_loader(utt.audio for utt in data.utterances(read_audio=True))
        emb_1, emb_2 = zip(*[ (net_1.encode_audio(batch.to(net_1.device)).squeeze(dim=1),
                               net_2.encode_audio(batch.to(net_1.device)).squeeze(dim=1))
                              for batch in loader ])
    emb_1 = torch.cat(emb_1)
    emb_2 = torch.cat(emb_2)
    sim_1 = cosine_matrix(emb_1, emb_1).cpu()
    sim_2 = cosine_matrix(emb_2, emb_2).cpu()
    logging.info(f"Computed similarities: {sim_2.shape}")
    glove_model = GloVe(name='840B', dim=300)
    if multiword:
        encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        def avg_glove(s):
            return torch.stack([ glove_model[word] for word in s.split() ]).sum(dim=0)
        
        #utts = [ utt for utt in data.utterances(read_audio=False,
        #                                        embed=lambda u: encoder.encode([u], convert_to_tensor=True)[0]) ]
        utts = [ utt for utt in data.utterances(read_audio=False, embed=avg_glove) ]
    else:
        utts = [ utt for utt in data.utterances(read_audio=False,
                                                embed=glove_model ) ]
    for i, utt in enumerate(utts):
        utt.embedding_1 = emb_1[i]
        utt.embedding_2 = emb_2[i]
    torch.save(dict(path=net_path, version=version, utt=utts), f"data/out/utt_{'multi' if multiword else ''}word_{version}_{fragment_type}.pt")
    for i, utt1 in enumerate(utts):
        logging.info(f"Processing word {i}")
        for j, utt2 in enumerate(utts):
            if i < j:
                yield dict(spelling1=utt1.spelling,
                           phonemes1=utt1.phonemes,
                           duration1=utt1.duration,
                           speaker1=utt1.speaker,
                           episode1=utt1.episode,
                           spelling2=utt2.spelling,
                           phonemes2=utt2.phonemes,
                           duration2=utt2.duration,
                           speaker2=utt2.speaker,
                           episode2=utt2.episode,
                           distance=normalized_distance(utt1.phonemes, utt2.phonemes)
                                if utt1.phonemes is not None and utt2.phonemes is not None else None,
                           semsim=cos(utt1.embedding_t, utt2.embedding_t).item(),
                           sametype=utt1.spelling==utt2.spelling,
                           samespeaker=None if utt1.speaker is None or utt2.speaker is None else utt1.speaker==utt2.speaker,
                           sameepisode=utt1.episode==utt2.episode,
                           dialog=fragment_type=='dialog',
                           durationdiff=abs(utt1.duration-utt2.duration),
                           sim_1=sim_1[i, j].item(),
                           sim_2=sim_2[i, j].item())

def unpairwise(version):
    dialog = torch.load(f"data/out/utt_multiword_{version}_dialog.pt")
    narration = torch.load(f"data/out/utt_multiword_{version}_narration.pt")
    utt = [utt for utt in dialog['utt'] + narration['utt'] if utt.speaker is not None]
    data = pd.DataFrame.from_records(unpairwise_data(utt))
    data.to_csv(f"data/out/unpairwise_similarities_{version}.csv", index=False, header=True)
    
def unpairwise_data(utt):
    import random
    from pig.triplet import pairs
    cosine = torch.nn.CosineSimilarity(dim=1)
    random.shuffle(utt)
    p1, p2 = zip(*pairs(utt))
    sim_2 = cosine(torch.stack([ x.embedding_2 for x in p1]),
                   torch.stack([ x.embedding_2 for x in p2]))
    sim_1 = cosine(torch.stack([ x.embedding_1 for x in p1]),
                   torch.stack([ x.embedding_1 for x in p2]))
    semsim = cosine(torch.stack([ x.embedding_t for x in p1]),
                    torch.stack([ x.embedding_t for x in p2]))
    for i in range(len(p1)):
        yield dict(spelling1 =p1[i].spelling,
                   duration1 =p1[i].duration,
                   speaker1  =p1[i].speaker,
                   episode1  =p1[i].episode,
                   spelling2 =p2[i].spelling,
                   duration2 =p2[i].duration,
                   speaker2  =p2[i].speaker,
                   episode2  =p2[i].episode,
                   sametype  =p1[i].spelling==p2[i].spelling,
                   samespeaker  =None if p1[i].speaker is None or p2[i].speaker is None else p1[i].speaker==p2[i].speaker,
                   sameepisode  =p1[i].episode==p2[i].episode,
                   durationdiff =abs(p1[i].duration - p2[i].duration),
                   durationsum  =p1[i].duration + p2[i].duration,
                   distance     =normalized_distance(p1[i].spelling, p2[i].spelling),
                   semsim       = semsim[i].item(),
                   sim_1        = sim_1[i].item(),
                   sim_2        = sim_2[i].item())
                   

def word_type():
    from pig.util import grouped, triu, pearson_r, cosine_matrix
    rows = []
    for fragment_type in ['dialog', 'narration']:
        data = torch.load(f"data/out/words_{fragment_type}.pt")
        model_version = CHECKPOINT_PATH
        embedding = []
        glove = []
        for typ, toks in grouped(data['words'], key=lambda w: w.spelling):
            toks = list(toks)
            if toks[0].glove.sum() != 0.0:
                embedding.append(torch.stack([ tok.embedding_2 for tok in toks]).mean(dim=0))
                glove.append(toks[0].glove)
        embedding = torch.stack(embedding)
        glove = torch.stack(glove)
        sim_emb = triu(cosine_matrix(embedding, embedding).cpu())
        sim_glove = triu(cosine_matrix(glove.double(), glove.double()).cpu())
        rows.append(dict(fragment_type=fragment_type,
                         pearson_r=pearson_r(sim_emb, sim_glove).item(),
                         N=sim_emb.shape[0],
                         model_version=model_version))
    pd.DataFrame.from_records(rows).to_csv("results/word_type_rsa.csv", index=False, header=True)
        
        
def main(versions=VERSIONS):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import pandas
    logging.getLogger().setLevel(level=logging.INFO)
    tables = []
    for version in versions:
        for fragment_type in ['dialog', 'narration']:
            for multiword in [True, False]:
                data = pandas.DataFrame.from_records(pairwise(version,
                                                              fragment_type=fragment_type,
                                                              multiword=multiword))
                data['version'] = version
                data['fragment_type'] = fragment_type
                data['multiword'] = multiword
                tables.append(data)

    table = pd.concat(tables)
    table.to_csv(f"data/out/pairwise_similarities.csv", index=False, header=True, na_rep="NA")
