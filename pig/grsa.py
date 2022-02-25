import yaml
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging
import glob
import moviepy.editor as m
import json
import os
import os.path
import torch
import torch.nn
import evaluation
import random
import plotnine as pn
from pig.models import PeppaPig
from pig.data import audioclip_loader, audioarray_loader, \
    grouped_audioclip_loader, grouped_audioarray_loader
from sentence_transformers import SentenceTransformer
from torchtext.vocab import GloVe
from copy import deepcopy
    
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

def embed_utterances(version, fragment_type='dialog', grouped=True, embedder='st', projection=False):
    audio_paths = glob.glob(f"data/out/realign/{fragment_type}/ep_*/*/*.wav")
    anno_paths  = [ meta(path) for path in audio_paths ]

    data = UttData(audio_paths, anno_paths, multiword=True)

        
    net_2, net_path = evaluation.load_best_model(checkpoint_path(version))
    config_1 = deepcopy(net_2.config)
    config_1['audio']['pooling'] = 'average'
    config_1['audio']['project'] = projection
    net_1 = PeppaPig(config_1)
    net_2.eval(); net_2.cuda()
    net_1.eval(); net_1.cuda()
    with torch.no_grad():
        if grouped:
            loader = grouped_audioclip_loader(utt.audio for utt in data.utterances(read_audio=True))
        else:
            loader = audioclip_loader(utt.audio for utt in data.utterances(read_audio=True))

        emb_1, emb_2 = zip(*[ (net_1.encode_audio(batch.to(net_1.device)).squeeze(dim=1),
                               net_2.encode_audio(batch.to(net_2.device)).squeeze(dim=1))
                              for batch in loader ])
    emb_1 = torch.cat(emb_1)
    emb_2 = torch.cat(emb_2)
    if embedder == 'st':
        encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embed = lambda u: encoder.encode([u], convert_to_tensor=True)[0]
    elif embedder == 'glove':
        glove_model = GloVe(name='840B', dim=300)
        embed = lambda s: torch.stack([ glove_model[word] for word in s.split() ]).sum(dim=0)
    utts = [ utt for utt in data.utterances(read_audio=False,  embed=embed) ]
    for i, utt in enumerate(utts):
        utt.embedding_1 = emb_1[i]
        utt.embedding_2 = emb_2[i]
    return utts

def pairwise(version, fragment_type='dialog', multiword=False):
    from pig.models import PeppaPig
    from pig.data import audioclip_loader
    from pig.util import cosine_matrix
    from torchtext.vocab import GloVe
    from sentence_transformers import SentenceTransformer
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    audio_paths = glob.glob(f"data/out/realign/{fragment_type}/ep_*/*/*.wav")
    anno_paths  = [ meta(path) for path in audio_paths ]

    data = UttData(audio_paths, anno_paths, multiword=multiword)

        
    net_2, net_path = evaluation.load_best_model(checkpoint_path(version))
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

def unpairwise(version, grouped=True, embedder='st', n_samples=100):
    from pig.stats import unpairwise_ols 
    dialog = embed_utterances(version, "dialog", grouped=grouped, embedder=embedder,
                              projection=True)
    narration = embed_utterances(version, "narration", grouped=grouped, embedder=embedder,
                                 projection=True)
    utt = [utt for utt in dialog + narration if utt.speaker is not None]
    results = []
    for n in range(n_samples):
        data = pd.DataFrame.from_records(unpairwise_data(utt))
        result = unpairwise_ols(data)
        result['sample'] = n
        results.append(result)
    pd.concat(results).to_csv("results/unpairwise_coef.csv", index=False, header=True)
    g = pn.ggplot(results, pn.aes(x='Variable', y='Value', color='Dependent Var.')) + \
        pn.geom_boxplot() + \
        pn.coord_flip()
    pn.ggsave(g, "results/unpairwise_boxplots.pdf")


    
def unpairwise_data(utt):
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

def prepare_probe(embedder, feature, label, balanced=True):
    X_d, Y_d = embedder.feature_label('dialog', feature, label)
    X_n, Y_n = embedder.feature_label('narration', feature, label)
    if balanced:
        N = len(Y_d)
        ixs = random.sample(range(len(Y_n)), N)
        X = np.concatenate([X_d, X_n[ixs]])
        Y = np.concatenate([Y_d, Y_n[ixs]])
    else:
        X = np.concatenate([X_d, X_n])
        Y = np.concatenate([Y_d, Y_n])
    return X, Y

def probe(embedder, labels=['speaker']):
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, scale
    from collections import Counter
    records = []
    for label in labels:
        for feature in embedder.embedding['dialog'].keys():
            if label == 'speaker':
                X, Y = prepare_probe(embedder, feature, label, balanced=True)
            else:
                X, Y = prepare_probe(embedder, feature, label, balanced=False)
            if label == 'duration':
                model = GridSearchCV(make_pipeline(StandardScaler(),
                                                   MLPRegressor(max_iter=1000)),
                                     param_grid={'mlpregressor__alpha':
                                                 [10**n for n in range(-4, 5) ]},
                                     n_jobs=12)
                model.fit(X, scale(Y))
                score = model.best_score_
                records.append(dict(model='ridge', label=label, feature=feature,
                                    maj=None, score=score))
            else:
                count = Counter(Y)
                maj = max(count.values())/sum(count.values())
                Y = np.array([ z if count[z]>4 else 'other' for z in Y])
                model = GridSearchCV(make_pipeline(StandardScaler(),
                                                   MLPClassifier(max_iter=1000)),
                                     param_grid={'mlpclassifier__alpha': [0.1, 1.0, 10],
                                                 'mlpclassifier__hidden_layer_sizes':
                                                 [(50,), (100,), (200,)]},
                                     n_jobs=12)
                model.fit(X, Y)
                score = rer(model.best_score_, maj)
                records.append(dict(model='lr', label=label, feature=feature, maj=maj, score=score))
    return pd.DataFrame.from_records(records)

def vanilla_rsa(embedder, labels=['speaker']):
    from pig.util import pearson_r, triu, cosine_matrix
    records = []
    for label in labels:
        for feature in embedder.embedding['dialog'].keys():
            X, Y = prepare_probe(embedder, feature, label)
            X = torch.tensor(X)
            X_sim = cosine_matrix(X, X)
            Y_sim = torch.tensor([[y1 == y2 for y1 in Y] for y2 in Y]).float()
            r = pearson_r(triu(X_sim), triu(Y_sim)).item()
            records.append(dict(label=label, feature=feature, r=r))
    return pd.DataFrame.from_records(records)

def rer(hi_acc, low_acc):
    return ((1-low_acc)-(1-hi_acc))/(1-low_acc)


class Embedder:
    def __init__(self, version):
        self.version = version
        self.data = {}
        self.audio    = dict(dialog=[], narration=[])
        self.duration  = dict(dialog=[], narration=[])
        self.speaker  = dict(dialog=[], narration=[])
        self.spelling = dict(dialog=[], narration=[])
        self.embedding = dict(dialog={}, narration={})
        for fragment_type in ['dialog', 'narration']:
            audio_paths = glob.glob(f"data/out/realign/{fragment_type}/ep_*/*/*.wav")
            anno_paths  = [ meta(path) for path in audio_paths ]
            self.data[fragment_type] = UttData(audio_paths, anno_paths, multiword=False)
            
    def load_audio(self):
        for fragment_type in self.audio:
            for utt in self.data[fragment_type].utterances(read_audio=True):
                self.audio[fragment_type].append(pig.data.featurize_audio(utt.audio, samplerate=44100))
                self.speaker[fragment_type].append(utt.speaker)
                self.spelling[fragment_type].append(utt.spelling)
                self.duration[fragment_type].append(utt.duration)
        
    def embed(self, grouped=True):
        net_2, net_path = evaluation.load_best_model(checkpoint_path(self.version))
        net_2.eval().cuda()
        net_1 = PeppaPig(net_2.config).eval().cuda()
        config_0 = deepcopy(net_2.config)
        config_0['audio']['pretrained'] = False
        net_0 = PeppaPig(config_0).eval().cuda()
        embed_untrained = lambda batch: net_0.encode_audio(batch.to(net_0.device)).squeeze(dim=1)
        embed_trained = lambda batch: net_2.encode_audio(batch.to(net_2.device)).squeeze(dim=1)
        embed_project     = lambda batch: net_1.encode_audio(batch.to(net_1.device)).squeeze(dim=1)
        def embed_wav2vec(batch):
            feat, _ = net_2.audio_encoder.audio.extract_features(batch.to(net_2.device).squeeze(dim=1))
            return feat.mean(dim=1)
        def embed_conv(batch):
            feat, _ = net_2.audio_encoder.audio.feature_extractor(batch.to(net_2.device).squeeze(dim=1),
                                                                  None)
            return feat.mean(dim=1)
        for fragment_type in self.embedding:
            with torch.no_grad():
                if grouped:
                    loader = grouped_audioarray_loader
                else:
                    loader = audioarray_loader
                self.embedding[fragment_type]['untrained'] = \
                    torch.cat([embed_untrained(batch) for batch
                               in loader(self.audio[fragment_type])]).cpu().numpy()
                self.embedding[fragment_type]['trained'] = \
                    torch.cat([embed_trained(batch) for batch
                               in loader(self.audio[fragment_type])]).cpu().numpy()
                self.embedding[fragment_type]['project'] = \
                    torch.cat([embed_project(batch) for batch
                               in loader(self.audio[fragment_type])]).cpu().numpy()
                self.embedding[fragment_type]['wav2vec'] = \
                    torch.cat([embed_wav2vec(batch) for batch
                               in loader(self.audio[fragment_type])]).cpu().numpy()
                self.embedding[fragment_type]['conv'] = \
                    torch.cat([embed_conv(batch) for batch
                               in loader(self.audio[fragment_type])]).cpu().numpy()

    def feature_label(self, fragment_type, feature, label):
        X = self.embedding[fragment_type][feature]
        Y = getattr(self, label)[fragment_type]
        X, Y = zip(*[(x,y) for x,y in zip(X, Y) if y is not None])
        return np.array(list(X)), np.array(list(Y))
    
def feat_speak(fragment_type, embed=None, version=None):
    """Either embed or version must be specified."""
    if embed is None:
        net_2, net_path = evaluation.load_best_model(checkpoint_path(version))
        net_2.eval(); net_2.cuda()
        embed = lambda batch: net_2.encode_audio(batch.to(net_2.device)).squeeze(dim=1)
    loader = audioclip_loader(utt.audio for utt in data.utterances(read_audio=True))
    with torch.no_grad():
        emb_2 = torch.cat([embed(batch) for batch in loader ])
    Y = [x.speaker for x in data.utterances(read_audio=False)]
    X, Y = zip(*[(x,y) for x,y in zip(emb_2, Y) if y is not None])
    return torch.stack(X).cpu().numpy(), np.array(Y)
    
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
