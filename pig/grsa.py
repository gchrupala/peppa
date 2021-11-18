import yaml
from dataclasses import dataclass
import pandas as pd
import logging
import glob
import moviepy.editor as m
import time
import json
from pig.forced_align import align
import os

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

def realign(tokens=False):
    data = pd.read_csv("data/in/peppa_pig_dataset-video_list.csv", sep=';', quotechar="'",
                       names=["id", "title", "path"], index_col=0)
    titles = dict(zip(data['title'], data['path'].map(lambda x: f'data/in/peppa/{x[4:]}')))
    #episodes = glob.glob("data/out/speaker_id/*.yaml")
    epids = [197, 198, 199, 200, 201, 202]
    
    for  epid in epids:
        path = f"data/out/speaker_id/ep_{epid}.yaml"
        annotation = yaml.safe_load(open(path))
        with m.AudioFileClip(titles[annotation['title']]) as audio:
            for i, part in enumerate(annotation['narrator_splits']):
                for j, sub in enumerate(part['context']['subtitles']):
                    transcript = clean(sub['text'])
                    if len(transcript) > 0:
                        os.makedirs(f"data/out/realign/ep_{epid}/{i}/", exist_ok=True)
                        start = pd.Timedelta(sub['begin'])-pd.Timedelta(seconds=0.5)
                        end = pd.Timedelta(sub['end'])+pd.Timedelta(seconds=0.5)
                        audiopath = f"data/out/realign/ep_{epid}/{i}/{j}.wav"
                        audio.subclip(start.seconds, end.seconds).write_audiofile(audiopath)
                        result = align(audiopath, transcript)
                        result['speaker'] = sub['speaker']
                        json.dump(result, open(f"data/out/realign/ep_{epid}/{i}/{j}.json", "w"), indent=2)

def featurize(tokens=False):
    data = pd.read_csv("data/in/peppa_pig_dataset-video_list.csv", sep=';', quotechar="'",
                       names=["id", "title", "path"], index_col=0)
    titles = dict(zip(data['title'], data['path'].map(lambda x: f'data/in/peppa/{x[4:]}')))
    #episodes = glob.glob("data/out/speaker_id/*.yaml")
    episodes = glob.glob("data/in/peppa/episodes/ep_198.json")
    for path in episodes:
        #annotation = yaml.safe_load(open(path))
        annotation = json.load(open(path))
        print(path)
        with m.VideoFileClip(titles[annotation['title']]) as video:
            for part in annotation['narrator_splits']:
                if tokens:
                    for token in part['context']['tokenized']:
                        if pd.Timedelta(token['end'])-pd.Timedelta(token['begin']) >= pd.Timedelta(seconds=1):
                            clip = video.audio.subclip(t_start=token['begin'], t_end=token['end'])
                            print(token['token'], clip.duration)
                            clip.preview()
                            time.sleep(2)
                else:
                    for line in part['context']['subtitles']:
                        if pd.Timedelta(line['end'])-pd.Timedelta(line['begin']) >= pd.Timedelta(seconds=1):
                            
                            clip = video.audio.subclip(t_start=line['begin'],
                                                       t_end=line['end'])
                            print(line['text'])
                            clip.preview()
                            time.sleep(2)
                    
