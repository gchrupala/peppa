import yaml
from dataclasses import dataclass
import pandas as pd
import logging
import glob
import moviepy.editor as m
import time
import json


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
                        clip = video.audio.subclip(t_start=token['begin'], t_end=token['end'])
                        print(token['token'])
                        clip.preview()
                        time.sleep(1)
                else:
                    for line in part['context']['subtitles']:
                        clip = video.audio.subclip(t_start=line['begin'],
                                                   t_end=line['end'])
                        print(line['text'])
                        clip.preview()
                        time.sleep(3)
                    
