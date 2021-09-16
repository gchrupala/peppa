import os
import glob
import json
import moviepy.editor as m
import logging
import pandas as pd


def extract():
    logging.basicConfig(level=logging.INFO)
    os.makedirs("data/out/dialog", exist_ok=True)
    os.makedirs("data/out/narration", exist_ok=True)
    data = pd.read_csv("data/in/peppa_pig_dataset-video_list.csv", sep=';', quotechar="'", names=["id", "title", "path"], index_col=0)
    
    titles = dict(zip(data['title'], data['path'].map(lambda x: f'data/in/peppa/{x[4:]}')))
    episodes = glob.glob("data/in/peppa/episodes/*.json")
    for path in episodes:
        annotation = json.load(open(path))
        with m.VideoFileClip(titles[annotation['title']]) as video:
            extract_from_episode(annotation, video)
            video.close()
        


    
def extract_from_episode(annotation, video):
    narrations = []
    dialogs = []
    
    for segment in annotation['narrator_splits']:
        if len(segment['context']['tokenized']) > 0:
            dialogs.append(video.subclip(segment['context']['tokenized'][0]['begin'],
                                     segment['context']['tokenized'][-1]['end']))
        if len(segment['narration']['tokenized']) > 0:
            narrations.append(video.subclip(segment['narration']['tokenized'][0]['begin'],
                                            segment['narration']['tokenized'][-1]['end']))
    os.makedirs(f"data/out/dialog/{annotation['id']}", exist_ok=True)
    os.makedirs(f"data/out/narration/{annotation['id']}", exist_ok=True)
    for i, clip in enumerate(dialogs):
        
        logging.info(f"Writing dialog {i} from episode {annotation['id']}") 
        clip.resize(1/4).write_videofile(f"data/out/dialog/{annotation['id']}/{i}.avi",
                                         fps=10,
                                         codec='mpeg4')
    for i, clip in enumerate(narrations):
        
        logging.info(f"Writing narration {i} from episode {annotation['id']}") 
        clip.resize(1/4).write_videofile(f"data/out/narration/{annotation['id']}/{i}.avi",
                                         fps=10,
                                         codec='mpeg4')
        
        
def segment(clip, duration=3.2):
    start = 0
    end = duration
    while end <= clip.duration:
        yield clip.subclip(start, end)
        start = end
        end   = end+duration
        
