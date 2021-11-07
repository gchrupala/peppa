import os
import glob
import json
import moviepy.editor as m
import logging
import pandas as pd
import random


def extract(target_size=(180, 100)):
    logging.basicConfig(level=logging.INFO)
    os.makedirs("data/out/dialog", exist_ok=True)
    os.makedirs("data/out/narration", exist_ok=True)
    data = pd.read_csv("data/in/peppa_pig_dataset-video_list.csv", sep=';', quotechar="'", names=["id", "title", "path"], index_col=0)
    
    titles = dict(zip(data['title'], data['path'].map(lambda x: f'data/in/peppa/{x[4:]}')))
    episodes = glob.glob("data/in/peppa/episodes/*.json")
    for path in episodes:
        annotation = json.load(open(path))
        with m.VideoFileClip(titles[annotation['title']]) as video:
            extract_from_episode(annotation, video, target_size=target_size)
            video.close()
        


    
def extract_from_episode(annotation, video, target_size):
    width, height = target_size
    narrations = []
    narrations_meta = []
    dialogs = []
    dialogs_meta = []
    for segment in annotation['narrator_splits']:
        if len(segment['context']['tokenized']) > 0:
            dialogs.append(video.subclip(segment['context']['tokenized'][0]['begin'],
                                         segment['context']['tokenized'][-1]['end']))
            dialogs_meta.append(segment['context'])
        if len(segment['narration']['tokenized']) > 0:
            narrations.append(video.subclip(segment['narration']['tokenized'][0]['begin'],
                                            segment['narration']['tokenized'][-1]['end']))
            narrations_meta.append(segment['narration'])
    os.makedirs(f"data/out/{width}x{height}/dialog/{annotation['id']}", exist_ok=True)
    os.makedirs(f"data/out/{width}x{height}/narration/{annotation['id']}", exist_ok=True)
    for i, clip in enumerate(dialogs):
        
        logging.info(f"Writing dialog {i} from episode {annotation['id']}") 
        clip.resize(target_size).write_videofile(f"data/out/{width}x{height}/dialog/{annotation['id']}/{i}.avi",
                                         fps=10,
                                         codec='mpeg4')
        logging.info(f"Writing dialog {i} from episode {annotation['id']}") 
        json.dump(dialogs_meta[i], open(f"data/out/{width}x{height}/dialog/{annotation['id']}/{i}.json", 'w'))
    for i, clip in enumerate(narrations):
        
        logging.info(f"Writing narration {i} from episode {annotation['id']}") 
        clip.resize(target_size).write_videofile(f"data/out/{width}x{height}/narration/{annotation['id']}/{i}.avi",
                                         fps=10,
                                         codec='mpeg4')
        logging.info(f"Writing narration metadata {i} from episode {annotation['id']}") 
        json.dump(narrations_meta[i], open(f"data/out/{width}x{height}/narration/{annotation['id']}/{i}.json", 'w'))
        
def lines(clip, metadata):
    start = pd.Timedelta(metadata['subtitles'][0]['begin'])
    logging.info(f"Extracting lines from {clip.filename}, {clip.duration} seconds")
    logging.info(f"Time offset {start}")
    for line in metadata['subtitles']:
        #logging.info(f"Line: {line}")
        begin = (pd.Timedelta(line['begin'])-start).seconds
        end = min(clip.duration, (pd.Timedelta(line['end'])-start).seconds)
        if begin < clip.duration:
            yield clip.subclip(begin, end)
        else:
            logging.warning(f"Line {line} starts past end of clip {clip.filename}")
            
                           
    
    
def segment(clip, duration=3.2, randomize=False, factor=1):
    offset = random.uniform(0, duration/factor) if randomize else 0
    start = 0 + offset
    end = start + duration
    while end <= clip.duration:
        sub = clip.subclip(start, end)
        start = end
        end   = end+duration
        yield sub
