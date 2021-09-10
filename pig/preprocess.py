import os
import glob
import json
import moviepy.editor as m
import logging


def title(name):
    words = name.split()
    if words[1] == '-':
        t = " ".join(words[2:])[:-4]
    else:
        t = " ".join(words[1:])[:-4]
    # Fix broken titles
    if t == 'Captian Daddy Pig':
        t = 'Captain Daddy Pig'
    if t == 'Wishing Well':
        t = 'The Wishing Well'
    return t

def extract_dialog():
    logging.basicConfig(level=logging.INFO)
    os.makedirs("data/out/dialog", exist_ok=True)
    os.makedirs("data/out/narration", exist_ok=True)
    titles = dict((title(path), path)  for path  in glob.glob("data/in/peppa/videos/*/*.avi"))
    episodes = glob.glob("data/in/peppa/episodes/*.json")
    for path in episodes:
        annotation = json.load(open(path))
        if annotation['title'] in titles:
            video = m.VideoFileClip(titles[annotation['title']])
            extract_from_episode(annotation, video)
            video.close()
        else:
            logging.warning(f"Video file for {annotation['title']} not found") 


    
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
        clip.resize(1/4).write_videofile(f"data/out/narration/{annotation['id']}/{i}.avi",
                                         fps=10,
                                         codec='mpeg4')
        
        
def segment(clip, duration=3.2):
    start = 0
    end = duration
    while start < clip.duration:
        yield clip.subclip(start, end)
        start = end
        end   = end+duration
        
