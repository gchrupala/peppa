import os
import glob
import json
import moviepy.editor as m
import logging


def extract_dialog():
    logging.basicConfig(level=logging.INFO)
    os.makedirs("data/out/dialog", exist_ok=True)
    os.makedirs("data/out/narration", exist_ok=True)

    episodes = glob.glob("data/in/peppa/episodes/*.json")
    annotation = json.load(open("data/in/peppa/episodes/ep_1.json"))
    video = m.VideoFileClip("data/in/peppa/videos/S01/S01E01 - Muddy Puddles.avi")
    extract_from_episode(annotation, video)


    
def extract_from_episode(annotation, video):
    narrations = []
    dialogs = []
    
    for segment in annotation['narrator_splits']:
        dialogs.append(video.subclip(segment['context']['tokenized'][0]['begin'],
                                     segment['context']['tokenized'][-1]['end']))
        narrations.append(video.subclip(segment['narration']['tokenized'][0]['begin'],
                                        segment['narration']['tokenized'][-1]['end']))
    os.makedirs(f"data/out/dialog/{annotation['id']}", exist_ok=True)
    os.makedirs(f"data/out/narration/{annotation['id']}", exist_ok=True)
    for i, clip in enumerate(dialogs):
        logging.info(f"Writing dialog {i} from episode {annotation['id']}") 
        clip.write_videofile(f"data/out/dialog/{annotation['id']}/{i}.avi", codec='mpeg4')
    for i, clip in enumerate(narrations):
        clip.write_videofile(f"data/out/narration/{annotation['id']}/{i}.avi", codec='mpeg4')
        
        
        
        
