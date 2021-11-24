import json
import gentle
import logging
import multiprocessing
import pandas as pd

nthreads = multiprocessing.cpu_count()

resources = gentle.Resources()

def on_progress(p):
    for k,v in p.items():
        logging.debug(f"{k}: {v}")


def align(audiopath, transcript):
    logging.info("converting audio to 8K sampled wav")
    with gentle.resampled(audiopath) as wavfile:
        logging.info("starting alignment")
        aligner = gentle.ForcedAligner(resources, transcript, nthreads=nthreads, disfluency=False, 
                                       conservative=False)
        return json.loads(aligner.transcribe(wavfile,
                                            progress_cb=on_progress, logging=logging).to_json())

def realign_all():
    for fragment_type in ['dialog', 'narration']:
        realign(fragment_type)
        
def realign(fragment_type='dialog'):
    from pig.data import SPLIT_SPEC
    import yaml
    import moviepy.editor as m
    names = dict(narration='narration', dialog='context') 
    data = pd.read_csv("data/in/peppa_pig_dataset-video_list.csv", sep=';', quotechar="'",
                       names=["id", "title", "path"], index_col=0)
    titles = dict(zip(data['title'], data['path'].map(lambda x: f'data/in/peppa/{x[4:]}')))
    for epid in SPLIT_SPEC[fragment_type]['val']:
        if fragment_type=='dialog':
            path = f"data/out/speaker_id/ep_{epid}.yaml"
            annotation = yaml.safe_load(open(path))
        else:
            path = f"data/in/peppa/episodes/ep_{epid}.json"
            annotation = json.load(open(path))
        with m.AudioFileClip(titles[annotation['title']]) as audio:
            for i, part in enumerate(annotation['narrator_splits']):
                for j, sub in enumerate(part[names[fragment_type]]['subtitles']):
                    transcript = clean(sub['text'])
                    if len(transcript) > 0:
                        os.makedirs(f"data/out/realign/{fragment_type}/ep_{epid}/{i}/",
                                    exist_ok=True)
                        start = pd.Timedelta(sub['begin'])-pd.Timedelta(seconds=0.5)
                        end = pd.Timedelta(sub['end'])+pd.Timedelta(seconds=0.5)
                        audiopath = f"data/out/realign/{fragment_type}/ep_{epid}/{i}/{j}.wav"
                        audio.subclip(start.seconds, end.seconds).write_audiofile(audiopath)
                        result = align(audiopath, transcript)
                        result['speaker'] = sub.get('speaker') if fragment_type == 'dialog' else 'Narrator'
                        result['episode_filepath'] = titles[annotation['title']]
                        result['episode_metadata_path'] = path
                        result['episode_title'] = annotation['title']
                        result['clipStart'] = start
                        result['clipEnd'] = end
                        result['partIndex'] = i
                        result['clipIndex'] = j
                        json.dump(result,
                                  open(f"data/out/realign/{fragment_type}/ep_{epid}/{i}/{j}.json",
                                       "w"), indent=2)

def clean(text):
    import re
    pattern = r'\[[^()]*\]'
    return re.sub(pattern, '', text)
                        
