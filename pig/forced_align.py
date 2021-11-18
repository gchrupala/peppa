import json
import gentle
import logging
import multiprocessing
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
    
