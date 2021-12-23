# Peppa

## Installation

Install prerequistes (preferably inside a virtual python environment):
```
pip install -r requirements.txt
```


## Get data

- Get the pretrained wav2vec model (small, no finetuning) from https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt and unpack it into `peppa/data/in/wav2vec/`
- Get the preprocessed Peppa video snippets from https://surfdrive.surf.nl/files/index.php/s/S9HA9wicV7kCwet and unpack them into `peppa/data/out/`.


## Run

### Train
There is a rudimentary command-line interface. You can run the training code by executing the function script `run.py`, and optionally passing 
in a configuration file.
```
python run.py --config_file config.yaml
```
Example configuration files are in [lightning_logs](lightning_logs) (named `hparams.yaml`).


### Use
You can find saved checkpoints of trained models at https://surfdrive.surf.nl/files/index.php/s/gNnZ4iSoKBDsOGK. 
See [example.py](example.py) for an example of how to load a model and use it to encode some audio files.

### Analyze

Generate tables in [results/](results/).

```
python -c 'import pig.evaluation as m; m.main(0)'
```

Generate tables and figures in [results/](results/).

```
python -c  'import pig.grsa as m; m.main()'
python -c  'import pig.stats as m; m.main()'
```



