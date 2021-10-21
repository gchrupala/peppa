# Peppa

## Installation

Install prerequistes (preferably inside a virtual python environment):
```
pip install -r requirements.txt
```


## Get data

- Get the pretrained wav2vec model (small, no finetuning) from `https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt` and unpack it into `peppa/data/in/wav2vec/`
- Get the preprocessed Peppa video snippets from `https://surfdrive.surf.nl/files/index.php/s/S9HA9wicV7kCwet` and unpack them into `peppa/data/out/`.


## Run

There is no command-line interface. You can run the training code by executing the function `pig.models.main`:
```
python -c 'import pig.models as m; m.main()'
```


