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
python run.py --config_file hparams_base.yaml
```
Configuration files for the ablation experiments in the paper are in the repository, e.g.: [hparams_base.yaml](hparams_base.yaml).



### Use
The file [conditions.yaml](conditions.yaml) specifies the run IDs for each ablation configuration.

The corresponding trained models can be found at https://surfdrive.surf.nl/files/index.php/s/gNnZ4iSoKBDsOGK. 

### Evaluate

Create evaluation data files for trained model checkpoints as specified in the `--versions` argument:
```
python evaluate.py --versions 335
```

After generating evaluation data for all the necessary runs, you can:

- generate figures 3-7 in [results/ablations](results/ablations)
```
python -c 'import pig.plotting as m; m.plots()'
```

- generate table 2 in [results/scores_test.tex](results/scores_test.tex)
 ```
 python -c 'import pig.evaluation as m; m.test_run(); m.test_table()'
 ```


Run the minimal pairs evaluation:
```
python evaluation_targeted_triplets.py --versions 335
```

