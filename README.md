# Peppa

## Installation

Install prerequistes (preferably inside a virtual python environment):
```
pip install -r requirements.txt
```


## Get data

Get preprocessed video snippets from `https://surfdrive.surf.nl/files/index.php/s/S9HA9wicV7kCwet` and unpack them into `peppa/data/out`.


## Run

There is no command-line interface. You can run the training code by executing the function `pig.models.main`:
```
python -c 'import pig.models as m; m.main()'
```


