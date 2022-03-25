#!/bin/bash
#SBATCH -p GPUExtended # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-12:00 # time (D-HH:MM)
#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err # STDERR
#SBATCH --gres=gpu:rtx:1

conda activate home
cd /home/gchrupal/peppa
source ./bin/activate

python evaluate.py  --versions=$1

