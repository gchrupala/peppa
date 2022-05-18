#!/bin/bash
#SBATCH -p GPUExtended # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-48:00 # time (D-HH:MM)
#SBATCH -o slurm.%A_%a.out # STDOUT
#SBATCH -e slurm.%j_%a.err # STDERR
#SBATCH --array 1-4
#SBATCH --gres=gpu:rtx:1
conda activate home
cd /home/gchrupal/peppa
source ./bin/activate

python run.py --config_file $1
