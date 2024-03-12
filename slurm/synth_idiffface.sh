#!/bin/bash
#SBATCH --account=IscrC_LM-MAD
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=synth_idiffface
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --output=synth_idiffface-%j.out
#SBATCH --error=synth_idiffface-%j.err

eval "$(conda shell.bash hook)"
conda activate sdfr

cd /leonardo_work/IscrC_LM-MAD/sdfr
srun python main.py fit -c experiments/leonardo/train.yml -c experiments/leonardo/synth_idiffface.yml
