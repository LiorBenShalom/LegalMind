#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=5-12:00:00
#SBATCH --job-name=tag_citation
#SBATCH --output=/home/liorkob/M.Sc/thesis/pre process/tag_citation.log
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=48G

module load anaconda
source activate new_env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

papermill /home/liorkob/M.Sc/thesis/pre process/tag_citation copy.ipynb \
          /home/liorkob/M.Sc/thesis/pre process/tag_citation copy_out.ipynb
