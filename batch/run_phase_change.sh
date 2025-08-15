#!/bin/bash
#SBATCH --job-name=phase_change
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 18:00:00

module reset
module load miniconda
conda activate notebook_env
papermill phase_changes_revised.ipynb phase_changes_revised_output.ipynb