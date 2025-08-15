#!/bin/bash
#SBATCH --job-name=phase_change
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 23:30:00

module reset
module load miniconda
conda activate notebook_env
papermill phase_changes_revised_2.ipynb phase_changes_revised_2_output.ipynb