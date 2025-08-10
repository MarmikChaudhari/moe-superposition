#!/bin/bash
#SBATCH -p day
#SBATCH -c 4
#SBATCH -t 3:00:00

module reset
module load miniconda
conda activate notebook_env
papermill phase_changes_revised.ipynb phase_changes_revised_output.ipynb