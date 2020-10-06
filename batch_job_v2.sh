#!/bin/bash
#SBATCH --account=project_2001281
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem-per-gpu=32G 
#SBATCH --mail-user=nicola.dainese@aalto.fi
#SBATCH --mail-type=FAIL,REQUEUE,TIME_LIMIT_80

# email research-support@csc.fi for help
module load pytorch/nvidia-20.03-py3
singularity_wrapper exec python monobeast_v2.py $*
