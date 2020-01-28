#!/bin/bash
# specify a partition
#SBATCH --partition=dggpu
# Request nodes
#SBATCH --nodes=1
# Request some processor cores
#SBATCH --ntasks=32
# Request GPUs
#SBATCH --gres=gpu:8
# Request memory
#SBATCH --mem=16G
# Maximum runtime of 10 minutes
#SBATCH --time=30:00
# Name of this job
#SBATCH --job-name=sketches
# Output of this job, stderr and stdout are joined by default
# %x=job-name %j=jobid
#SBATCH --output=output/%x_%j.out
# change to the directory where you submitted this script
cd ${SLURM_SUBMIT_DIR}
# your job execution follows:
source activate py36dg
time python transferLearn.py --epoch2=100 --epoch1=5 --load_dataset=True
