#!/bin/bash
# Job name:
#SBATCH --job-name=train
#
# Account:
#SBATCH --account=fc_mlsec
# 
# Wall clock limit:
#SBATCH --time=23:59:00 
#  
# Partition:
#SBATCH --partition=savio2
#
# Mail type:
#SBATCH --mail-type=all
#
# Mail user:
#SBATCH --mail-user=alex_vlissidis@berkeley.edu
# 
## Command(s) to run:
module load python/3.6
source activate cs294-131
python train.py
