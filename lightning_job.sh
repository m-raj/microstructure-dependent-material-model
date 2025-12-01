#!/bin/bash

#SBATCH --mem-per-cpu=128G # memory per CPU core
#SBATCH --nodes=1

# Number of GPUs 
# SBATCH --constraint="cascadelake"
#SBATCH --partition gpu
#SBATCH --gres gpu:1

# Wall time: maxctivate base
# Maximum allowed run time
#SBATCH --time=5:30:00

# Send email to user
#SBATCH --mail-user=mraj@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Run the mpi job
args=(
	--run_id $1
	--mode online 
	--device cuda 
#	--pca 
#	--encoder_path encoder_path
	--data_path hmc.txt
	--encoder_latent_dim 15 
	--encoder_hidden_dim 128 
	--encoder_epochs 10 
	--encoder_batch_size 1000 
	--encoder_lr 0.001 
	--material_model m_dependent_c 
	--hidden_dim 128 
	--epochs 10
	--lr 0.001 
	--batch_size 200 
	--niv 1
)

srun python lightning_main.py "${args[@]}"
	

