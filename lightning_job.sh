#!/bin/bash

#SBATCH --mem-per-cpu=128G # memory per CPU core
#SBATCH --nodes=1

# Number of GPUs 
# SBATCH --constraint="cascadelake"
#SBATCH --partition gpu
#SBATCH --gres gpu:p100:1

# Wall time: maxctivate base
# Maximum allowed run time
#SBATCH --time=16:30:00

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
#	--encoder_path material_model_run_o1
	--data_path hmc.txt
	--encoder_latent_dim 15
	--encoder_hidden_dim 200
	--encoder_epochs 1 
	--encoder_batch_size 1000 
	--encoder_lr 0.001 
	--material_model m_dependent_b_decomposition 
	--hidden_dim 150
	--epochs 2000
	--lr 0.001 
	--batch_size 32 
	--niv 1
	--step 50
#	--freeze_encoder
)

srun python lightning_main.py "${args[@]}"
	

