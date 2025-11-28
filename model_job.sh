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
python model_main.py --run_id $1  \
                --data_path "hmc.txt" \
				--pca 0 \
				--epochs 5000 \
				--lr 1e-3 \
				--hidden_dim 600 \
				--encoder_hidden_dim 200 \
				--encoder_latent_dim 15 \
				--step 50 \
				--encoder_path encoder_run_h1 \
				--material_model m_dependent_d \
				--device cuda \
				--batch_size 200 \
				--hrs 0.1\
				--niv 1\
				--mode online
