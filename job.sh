#!/bin/bash

#SBATCH --mem-per-cpu=128G # memory per CPU core
#SBATCH --nodes=1

# Number of GPUs
# SBATCH --constraint="cascadelake"

# Wall time: maximum allowed run time
#SBATCH --time=2:00:00
# SBATCH --qos=debug

# Send email to user
#SBATCH --mail-user=mraj@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Run the mpi job
python microstructure_encoder_main.py --run_id $1  \
                            	--data_path data/2024-10-13_PC1D_process10_data.pkl \
			   	--epochs 1000 \
				--lr 1e-3 \
				--hidden_dim 128 \
				--latent_dim 10 \
				--step 50 \
				--n_samples 10 \
				--device $2
