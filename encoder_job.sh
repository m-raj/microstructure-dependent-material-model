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
python encoder_main.py --run_id $1  \
 		      		 --data_path "hmc.txt"\
			   	--epochs 2000 \
				--lr 1e-3 \
				--hidden_dim 512 \
				--latent_dim $2 \
				--step 50 \
				--batch_size 1000 \
				--device cpu
