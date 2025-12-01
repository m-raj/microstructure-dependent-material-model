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
python pca.py --run_id $1  \
                --data_files "data/2024-10-13_PC1D_process10_data.pkl,
				data/2024-10-13_PC1D_process11_data.pkl,
				data/2024-10-13_PC1D_process12_data.pkl,
				data/2024-10-13_PC1D_process13_data.pkl,
				data/2024-10-13_PC1D_process14_data.pkl,
				data/2024-10-13_PC1D_process15_data.pkl,
				data/2024-10-13_PC1D_process16_data.pkl,
				data/2024-10-13_PC1D_process17_data.pkl,
				data/2024-10-13_PC1D_process18_data.pkl,
				data/2024-10-13_PC1D_process19_data.pkl" \
		--features 15
