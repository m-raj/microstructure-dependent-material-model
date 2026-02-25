#!/bin/bash

#SBATCH --mem-per-cpu=128G # memory per CPU core
#SBATCH --nodes=1

# Number of GPUs 
# SBATCH --constraint="cascadelake"
#SBATCH --partition gpu
#SBATCH --gres gpu:v100:1

# Wall time: maxctivate base
# Maximum allowed run time
#SBATCH --time=8:30:00
# SBATCH --qos debug

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
	--data_path 1
	--material_model m_evp 
	--hidden_dim 150
	--epochs 2000
	--lr 0.002 
	--batch_size 400 
	--niv 1
	--step 20
	--final_step 2500
	--modes 3
	--out_dim 2
	--u_dim 20
	--z_dim 20
	--loss_type mse
#	--tol 1e-6
#	--solver_lr 0.004
#	--iter_limit 10000
#	--num_workers 0
)

srun python evp_main.py "${args[@]}"
