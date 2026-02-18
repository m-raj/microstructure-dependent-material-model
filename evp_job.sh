#!/bin/bash

#SBATCH --mem-per-cpu=128G # memory per CPU core
#SBATCH --nodes=1

# Number of GPUs 
# SBATCH --constraint="cascadelake"
#SBATCH --partition gpu
#SBATCH --gres gpu:p100:1

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
	--data_path evp_data.txt
	--material_model m_evp_adjoint 
	--hidden_dim 150
	--epochs 2000
	--lr 0.001 
	--batch_size 400 
	--niv 1
	--step 20
	--final_step 5000
	--modes 3
	--out_dim 1
	--u_dim 50
	--z_dim 50
	--loss_type adjoint
	--tol 1e-4
	--solver_lr 0.0004
	--iter_limit 100
#	--method newton
)

srun python evp_main.py "${args[@]}"
