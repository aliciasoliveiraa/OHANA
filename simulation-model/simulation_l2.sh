#!/usr/bin/env bash
#SBATCH --job-name=simulation_l2    # job name
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=1         # number of MPI task per node
#SBATCH --output=simulation_l2_job%j.out  # std out
#SBATCH --error=simulation_l2_job%j.err   # std err
#SBATCH --exclusive
#SBATCH --account=i20240003g
#SBATCH --time=72:00:00 
#SBATCH --partition=large-x86

module load Python/3.9.5-GCCcore-10.3.0
source /path/to/env/bin/activate
srun pip3 install numpy==1.21.5 nibabel==4.0.2 scipy==1.4.1 imageio==2.6.1 opencv-python
srun python motion_simulation_l2.py
