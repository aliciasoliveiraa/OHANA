#!/usr/bin/env bash
#SBATCH --job-name=processing    # job name
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=1         # number of MPI task per node
#SBATCH --output=processing_job%j.out  # std out
#SBATCH --error=processing_job%j.err   # std err
#SBATCH --exclusive
#SBATCH --account=i20240003g
#SBATCH --time=72:00:00 
#SBATCH --partition=large-x86

module load Python/3.9.5-GCCcore-10.3.0
source path/to/env/bin/activate
srun python preprocessing.py
