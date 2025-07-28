#!/usr/bin/env bash
#SBATCH --job-name=submit_job    # job name
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=1         # number of MPI task per node
#SBATCH --output=submit_job%j.out  # std out
#SBATCH --error=submit_job%j.err   # std err
#SBATCH --exclusive
#SBATCH --account=i20240003g
#SBATCH --time=72:00:00 
#SBATCH --partition=large-x86

VENV_DIR="/path/to/moana-fl-env"

module load Python/3.9.5-GCCcore-10.3.0
echo "PYTHONPATH is ${PYTHONPATH}"
source "${VENV_DIR}/bin/activate"

algorithms_dir="${PWD}/jobs"
workspace="${PWD}/workspaces/workspace"

admin_username="admin@nvidia.com"

job="moana-fl-fedprox-t1"

# submit job
COMMAND="python3 ./submit_job.py --admin_dir=${workspace}/${admin_username} --username=${admin_username} --job=${algorithms_dir}/${job}"
echo "Running: ${COMMAND}"
eval "${COMMAND}"
