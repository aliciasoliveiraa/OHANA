#!/usr/bin/env bash
#SBATCH --job-name=clients_job    # job name
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=1         # number of MPI task per node
#SBATCH --output=clients_job_%A_%a.out  # std out
#SBATCH --error=clients_job_%A_%a.err   # std err
#SBATCH --exclusive 
#SBATCH --account=i20240003g
#SBATCH --time=72:00:00 
#SBATCH --partition=large-x86
#SBATCH --array=1-2

VENV_DIR="/path/to/moana-fl-env"

module load Python/3.9.5-GCCcore-10.3.0
echo "PYTHONPATH is ${PYTHONPATH}"
source "${VENV_DIR}/bin/activate"

workspace="${PWD}/workspaces/workspace"
site_pre="site-"
id=$SLURM_ARRAY_TASK_ID

# start clients
echo "STARTING ${site_pre}${id} CLIENTS"
hostname
"${workspace}/${site_pre}${id}/startup/start.sh"

while true
do
  sleep 60
done
