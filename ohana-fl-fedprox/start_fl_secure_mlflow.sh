#!/usr/bin/env bash
#SBATCH --job-name=mlflow_job    # job name
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=1         # number of MPI task per node
#SBATCH --output=mlflow_job%j.out  # std out
#SBATCH --error=mlflow_job%j.err   # std err
#SBATCH --exclusive
#SBATCH --account=i20240003g
#SBATCH --time=72:00:00 
#SBATCH --partition=large-x86

VENV_DIR="/path/to/moana-fl-env"

module load Python/3.9.5-GCCcore-10.3.0
echo "PYTHONPATH is ${PYTHONPATH}"
source "${VENV_DIR}/bin/activate"

# start server
echo "STARTING MLFLOW"
mlflow server --host $HOSTNAME --port 5000 &
sleep 1

while true
do
  sleep 60
done
