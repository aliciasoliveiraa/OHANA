#!/usr/bin/env bash
#SBATCH --job-name=server_job    # job name
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=1         # number of MPI task per node
#SBATCH --output=server_job%j.out  # std out
#SBATCH --error=server_job%j.err   # std err
#SBATCH --exclusive
#SBATCH --account=i20240003g
#SBATCH --time=72:00:00 
#SBATCH --partition=large-x86

VENV_DIR="/path/to/moana-fl-env"

module load Python/3.9.5-GCCcore-10.3.0
echo "PYTHONPATH is ${PYTHONPATH}"
source "${VENV_DIR}/bin/activate"

servername=$HOSTNAME

#CREATE WORKSPACE

sed -i "s/originalhostname/${HOSTNAME}/g" /path/to/moana-fl-fedprox/workspaces/project.yml

cd "${PWD}/workspaces/"

if [ -d "workspace" ]; then
    echo "Removing existing 'workspace' directory..."
    rm -r workspace
fi

nvflare provision -p ./project.yml
cp -r ./workspace/project/prod_00/. ./workspace
cd ..

sed -i "s/${HOSTNAME}/originalhostname/g" /path/to/moana-fl-fedprox/workspaces/project.yml

workspace="${PWD}/workspaces/workspace"

# start server
echo "STARTING SERVER"
"${workspace}/${servername}/startup/start.sh" &
sleep 1

while true
do
  sleep 60
done
