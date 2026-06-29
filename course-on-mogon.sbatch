#!/bin/bash
#SBATCH -A ki-deepprakt
#SBATCH -p a100ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256GB
#SBATCH --time=24:00:00
#SBATCH -J jupyter-course
#SBATCH -o jupyter-%j.out

export SLURM_CLUSTERS=mogonki
export SBATCH_CLUSTERS=mogonki
 
module load tools/Apptainer
 
# Recommended for big images
export APPTAINER_TMPDIR=/localscratch/${SLURM_JOB_ID}/apptainer_tmp
mkdir -p "$APPTAINER_TMPDIR"
 
COURSE_DIR="$HOME/alphafold-reproduction"
SIF="/lustre/project/ki-deepprakt/course-cu126.sif"
 
# Choose a port on the compute node
PORT=18888
 
# Print connection help to the job output
echo "Job running on: $(hostname)"
echo "Connect from laptop with: ssh -N -L ${PORT}:$(hostname):${PORT} <USER>@mogon... (via login node)"
echo "Then open: http://localhost:${PORT}"
 
# Start Jupyter inside the container
apptainer exec --nv \
  --bind "${COURSE_DIR}:/workspace" \
  --bind "/lustre/project/ki-deepprakt/:/data" \
  "$SIF" \
  bash -lc "jupyter lab --ip=0.0.0.0 --port=${PORT} --no-browser --ServerApp.token='' --ServerApp.password='' --notebook-dir=/workspace"
