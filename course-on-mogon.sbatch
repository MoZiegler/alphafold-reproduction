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
SIF="${COURSE_DIR}/containers/mogon-cu126.sif"
DEF="${COURSE_DIR}/containers/mogon-cu126.def"
 
# Choose a port on the compute node
PORT=18888
 
# Print connection help to the job output
echo "Job running on: $(hostname)"
echo "Course dir: ${COURSE_DIR}"
echo "Container: ${SIF}"
echo "Connect from laptop with: ssh -N -L ${PORT}:$(hostname):${PORT} <USER>@mogon... (via login node)"
echo "Then open: http://localhost:${PORT}"

if [[ ! -f "${SIF}" ]]; then
  echo "ERROR: Container image not found: ${SIF}"
  echo "Build it on Mogon first with:"
  echo "  module load tools/Apptainer"
  echo "  cd ${COURSE_DIR}"
  echo "  apptainer build ${SIF} ${DEF}"
  exit 2
fi

mkdir -p "${COURSE_DIR}/.jupyter-config" "${COURSE_DIR}/.jupyter-data" "${COURSE_DIR}/.jupyter-runtime"
 
# Start Jupyter inside the container
apptainer exec --nv \
  --cleanenv \
  --bind "${COURSE_DIR}:/workspace" \
  --bind "/lustre/project/ki-deepprakt/:/data" \
  --env JUPYTER_CONFIG_DIR=/workspace/.jupyter-config \
  --env JUPYTER_DATA_DIR=/workspace/.jupyter-data \
  --env JUPYTER_RUNTIME_DIR=/workspace/.jupyter-runtime \
  "$SIF" \
  bash -lc "python - <<'PY'
import torch
print('python smoke ok')
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
PY
jupyter lab --ip=0.0.0.0 --port=${PORT} --no-browser --ServerApp.token='' --ServerApp.password='' --notebook-dir=/workspace"
