#!/bin/bash
#SBATCH -A ki-deepprakt
#SBATCH -p a100ai
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH -J build-af-repro-cu126
#SBATCH -o build-af-repro-cu126-%j.out

set -euo pipefail

export SLURM_CLUSTERS=mogonki
export SBATCH_CLUSTERS=mogonki

module load tools/Apptainer

COURSE_DIR="${HOME}/alphafold-reproduction"
DEF="${COURSE_DIR}/containers/mogon-cu126.def"
SIF="${COURSE_DIR}/containers/mogon-cu126.sif"

export APPTAINER_TMPDIR="/localscratch/${SLURM_JOB_ID}/apptainer_tmp"
export APPTAINER_CACHEDIR="/localscratch/${SLURM_JOB_ID}/apptainer_cache"
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

cd "${COURSE_DIR}"

echo "Building ${SIF}"
echo "Definition: ${DEF}"
echo "Node: $(hostname)"

apptainer build --fakeroot "${SIF}" "${DEF}"

echo "Testing ${SIF}"
apptainer test --nv "${SIF}"

echo "Done: ${SIF}"
