#!/usr/bin/env bash
#SBATCH --job-name=senior-dist
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/workspace/runs/senior_2020/slurm/%x-%j.out
set -euo pipefail
cd "/workspace"
jupyter nbconvert --to notebook --execute 01_Senior_2020_AlphaFold_CASP13_Reproduction.ipynb --output results/senior_2020/executed.ipynb
