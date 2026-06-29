from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


COMMON_SETUP = r'''
from __future__ import annotations

import json
import math
import os
import random
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path.cwd()
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_ROOT = PROJECT_ROOT / "models"
RESULTS_ROOT = PROJECT_ROOT / "results"
RUNS_ROOT = PROJECT_ROOT / "runs"

for path in [DATA_ROOT, MODEL_ROOT, RESULTS_ROOT, RUNS_ROOT]:
    path.mkdir(parents=True, exist_ok=True)

def latest_environment_report() -> dict:
    report_dir = DATA_ROOT / "environment_reports"
    reports = sorted(report_dir.glob("environment_report_*_utc.json"))
    if not reports:
        return {}
    return json.loads(reports[-1].read_text(encoding="utf-8"))

ENV_REPORT = latest_environment_report()

def seed_everything(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path

def run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 30, dry_run: bool = True):
    print("$", " ".join(shlex.quote(str(x)) for x in cmd))
    if dry_run:
        print("DRY_RUN=True, command was not executed.")
        return None
    completed = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    print(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}")
    return completed

def gpu_summary() -> str:
    devices = ENV_REPORT.get("torch", {}).get("devices", [])
    if devices:
        first = devices[0]
        return f"{first.get('name')} / {first.get('total_memory_gb')} GB / cc {first.get('major')}.{first.get('minor')}"
    return "No saved GPU report found."

seed_everything(7)
print(f"Project root: {PROJECT_ROOT}")
print(f"Device: {device()}")
print(f"Saved cluster GPU: {gpu_summary()}")
'''


SCORING_AND_CLUSTER = r'''
## Cluster execution template

The notebooks are deliberately importable and runnable from `papermill`, `jupyter nbconvert --execute`, or ordinary notebook execution. For long runs on the cluster, the code writes a small SLURM script that executes this notebook with parameters rather than keeping GPU time tied to the browser session.
'''


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def senior_notebook() -> dict:
    return notebook([
        md(r'''
# 01 - Senior et al. 2020: AlphaFold CASP13 Reproduction

This notebook is a **from-scratch PyTorch reproduction** of the first AlphaFold paper, not a wrapper around DeepMind's CASP13 code or weights.

The Senior et al. system can be decomposed into four reproducible ideas:

1. Build leakage-controlled sequence/template features for CASP13 targets.
2. Train a residual network that predicts inter-residue distance distributions, torsions, and auxiliary structure features.
3. Convert predicted distance probabilities into a differentiable potential of mean force.
4. Optimize 3D coordinates under that potential and score them against CASP13 structures.

The first faithful target is "same benchmark, same information boundary." The improvement target is "beat their benchmark with clearly labeled enhanced experiments."
'''),
        md(r'''
## Data and model layout

All input data and trained checkpoints are ours:

- `data/senior_2020/raw`: CASP13 FASTA, ground-truth structures, dated PDB/CATH snapshots, and MSA database snapshots.
- `data/senior_2020/features`: tensorized MSA/template/contact labels.
- `models/senior_2020/checkpoints`: our PyTorch checkpoints.
- `results/senior_2020`: distograms, optimized PDBs, and score tables.

No official AlphaFold API, Docker runner, model parameters, or inference script is used.
'''),
        code(COMMON_SETUP),
        code(r'''
PAPER = "senior_2020"
DATA_DIR = DATA_ROOT / PAPER
MODEL_DIR = MODEL_ROOT / PAPER
RESULT_DIR = RESULTS_ROOT / PAPER
RUN_DIR = RUNS_ROOT / PAPER

paths = {
    "raw": DATA_DIR / "raw",
    "features": DATA_DIR / "features",
    "labels": DATA_DIR / "labels",
    "checkpoints": MODEL_DIR / "checkpoints",
    "distograms": RESULT_DIR / "distograms",
    "structures": RESULT_DIR / "structures",
    "metrics": RESULT_DIR / "metrics",
    "slurm": RUN_DIR / "slurm",
}
for p in paths.values():
    p.mkdir(parents=True, exist_ok=True)
print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
'''),
        md(r'''
## Step 1 - Benchmark contract

The contract below is the guardrail against accidental leakage. A faithful Senior reproduction should not train on structures that post-date CASP13 and should not use sequence/template databases newer than the paper-era versions. Later, enhanced experiments may use modern databases, but those rows must be marked `faithful=False`.

Scientifically, this step defines the counterfactual question: "What could the method have known at CASP13 time?" Protein-structure predictors can silently improve just by seeing homologous future structures or larger modern sequence databases, so the benchmark boundary is part of the experiment, not bookkeeping.

Computationally, we serialize the boundary as JSON so every later feature file, checkpoint, and score table can point back to the same data contract. Mathematically, it fixes the training and evaluation distributions: training samples are drawn from structures and alignments available before the cutoff, while CASP13 targets remain held out. Without that separation, a high TM-score would not estimate generalization.
'''),
        code(r'''
contract = {
    "paper": "Senior et al. 2020",
    "benchmark": "CASP13 free modelling domains",
    "template_cutoff": "2018-03-15",
    "cath_cutoff": "2018-03-16",
    "uniclust30_version": "2017-10",
    "nr_snapshot": "2017-12-15",
    "implementation": "own_pytorch",
    "uses_official_weights_or_api": False,
}
write_text(DATA_DIR / "benchmark_contract.json", json.dumps(contract, indent=2))
print(json.dumps(contract, indent=2))
'''),
        md(r'''
## Step 2 - Feature tensors

Senior-style distance prediction is naturally pairwise. We represent each protein as:

- `msa_profile`: `[L, A]`, amino-acid frequencies from the MSA.
- `msa_covariance`: `[L, L, C]`, cheap co-evolution/covariance summaries.
- `template_pair`: `[L, L, T]`, template-derived distances/masks if allowed by cutoff.
- `dist_bin`: `[L, L]`, supervised C-beta/C-alpha distance bin label.

The dataset class below is intentionally file-format simple. Feature builders can write `.npz` files from HHblits/JackHMMER/template tools later, while the model and training code stay stable.

Scientifically, the MSA summarizes evolutionary pressure: residues that mutate together often contact or constrain each other in 3D. Templates encode the separate hypothesis that homologous folds provide partial geometry when they are legitimately available before the cutoff.

Computationally, we turn variable biological evidence into dense tensors. The central tensor is `[L, L, C]`, where each entry describes a residue pair. Mathematically, the supervised label is a binned distance random variable `Y_ij`; the model will learn `p(Y_ij = b | MSA, templates)` for every residue pair `(i, j)`.
'''),
        code(r'''
class SeniorFeatureDataset(Dataset):
    def __init__(self, feature_dir: Path, synthetic_if_empty: bool = True, n_synthetic: int = 8, length: int = 96, bins: int = 32):
        self.files = sorted(feature_dir.glob("*.npz"))
        self.synthetic_if_empty = synthetic_if_empty
        self.n_synthetic = n_synthetic
        self.length = length
        self.bins = bins

    def __len__(self):
        return len(self.files) if self.files else (self.n_synthetic if self.synthetic_if_empty else 0)

    def __getitem__(self, idx):
        if self.files:
            arr = np.load(self.files[idx])
            return {k: torch.as_tensor(arr[k]) for k in arr.files}

        L, B = self.length, self.bins
        msa_profile = F.one_hot(torch.randint(0, 21, (L,)), 21).float()
        msa_covariance = torch.randn(L, L, 16) * 0.1
        template_pair = torch.zeros(L, L, 8)
        pair = torch.cat([
            msa_profile[:, None, :].expand(L, L, 21),
            msa_profile[None, :, :].expand(L, L, 21),
            msa_covariance,
            template_pair,
        ], dim=-1)
        dist_bin = torch.randint(0, B, (L, L))
        dist_bin = torch.triu(dist_bin, diagonal=1) + torch.triu(dist_bin, diagonal=1).T
        return {"pair": pair, "dist_bin": dist_bin}

dataset = SeniorFeatureDataset(paths["features"])
batch = dataset[0]
print({k: tuple(v.shape) for k, v in batch.items()})
'''),
        md(r'''
## Step 3 - Our Senior-style distogram network

The original system used deep residual networks over pair features. We keep that inductive bias: 2D convolutions operate on the residue-pair matrix, preserving local patterns in sequence separation and pairwise geometry. The network predicts distance bins for every residue pair.

Scientifically, a distogram is richer than a contact map because it preserves approximate geometry instead of only saying "near" or "far." That matters because many folds can satisfy the same binary contacts, but far fewer satisfy a full collection of distance distributions.

Computationally, the model is a 2D residual CNN over the residue-pair image. Residual connections let us stack many transformations without destroying gradient flow. Mathematically, the final head produces logits `z_ijb`, and `softmax_b(z_ijb)` is our estimate of the distance-bin distribution for pair `(i, j)`.
'''),
        code(r'''
class Residual2DBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        pad = dilation
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=pad, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class SeniorDistogramNet(nn.Module):
    def __init__(self, pair_dim: int = 66, hidden: int = 128, blocks: int = 16, bins: int = 32):
        super().__init__()
        self.stem = nn.Conv2d(pair_dim, hidden, 1)
        self.blocks = nn.ModuleList([Residual2DBlock(hidden, dilation=1 + (i % 4)) for i in range(blocks)])
        self.dist_head = nn.Conv2d(hidden, bins, 1)

    def forward(self, pair):
        # pair: [B, L, L, C]
        x = pair.permute(0, 3, 1, 2).contiguous()
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        logits = self.dist_head(x).permute(0, 2, 3, 1).contiguous()
        logits = 0.5 * (logits + logits.transpose(1, 2))
        return {"distogram_logits": logits}

model = SeniorDistogramNet().to(device())
with torch.no_grad():
    out = model(batch["pair"][None].to(device()))
print(tuple(out["distogram_logits"].shape))
'''),
        md(r'''
## Step 4 - Training objective

The distogram target is a categorical distance distribution. We train with cross-entropy over residue pairs, excluding the diagonal and optionally downweighting pairs close in sequence if we want to emphasize long-range structure. This is the learning step that turns MSA/template evidence into geometric constraints.

Scientifically, the network is asked to learn statistical regularities between evolutionary features and physical distances: co-evolving residues, conserved motifs, and template hints should increase probability mass on compatible distance bins.

Computationally, each protein contributes roughly `O(L^2)` residue-pair examples, so masking and batching matter on GPU. Mathematically, the loss is the negative log-likelihood `-sum_ij log p_theta(y_ij | x)` over valid residue pairs. Minimizing this trains the model to assign calibrated probability mass to the observed structure-derived bins.
'''),
        code(r'''
def senior_loss(outputs, target_bins):
    logits = outputs["distogram_logits"]
    B, L, _, bins = logits.shape
    mask = torch.triu(torch.ones(L, L, device=logits.device, dtype=torch.bool), diagonal=1)
    loss = F.cross_entropy(logits[:, mask, :].reshape(-1, bins), target_bins[:, mask].reshape(-1))
    return loss

loader = DataLoader(dataset, batch_size=1, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
model.train()
for step, item in enumerate(loader):
    item = {k: v.to(device()) for k, v in item.items()}
    optimizer.zero_grad(set_to_none=True)
    loss = senior_loss(model(item["pair"]), item["dist_bin"].long())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    print(f"step={step} loss={float(loss.detach().cpu()):.4f}")
    if step >= 2:
        break
'''),
        md(r'''
## Step 5 - Potential of mean force and coordinate optimization

The network predicts probabilities, not coordinates. We reproduce the paper's core move: transform log-probabilities into a coordinate potential and optimize a chain. This code is deliberately PyTorch-native so later improvements can backpropagate through parts of the pipeline if useful.

Scientifically, this converts learned geometric preferences into an actual fold. The model says which distances are plausible; coordinate optimization asks for a 3D chain whose pairwise distances satisfy those preferences while remaining protein-like.

Computationally, we optimize coordinates directly with automatic differentiation. Mathematically, the potential is an energy `E(x) = -sum_ij log p_ij(bin(||x_i - x_j||))` with a reference correction and chain regularizers. Gradient descent changes coordinates in the direction that lowers this learned energy landscape.
'''),
        code(r'''
def distogram_energy(coords, bin_edges, pair_log_probs, reference_log_probs):
    distances = torch.cdist(coords, coords).clamp_min(1e-6)
    bin_idx = torch.bucketize(distances, bin_edges[1:-1])
    observed = pair_log_probs.gather(-1, bin_idx[..., None]).squeeze(-1)
    reference = reference_log_probs[bin_idx]
    mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=2)
    return -(observed - reference)[mask].mean()

def optimize_trace(pair_logits, steps: int = 300, lr: float = 0.05):
    L, _, bins = pair_logits.shape
    dev = pair_logits.device
    bin_edges = torch.linspace(2.0, 22.0, bins + 1, device=dev)
    pair_log_probs = pair_logits.log_softmax(dim=-1)
    reference = torch.full((bins,), -math.log(bins), device=dev)
    coords = torch.randn(L, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([coords], lr=lr)
    history = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        energy = distogram_energy(coords, bin_edges, pair_log_probs, reference)
        ca = (coords[1:] - coords[:-1]).norm(dim=-1)
        chain_loss = ((ca - 3.8) ** 2).mean()
        loss = energy + 0.05 * chain_loss
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.detach().cpu()), "energy": float(energy.detach().cpu())})
    return coords.detach(), history

model.eval()
with torch.no_grad():
    logits = model(batch["pair"][None].to(device()))["distogram_logits"][0]
coords, hist = optimize_trace(logits, steps=20)
print(hist[-1], tuple(coords.shape))
'''),
        md(r'''
## Step 6 - Scoring and improvement loop

Final scoring should use external structural metrics such as TM-score/US-align and CASP-style GDT. The notebook writes score schemas and experiment registries rather than hard-coding a single binary path.

Scientifically, score choice determines what "reproduction" means. TM-score and GDT measure global fold accuracy, while later additions such as lDDT or violation counts can expose local geometry failures that a global metric may hide.

Computationally, we separate metric schemas from metric binaries because clusters differ in installed tools. Mathematically, each experiment row is a paired comparison between predicted coordinates `X_hat` and reference coordinates `X`, after alignment or distance-based matching. The registry also prevents enhanced experiments from being mistaken for faithful reproduction.
'''),
        code(r'''
score_schema = {
    "target_id": "T0986s2",
    "prediction_path": "results/senior_2020/structures/T0986s2/model_0.pdb",
    "truth_path": "data/senior_2020/raw/casp13_targets/T0986s2.pdb",
    "tm_score": None,
    "gdt_ts": None,
    "faithful": True,
    "implementation": "own_pytorch",
}
experiments = [
    {"name": "senior_faithful_resnet_distogram", "faithful": True, "change": "paper-era data, own residual distogram model"},
    {"name": "senior_modern_msa", "faithful": False, "change": "new sequence databases"},
    {"name": "senior_af2_prior_potential", "faithful": False, "change": "AF2-like prior in coordinate optimizer"},
]
write_text(paths["metrics"] / "score_schema.json", json.dumps(score_schema, indent=2))
write_text(RUN_DIR / "experiment_registry.json", json.dumps(experiments, indent=2))
print(json.dumps(experiments, indent=2))
'''),
        md(SCORING_AND_CLUSTER),
        code(r'''
slurm = paths["slurm"] / "train_senior_distogram.sbatch"
slurm.write_text(f"""#!/usr/bin/env bash
#SBATCH --job-name=senior-dist
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output={paths['slurm']}/%x-%j.out
set -euo pipefail
cd "{PROJECT_ROOT}"
jupyter nbconvert --to notebook --execute 01_Senior_2020_AlphaFold_CASP13_Reproduction.ipynb --output results/senior_2020/executed.ipynb
""", encoding="utf-8")
print(slurm.read_text(encoding="utf-8"))
'''),
        md(r'''
## Step 7 - Example-protein comparison against structures and Senior predictions

This section turns a training run into a scientific comparison on concrete proteins. The next cells load three structures for each target: the experimental reference, our model prediction, and the Senior et al. paper prediction. If you have the paper predictions as PDB files, place them under `data/senior_2020/senior_paper_predictions/` or `results/senior_2020/senior_paper_predictions/` with filenames that contain the target id.

Scientifically, this is the first honest visual and numerical test of whether our reproduction is learning the same geometric signal as Senior et al. A loss curve can improve while structures remain wrong; a structure comparison exposes fold-level failures, long-range-contact errors, and coordinate artifacts.

Computationally, we parse C-alpha traces from PDB files, align predicted traces to the experimental trace with the Kabsch algorithm, and compute simple metrics. Mathematically, Kabsch solves `argmin_R,t ||R X_hat + t - X||_F` over rotations and translations. We then compare aligned coordinates, distance matrices `D_ij`, and contact maps so both coordinate-level and topology-level errors are visible.
'''),
        code(r'''
import matplotlib.pyplot as plt
import pandas as pd

COMPARISON_MANIFEST = DATA_DIR / "comparison_targets.json"
PLOT_DIR = RESULT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_PREDICTION_DIRS = [
    DATA_DIR / "senior_paper_predictions",
    RESULT_DIR / "senior_paper_predictions",
]
TRUTH_DIRS = [
    DATA_DIR / "raw" / "casp13_targets",
    DATA_DIR / "casp13_targets",
    DATA_DIR / "truth",
]
OUR_PREDICTION_DIRS = [
    RESULT_DIR / "structures",
    RESULT_DIR / "predictions",
]

def candidate_structure_files(target_id: str, roots: list[Path]) -> list[Path]:
    suffixes = ["*.pdb", "*.ent", "*.cif", "*.mmcif"]
    matches = []
    for root in roots:
        if not root.exists():
            continue
        for suffix in suffixes:
            matches.extend(root.rglob(f"*{target_id}*{suffix[1:]}"))
            target_dir = root / target_id
            if target_dir.exists():
                matches.extend(target_dir.glob(suffix))
    return sorted(set(p for p in matches if p.is_file()))

if not COMPARISON_MANIFEST.exists():
    example = [
        {
            "target_id": "T0986s2",
            "truth_pdb": "data/senior_2020/raw/casp13_targets/T0986s2.pdb",
            "ours_pdb": "results/senior_2020/structures/T0986s2/model_0.pdb",
            "senior_paper_pdb": "data/senior_2020/senior_paper_predictions/T0986s2.pdb",
        }
    ]
    write_text(COMPARISON_MANIFEST, json.dumps(example, indent=2))
    print(f"Wrote example comparison manifest: {COMPARISON_MANIFEST}")

comparison_rows = json.loads(COMPARISON_MANIFEST.read_text(encoding="utf-8"))
print(json.dumps(comparison_rows, indent=2))
'''),
        code(r'''
def parse_ca_pdb(path: Path):
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom = line[12:16].strip()
        if atom != "CA":
            continue
        altloc = line[16].strip()
        if altloc not in {"", "A"}:
            continue
        try:
            chain = line[21].strip() or "_"
            resseq = int(line[22:26])
            icode = line[26].strip()
            coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float64)
        except ValueError:
            continue
        rows.append({"key": (chain, resseq, icode), "chain": chain, "resseq": resseq, "coord": coord})
    return rows

def paired_ca_coords(reference_path: Path, prediction_path: Path):
    ref_rows = parse_ca_pdb(reference_path)
    pred_rows = parse_ca_pdb(prediction_path)
    ref_by_key = {r["key"]: r["coord"] for r in ref_rows}
    pred_by_key = {r["key"]: r["coord"] for r in pred_rows}
    common = [r["key"] for r in ref_rows if r["key"] in pred_by_key]
    if len(common) >= 3:
        return np.stack([ref_by_key[k] for k in common]), np.stack([pred_by_key[k] for k in common]), common
    n = min(len(ref_rows), len(pred_rows))
    if n < 3:
        raise ValueError(f"Need at least 3 paired C-alpha atoms: {reference_path}, {prediction_path}")
    return np.stack([r["coord"] for r in ref_rows[:n]]), np.stack([r["coord"] for r in pred_rows[:n]]), [r["key"] for r in ref_rows[:n]]

def kabsch_align(mobile: np.ndarray, target: np.ndarray):
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    X = mobile - mobile_center
    Y = target - target_center
    C = X.T @ Y
    V, _, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1.0, 1.0, d]) @ Wt
    return X @ R + target_center

def distance_matrix(coords: np.ndarray):
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))

def structure_metrics(reference: np.ndarray, prediction: np.ndarray):
    aligned = kabsch_align(prediction, reference)
    per_residue_error = np.linalg.norm(aligned - reference, axis=-1)
    rmsd = float(np.sqrt(np.mean(per_residue_error ** 2)))
    L = len(reference)
    d0 = max(0.5, 1.24 * max(L - 15, 1) ** (1 / 3) - 1.8)
    tm_like = float(np.mean(1.0 / (1.0 + (per_residue_error / d0) ** 2)))
    gdt_ts_like = float(np.mean([np.mean(per_residue_error <= t) for t in [1.0, 2.0, 4.0, 8.0]]))
    ref_d = distance_matrix(reference)
    pred_d = distance_matrix(aligned)
    sep_mask = np.triu(np.ones((L, L), dtype=bool), k=6)
    dist_mae = float(np.mean(np.abs(ref_d[sep_mask] - pred_d[sep_mask]))) if sep_mask.any() else np.nan
    ref_contacts = (ref_d < 8.0) & sep_mask
    pred_contacts = (pred_d < 8.0) & sep_mask
    contact_precision = float((ref_contacts & pred_contacts).sum() / max(pred_contacts.sum(), 1))
    contact_recall = float((ref_contacts & pred_contacts).sum() / max(ref_contacts.sum(), 1))
    return {
        "aligned": aligned,
        "per_residue_error": per_residue_error,
        "rmsd_ca": rmsd,
        "tm_like": tm_like,
        "gdt_ts_like": gdt_ts_like,
        "distance_mae_long_range": dist_mae,
        "contact_precision_8A": contact_precision,
        "contact_recall_8A": contact_recall,
        "reference_distance": ref_d,
        "prediction_distance": pred_d,
    }

def resolve_comparison_paths(row: dict):
    target_id = row["target_id"]
    truth = Path(row.get("truth_pdb", ""))
    ours = Path(row.get("ours_pdb", ""))
    senior = Path(row.get("senior_paper_pdb", ""))
    if not truth.exists():
        candidates = candidate_structure_files(target_id, TRUTH_DIRS)
        truth = candidates[0] if candidates else truth
    if not ours.exists():
        candidates = candidate_structure_files(target_id, OUR_PREDICTION_DIRS)
        ours = candidates[0] if candidates else ours
    if not senior.exists():
        candidates = candidate_structure_files(target_id, PAPER_PREDICTION_DIRS)
        senior = candidates[0] if candidates else senior
    return truth, ours, senior
'''),
        code(r'''
comparison_records = []
comparison_payloads = {}

for row in comparison_rows:
    target_id = row["target_id"]
    truth_path, ours_path, senior_path = resolve_comparison_paths(row)
    print(f"\nTarget {target_id}")
    print("  truth :", truth_path, truth_path.exists())
    print("  ours  :", ours_path, ours_path.exists())
    print("  Senior:", senior_path, senior_path.exists())
    if not truth_path.exists() or not ours_path.exists():
        print("  Skipping: need at least truth_pdb and ours_pdb.")
        continue

    reference, ours_coords, _ = paired_ca_coords(truth_path, ours_path)
    ours_metrics = structure_metrics(reference, ours_coords)
    comparison_records.append({
        "target_id": target_id,
        "method": "ours",
        "n_residues": len(reference),
        **{k: v for k, v in ours_metrics.items() if isinstance(v, float)},
    })
    comparison_payloads[(target_id, "ours")] = {"reference": reference, "prediction": ours_metrics["aligned"], **ours_metrics}

    if senior_path.exists():
        reference_s, senior_coords, _ = paired_ca_coords(truth_path, senior_path)
        senior_metrics = structure_metrics(reference_s, senior_coords)
        comparison_records.append({
            "target_id": target_id,
            "method": "Senior et al.",
            "n_residues": len(reference_s),
            **{k: v for k, v in senior_metrics.items() if isinstance(v, float)},
        })
        comparison_payloads[(target_id, "Senior et al.")] = {"reference": reference_s, "prediction": senior_metrics["aligned"], **senior_metrics}

metrics_df = pd.DataFrame(comparison_records)
metrics_path = paths["metrics"] / "example_structure_comparison.csv"
if not metrics_df.empty:
    metrics_df.to_csv(metrics_path, index=False)
    display(metrics_df.sort_values(["target_id", "method"]))
    print(f"Saved metrics to {metrics_path}")
else:
    print("No complete comparisons found yet. Fill comparison_targets.json or place PDB files in the documented folders.")
'''),
        code(r'''
def plot_ca_trace(ax, coords: np.ndarray, title: str, color: str):
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=1.6)
    ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color=color, s=30, marker="o")
    ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], color=color, s=30, marker="x")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

def plot_target_comparison(target_id: str):
    methods = [m for (tid, m) in comparison_payloads if tid == target_id]
    if not methods:
        print(f"No payloads for {target_id}")
        return

    fig = plt.figure(figsize=(5 * (len(methods) + 1), 4.5))
    reference = comparison_payloads[(target_id, methods[0])]["reference"]
    ax = fig.add_subplot(1, len(methods) + 1, 1, projection="3d")
    plot_ca_trace(ax, reference, f"{target_id} experimental", "black")
    for idx, method in enumerate(methods, start=2):
        payload = comparison_payloads[(target_id, method)]
        ax = fig.add_subplot(1, len(methods) + 1, idx, projection="3d")
        plot_ca_trace(ax, payload["reference"], "reference", "lightgray")
        plot_ca_trace(ax, payload["prediction"], method, "tab:blue" if method == "ours" else "tab:orange")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{target_id}_ca_trace_comparison.png", dpi=200, bbox_inches="tight")
    plt.show()

    fig, axes = plt.subplots(len(methods), 3, figsize=(13, 4 * len(methods)), squeeze=False)
    for row_idx, method in enumerate(methods):
        payload = comparison_payloads[(target_id, method)]
        ref_d = payload["reference_distance"]
        pred_d = payload["prediction_distance"]
        err_d = np.abs(ref_d - pred_d)
        for ax, mat, title in [
            (axes[row_idx, 0], ref_d, "experimental distance map"),
            (axes[row_idx, 1], pred_d, f"{method} distance map"),
            (axes[row_idx, 2], err_d, f"{method} |distance error|"),
        ]:
            im = ax.imshow(mat, cmap="viridis" if "error" not in title else "magma")
            ax.set_title(f"{target_id}: {title}")
            ax.set_xlabel("residue")
            ax.set_ylabel("residue")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{target_id}_distance_maps.png", dpi=200, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 4))
    for method in methods:
        payload = comparison_payloads[(target_id, method)]
        ax.plot(payload["per_residue_error"], label=method)
    ax.set_title(f"{target_id}: per-residue C-alpha error after Kabsch alignment")
    ax.set_xlabel("paired residue index")
    ax.set_ylabel("error (Angstrom)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(PLOT_DIR / f"{target_id}_per_residue_error.png", dpi=200, bbox_inches="tight")
    plt.show()

if not metrics_df.empty:
    for target_id in metrics_df["target_id"].unique():
        plot_target_comparison(target_id)
'''),
        code(r'''
if not metrics_df.empty:
    score_cols = ["rmsd_ca", "tm_like", "gdt_ts_like", "distance_mae_long_range", "contact_precision_8A", "contact_recall_8A"]
    available_score_cols = [c for c in score_cols if c in metrics_df.columns]
    for metric in available_score_cols:
        pivot = metrics_df.pivot(index="target_id", columns="method", values=metric)
        ax = pivot.plot(kind="bar", figsize=(10, 4), rot=45)
        ax.set_title(f"Our Senior reproduction vs Senior et al.: {metric}")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"method_comparison_{metric}.png", dpi=200, bbox_inches="tight")
        plt.show()

    if {"ours", "Senior et al."}.issubset(set(metrics_df["method"])):
        paired = metrics_df.pivot(index="target_id", columns="method", values="tm_like").dropna()
        if not paired.empty:
            paired["ours_minus_senior"] = paired["ours"] - paired["Senior et al."]
            colors = np.where(paired["ours_minus_senior"] >= 0, "tab:green", "tab:red")
            ax = paired["ours_minus_senior"].plot(kind="bar", figsize=(10, 3), color=colors)
            ax.axhline(0, color="black", linewidth=1)
            ax.set_title("TM-like delta: ours minus Senior et al.")
            ax.set_ylabel("delta")
            ax.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "ours_minus_senior_tm_like_delta.png", dpi=200, bbox_inches="tight")
            plt.show()
            display(paired)
'''),
    ])


def jumper_notebook() -> dict:
    return notebook([
        md(r'''
# 02 - Jumper et al. 2021: AlphaFold2 Reproduction

This notebook is a **from-scratch PyTorch reproduction** of AlphaFold2's main ideas. It does not use the official AlphaFold repository, runners, model parameters, or APIs.

We reproduce the reasoning rather than the exact proprietary training recipe:

1. MSA and pair representations are processed jointly.
2. Evoformer-like blocks exchange information between MSA rows and residue pairs.
3. A structure module predicts coordinates and confidence heads.
4. Recycling feeds predictions back through the network.
5. CASP14-style scoring compares predicted structures to held-out targets.
'''),
        md(r'''
## Layout and cluster assumptions

Your hardware report says an A100 80GB is available. That is enough for realistic inference and medium-size training experiments, but full AF2-scale training remains extremely expensive. The notebook is written to scale: small synthetic smoke batches now, real feature tensors and distributed training later.
'''),
        code(COMMON_SETUP),
        code(r'''
PAPER = "jumper_2021"
DATA_DIR = DATA_ROOT / PAPER
MODEL_DIR = MODEL_ROOT / PAPER
RESULT_DIR = RESULTS_ROOT / PAPER
RUN_DIR = RUNS_ROOT / PAPER

paths = {
    "raw": DATA_DIR / "raw",
    "features": DATA_DIR / "features",
    "labels": DATA_DIR / "labels",
    "checkpoints": MODEL_DIR / "checkpoints",
    "predictions": RESULT_DIR / "predictions",
    "metrics": RESULT_DIR / "metrics",
    "slurm": RUN_DIR / "slurm",
}
for p in paths.values():
    p.mkdir(parents=True, exist_ok=True)
print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
'''),
        md(r'''
## Step 1 - Benchmark contract

For faithful reproduction, train using data available before CASP14 and use the CASP14 template boundary. Enhanced runs can use newer databases or architectures, but must be separate experiment rows.

Scientifically, AF2's CASP14 result was a generalization claim under a temporal information boundary. If we train or template against later structures, we are no longer asking whether the architecture reproduces the paper; we are asking a different, easier question.

Computationally, the contract becomes metadata attached to every feature tensor and checkpoint. Mathematically, it fixes the train/test split and the allowed conditioning variables, so score changes can be attributed to model or input changes instead of hidden leakage.
'''),
        code(r'''
contract = {
    "paper": "Jumper et al. 2021",
    "benchmark": "CASP14 monomer targets",
    "max_template_date": "2020-05-14",
    "implementation": "own_pytorch",
    "uses_official_weights_or_api": False,
}
write_text(DATA_DIR / "benchmark_contract.json", json.dumps(contract, indent=2))
print(json.dumps(contract, indent=2))
'''),
        md(r'''
## Step 2 - Feature representation

AF2's practical breakthrough is not just a larger network; it is the representation. The MSA track stores evolutionary variation across sequences, while the pair track stores residue-residue geometry. The two tracks repeatedly talk to each other.

Scientifically, the MSA dimension captures evolutionary experiments performed by nature, and the pair dimension captures geometric relationships that must become consistent in 3D. AF2's strength comes from letting these two views iteratively constrain each other.

Computationally, we keep two tensors: `msa` with shape `[N_msa, L, C_msa]` and `pair` with shape `[L, L, C_pair]`. Mathematically, row/column operations on the MSA estimate residue-specific and co-evolutionary signals, while pair updates approximate a learned function over residue-residue potentials.
'''),
        code(r'''
class AF2FeatureDataset(Dataset):
    def __init__(self, feature_dir: Path, synthetic_if_empty: bool = True, n_synthetic: int = 6, n_msa: int = 32, length: int = 96):
        self.files = sorted(feature_dir.glob("*.npz"))
        self.synthetic_if_empty = synthetic_if_empty
        self.n_synthetic = n_synthetic
        self.n_msa = n_msa
        self.length = length

    def __len__(self):
        return len(self.files) if self.files else (self.n_synthetic if self.synthetic_if_empty else 0)

    def __getitem__(self, idx):
        if self.files:
            arr = np.load(self.files[idx])
            return {k: torch.as_tensor(arr[k]) for k in arr.files}
        N, L = self.n_msa, self.length
        msa = F.one_hot(torch.randint(0, 22, (N, L)), 22).float()
        residue_index = torch.arange(L)
        relpos = (residue_index[:, None] - residue_index[None, :]).clamp(-32, 32) + 32
        pair = F.one_hot(relpos, 65).float()
        true_ca = torch.cumsum(torch.randn(L, 3) * 0.5 + torch.tensor([3.8, 0.0, 0.0]), dim=0)
        return {"msa": msa, "pair": pair, "true_ca": true_ca}

dataset = AF2FeatureDataset(paths["features"])
sample = dataset[0]
print({k: tuple(v.shape) for k, v in sample.items()})
'''),
        md(r'''
## Step 3 - Evoformer-like blocks

This is a compact research implementation, not a line-by-line clone. It keeps the essential message passing:

- MSA row attention: compare residues within each aligned sequence.
- MSA column summary: aggregate evolutionary signal into the pair track.
- Pair update: reason over residue-pair features.

Full AF2 adds triangle multiplication/attention, extra normalization details, template embedding, recycling features, and many loss heads. We add those incrementally after the baseline trains.

Scientifically, an Evoformer block is a consistency engine: residue-pair beliefs should agree with MSA evidence and with other pairs that form triangles in 3D space. A contact between `i` and `k` plus a contact between `k` and `j` constrains what can be true about `i` and `j`.

Computationally, this compact version uses attention and feed-forward updates rather than the full AF2 block, keeping the notebook runnable while preserving the information flow. Mathematically, the block alternates learned transformations `MSA <- f(MSA, pair)` and `pair <- g(pair, MSA summary)`, approximating iterative message passing over a complete residue graph.
'''),
        code(r'''
class EvoformerLiteBlock(nn.Module):
    def __init__(self, msa_dim: int, pair_dim: int, heads: int = 4):
        super().__init__()
        self.msa_attn = nn.MultiheadAttention(msa_dim, heads, batch_first=True)
        self.msa_ff = nn.Sequential(nn.LayerNorm(msa_dim), nn.Linear(msa_dim, 4 * msa_dim), nn.GELU(), nn.Linear(4 * msa_dim, msa_dim))
        self.outer = nn.Linear(msa_dim, pair_dim)
        self.pair_conv = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, 4 * pair_dim),
            nn.GELU(),
            nn.Linear(4 * pair_dim, pair_dim),
        )

    def forward(self, msa, pair):
        B, N, L, C = msa.shape
        msa_flat = msa.reshape(B * N, L, C)
        attn, _ = self.msa_attn(msa_flat, msa_flat, msa_flat, need_weights=False)
        msa = (msa_flat + attn).reshape(B, N, L, C)
        msa = msa + self.msa_ff(msa)
        summary = msa.mean(dim=1)
        pair_update = self.outer(summary[:, :, None, :] + summary[:, None, :, :])
        pair = pair + pair_update
        pair = pair + self.pair_conv(pair)
        return msa, pair


class AF2Lite(nn.Module):
    def __init__(self, msa_in: int = 22, pair_in: int = 65, msa_dim: int = 128, pair_dim: int = 128, blocks: int = 6):
        super().__init__()
        self.msa_embed = nn.Linear(msa_in, msa_dim)
        self.pair_embed = nn.Linear(pair_in, pair_dim)
        self.blocks = nn.ModuleList([EvoformerLiteBlock(msa_dim, pair_dim) for _ in range(blocks)])
        self.coord_head = nn.Sequential(nn.LayerNorm(pair_dim + msa_dim), nn.Linear(pair_dim + msa_dim, 256), nn.GELU(), nn.Linear(256, 3))
        self.plddt_head = nn.Sequential(nn.LayerNorm(msa_dim), nn.Linear(msa_dim, 50))

    def forward(self, msa, pair, recycles: int = 1):
        msa = self.msa_embed(msa)
        pair = self.pair_embed(pair)
        for _ in range(recycles):
            for block in self.blocks:
                msa, pair = block(msa, pair)
        single = msa[:, 0]
        pair_context = pair.mean(dim=2)
        ca = self.coord_head(torch.cat([single, pair_context], dim=-1))
        plddt_logits = self.plddt_head(single)
        return {"ca": ca, "plddt_logits": plddt_logits, "pair": pair, "single": single}

model = AF2Lite().to(device())
with torch.no_grad():
    out = model(sample["msa"][None].to(device()), sample["pair"][None].to(device()))
print({k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)})
'''),
        md(r'''
## Step 4 - Structure losses

AF2 uses a rich loss suite. The minimal baseline here combines coordinate distance geometry and local confidence supervision. The coordinate loss is expressed through pairwise distances so it is rotation/translation invariant; later we can add frame-aligned point error, violation losses, distogram heads, and pLDDT calibration.

Scientifically, a protein prediction should be judged by geometry rather than absolute coordinate frame: rotating or translating a perfect model should not change the loss. Pairwise distances capture fold geometry without requiring a rigid alignment inside the training loop.

Computationally, `torch.cdist` gives a differentiable distance matrix, and smooth L1 reduces sensitivity to large early errors. Mathematically, the loss compares `D_hat_ij = ||x_hat_i - x_hat_j||` to `D_ij = ||x_i - x_j||`, plus a bond-length regularizer that biases neighboring C-alpha atoms toward plausible spacing.
'''),
        code(r'''
def pairwise_distance_loss(pred_ca, true_ca):
    pred_d = torch.cdist(pred_ca, pred_ca)
    true_d = torch.cdist(true_ca, true_ca)
    return F.smooth_l1_loss(pred_d, true_d)

def af2_lite_loss(outputs, true_ca):
    geom = pairwise_distance_loss(outputs["ca"], true_ca)
    bond = ((outputs["ca"][:, 1:] - outputs["ca"][:, :-1]).norm(dim=-1) - 3.8).pow(2).mean()
    return geom + 0.05 * bond

loader = DataLoader(dataset, batch_size=1, shuffle=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
model.train()
for step, item in enumerate(loader):
    item = {k: v.to(device()) for k, v in item.items()}
    opt.zero_grad(set_to_none=True)
    outputs = model(item["msa"], item["pair"], recycles=1)
    loss = af2_lite_loss(outputs, item["true_ca"])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    print(f"step={step} loss={float(loss.detach().cpu()):.4f}")
    if step >= 2:
        break
'''),
        md(r'''
## Step 5 - Recycling and inference

Recycling lets the model refine a structure hypothesis by feeding information from an earlier pass back into the network. The compact model above has a `recycles` argument; the next implementation iteration should inject previous distance/coordinate embeddings into the pair representation rather than merely repeating the blocks.

Scientifically, folding is a self-consistency problem: an initial hypothesis reveals clashes, domain arrangements, and long-range constraints that can guide a second pass. AF2's recycling mechanism exploits that by making prediction iterative rather than one-shot.

Computationally, recycling trades runtime and memory for refinement. Mathematically, it applies a recurrence `h_{t+1} = F(h_t, features, structure_t)` and hopes the sequence converges toward a geometrically consistent fixed point. In this first notebook, repeated blocks are a scaffold for the fuller recurrence.
'''),
        code(r'''
model.eval()
with torch.no_grad():
    item = {k: v[None].to(device()) for k, v in dataset[0].items()}
    one = model(item["msa"], item["pair"], recycles=1)["ca"]
    three = model(item["msa"], item["pair"], recycles=3)["ca"]
print({"one_recycle_shape": tuple(one.shape), "three_recycle_shape": tuple(three.shape)})
'''),
        md(r'''
## Step 6 - Scoreboard and improvement track

The faithful row should use CASP14 targets, dated training/template boundaries, and this PyTorch implementation. Enhanced rows can use larger models, more recycles, modern databases, or new rerankers.

Scientifically, improvement claims need paired comparisons: the same targets, the same scoring metrics, and one controlled change at a time. Otherwise a better headline number may simply reflect a different benchmark surface.

Computationally, the registry is a small experiment database. Mathematically, each row defines a function from inputs to structures plus a metric vector; later we can compute deltas, confidence intervals, per-target wins/losses, and failure clusters.
'''),
        code(r'''
experiments = [
    {"name": "af2_lite_faithful_casp14", "faithful": True, "change": "own Evoformer-lite, CASP14 boundary"},
    {"name": "af2_lite_triangle_updates", "faithful": True, "change": "add triangle multiplication and attention"},
    {"name": "af2_lite_more_recycles", "faithful": False, "change": "extra inference recycles and seeds"},
    {"name": "af2_lite_modern_msa", "faithful": False, "change": "modern sequence databases"},
]
write_text(RUN_DIR / "experiment_registry.json", json.dumps(experiments, indent=2))
print(json.dumps(experiments, indent=2))
'''),
        md(SCORING_AND_CLUSTER),
        code(r'''
slurm = paths["slurm"] / "train_af2_lite.sbatch"
slurm.write_text(f"""#!/usr/bin/env bash
#SBATCH --job-name=af2-lite
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output={paths['slurm']}/%x-%j.out
set -euo pipefail
cd "{PROJECT_ROOT}"
jupyter nbconvert --to notebook --execute 02_Jumper_2021_AlphaFold2_CASP14_Reproduction.ipynb --output results/jumper_2021/executed.ipynb
""", encoding="utf-8")
print(slurm.read_text(encoding="utf-8"))
'''),
    ])


def abramson_notebook() -> dict:
    return notebook([
        md(r'''
# 03 - Abramson et al. 2024: AlphaFold3 Reproduction

This notebook is a **from-scratch PyTorch/RDKit reproduction** of AlphaFold3's central ideas. It does not use the official AF3 code, API, Docker image, or model parameters.

We assume RDKit is installed, as requested. RDKit is used for ligand parsing, atom/bond features, conformers, stereochemistry, and chemistry-aware benchmark preparation.

AF3's conceptual steps:

1. Represent all biomolecules as tokens and atoms: proteins, nucleic acids, ligands, ions, and modified residues.
2. Build pair and single representations with molecule-aware features.
3. Use Pairformer-style pair/single updates.
4. Predict coordinates with a diffusion-style denoising model.
5. Score across protein, nucleic-acid, ligand, and interface metrics.
'''),
        md(r'''
## Layout and assumptions

Your A100 80GB matches the class of GPU needed for serious AF3-style inference experiments. The notebook assumes RDKit will be installed before execution on the cluster. If the import fails, install RDKit first and rerun; this notebook intentionally does not fall back to a non-chemistry path.
'''),
        code(COMMON_SETUP),
        code(r'''
PAPER = "abramson_2024"
DATA_DIR = DATA_ROOT / PAPER
MODEL_DIR = MODEL_ROOT / PAPER
RESULT_DIR = RESULTS_ROOT / PAPER
RUN_DIR = RUNS_ROOT / PAPER

paths = {
    "raw": DATA_DIR / "raw",
    "features": DATA_DIR / "features",
    "ligands": DATA_DIR / "ligands",
    "benchmarks": DATA_DIR / "benchmarks",
    "checkpoints": MODEL_DIR / "checkpoints",
    "predictions": RESULT_DIR / "predictions",
    "metrics": RESULT_DIR / "metrics",
    "slurm": RUN_DIR / "slurm",
}
for p in paths.values():
    p.mkdir(parents=True, exist_ok=True)
print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
'''),
        md(r'''
## Step 1 - RDKit chemistry layer

AF3's protein-ligand benchmark surface is impossible to treat seriously without chemistry. RDKit gives us explicit atoms, formal charges, aromaticity, hybridization, bonds, stereochemistry, and conformers. Those features become ligand tokens/atoms for our own model.

Scientifically, ligands are not just residue-like tokens: bond order, charge, aromaticity, stereochemistry, and protonation state determine which poses are chemically possible. A model that ignores those constraints can appear close by RMSD while predicting an invalid molecule.

Computationally, RDKit turns SMILES/SDF chemistry into graph features and initial conformers. Mathematically, the ligand is represented as a graph `G = (V, E)` with atom feature matrix `X_v`, bond feature matrix `X_e`, and coordinates `R`. These features become conditioning variables for the neural coordinate model.
'''),
        code(r'''
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

print("RDKit version:", rdkit.__version__)

ATOM_TYPES = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "H", "OTHER"]
BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]

def one_hot_index(value, choices):
    idx = choices.index(value) if value in choices else len(choices) - 1
    out = torch.zeros(len(choices))
    out[idx] = 1.0
    return out

def featurize_ligand_smiles(smiles: str, seed: int = 7):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, randomSeed=seed)
    if status == 0:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    conf = mol.GetConformer()
    atom_features, coords = [], []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atom_features.append(torch.cat([
            one_hot_index(symbol, ATOM_TYPES),
            torch.tensor([
                atom.GetFormalCharge(),
                float(atom.GetIsAromatic()),
                atom.GetTotalDegree(),
                atom.GetTotalValence(),
            ], dtype=torch.float32),
        ]))
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append([p.x, p.y, p.z])
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = one_hot_index(bond.GetBondType(), BOND_TYPES)
        edge_index += [[i, j], [j, i]]
        edge_attr += [feat, feat]
    return {
        "atom_features": torch.stack(atom_features),
        "atom_coords": torch.tensor(coords, dtype=torch.float32),
        "edge_index": torch.tensor(edge_index, dtype=torch.long).T if edge_index else torch.empty(2, 0, dtype=torch.long),
        "edge_attr": torch.stack(edge_attr) if edge_attr else torch.empty(0, len(BOND_TYPES)),
        "canonical_smiles": Chem.MolToSmiles(Chem.RemoveHs(mol)),
    }

lig = featurize_ligand_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
print({k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in lig.items()})
'''),
        md(r'''
## Step 2 - Multimolecular tokenization

AF3 treats proteins, nucleic acids, and ligands in one coordinate-generating system. This simplified tokenizer creates:

- Protein residue tokens from amino-acid sequence.
- Ligand atom tokens from RDKit atom features.
- Pair features from sequence distance, molecule identity, and ligand bonds.

This is enough to train a tiny denoising model now and swap in real benchmark complexes later.

Scientifically, AF3's key expansion is that proteins, nucleic acids, ligands, ions, and modifications share one interaction model. Tokenization is where we decide what the model is allowed to know about each molecule and how cross-molecule contacts can be represented.

Computationally, this step concatenates protein residue tokens and ligand atom tokens, then builds pair features for sequence distance, molecule identity, and ligand bonds. Mathematically, we create a single token set of size `T` with single features `S in R^{T x C}` and pair features `P in R^{T x T x C_p}`; the model's job is to infer coordinates for all tokens jointly.
'''),
        code(r'''
AA = "ACDEFGHIKLMNPQRSTVWYX"

def tokenize_complex(sequence: str, ligand_smiles: str):
    protein_tokens = F.one_hot(torch.tensor([AA.index(a) if a in AA else AA.index("X") for a in sequence]), len(AA)).float()
    lig = featurize_ligand_smiles(ligand_smiles)
    ligand_tokens = F.pad(lig["atom_features"], (0, protein_tokens.shape[-1] - lig["atom_features"].shape[-1])) if lig["atom_features"].shape[-1] < protein_tokens.shape[-1] else lig["atom_features"][:, :protein_tokens.shape[-1]]
    single = torch.cat([protein_tokens, ligand_tokens], dim=0)
    molecule_id = torch.cat([torch.zeros(len(protein_tokens), dtype=torch.long), torch.ones(len(ligand_tokens), dtype=torch.long)])
    L = single.shape[0]
    pair = torch.zeros(L, L, 8)
    idx = torch.arange(L)
    pair[..., 0] = (idx[:, None] - idx[None, :]).abs().float().clamp(max=32) / 32
    pair[..., 1] = (molecule_id[:, None] == molecule_id[None, :]).float()
    pair[..., 2] = (molecule_id[:, None] != molecule_id[None, :]).float()
    offset = len(protein_tokens)
    for e in lig["edge_index"].T:
        i, j = int(e[0]) + offset, int(e[1]) + offset
        pair[i, j, 3] = 1.0
    true_coords = torch.cat([
        torch.cumsum(torch.randn(len(protein_tokens), 3) * 0.4 + torch.tensor([3.8, 0.0, 0.0]), dim=0),
        lig["atom_coords"] + torch.tensor([0.0, 8.0, 0.0]),
    ], dim=0)
    return {"single": single, "pair": pair, "true_coords": true_coords, "molecule_id": molecule_id}

complex_features = tokenize_complex("MSTNPKPQRKTKRNTNRRPQDVKFPGG", "CCO")
print({k: tuple(v.shape) for k, v in complex_features.items()})
'''),
        md(r'''
## Step 3 - Pairformer-style model

AF3 simplifies AF2's Evoformer into Pairformer-style updates over single and pair representations. This compact version alternates single self-attention with pair-conditioned updates. It is intentionally small but structurally honest: the model reasons over cross-molecule pair features before denoising atom coordinates.

Scientifically, interactions are pairwise and contextual: a ligand atom's placement depends on nearby protein residues, and a residue's relevance depends on the ligand pose. Pairformer-style updates give the network a place to represent these cross-molecule hypotheses.

Computationally, single-token attention mixes information across the complex, while pair updates maintain explicit `T x T` relational state. Mathematically, the block alternates updates to `S_i` and `P_ij`; the diffusion head then predicts coordinate noise from a combination of token state and averaged pair context.
'''),
        code(r'''
class PairformerLiteBlock(nn.Module):
    def __init__(self, single_dim: int, pair_dim: int, heads: int = 4):
        super().__init__()
        self.single_attn = nn.MultiheadAttention(single_dim, heads, batch_first=True)
        self.pair_to_bias = nn.Linear(pair_dim, heads)
        self.single_ff = nn.Sequential(nn.LayerNorm(single_dim), nn.Linear(single_dim, 4 * single_dim), nn.GELU(), nn.Linear(4 * single_dim, single_dim))
        self.pair_ff = nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim + single_dim, 4 * pair_dim), nn.GELU(), nn.Linear(4 * pair_dim, pair_dim))

    def forward(self, single, pair):
        attn, _ = self.single_attn(single, single, single, need_weights=False)
        single = single + attn
        single = single + self.single_ff(single)
        pair_single = single[:, :, None, :] + single[:, None, :, :]
        pair = pair + self.pair_ff(torch.cat([pair, pair_single], dim=-1))
        return single, pair


class AF3DiffusionLite(nn.Module):
    def __init__(self, token_in: int = 21, pair_in: int = 8, single_dim: int = 128, pair_dim: int = 128, blocks: int = 6):
        super().__init__()
        self.single_embed = nn.Linear(token_in, single_dim)
        self.pair_embed = nn.Linear(pair_in, pair_dim)
        self.time_embed = nn.Sequential(nn.Linear(1, single_dim), nn.SiLU(), nn.Linear(single_dim, single_dim))
        self.blocks = nn.ModuleList([PairformerLiteBlock(single_dim, pair_dim) for _ in range(blocks)])
        self.noise_head = nn.Sequential(nn.LayerNorm(single_dim + pair_dim), nn.Linear(single_dim + pair_dim, 256), nn.SiLU(), nn.Linear(256, 3))
        self.confidence_head = nn.Sequential(nn.LayerNorm(single_dim), nn.Linear(single_dim, 50))

    def forward(self, single, pair, noisy_coords, t):
        single = self.single_embed(single) + self.time_embed(t[:, None, None].float())
        pair = self.pair_embed(pair)
        coord_dist = torch.cdist(noisy_coords, noisy_coords)[..., None] / 20.0
        pair = pair + F.pad(coord_dist, (0, pair.shape[-1] - 1))
        for block in self.blocks:
            single, pair = block(single, pair)
        pair_context = pair.mean(dim=2)
        noise = self.noise_head(torch.cat([single, pair_context], dim=-1))
        confidence = self.confidence_head(single)
        return {"pred_noise": noise, "confidence_logits": confidence, "single": single, "pair": pair}

model = AF3DiffusionLite().to(device())
item = {k: v[None].to(device()) for k, v in complex_features.items() if k in ["single", "pair", "true_coords"]}
with torch.no_grad():
    t = torch.tensor([0.5], device=device())
    noisy = item["true_coords"] + torch.randn_like(item["true_coords"]) * t[:, None, None]
    out = model(item["single"], item["pair"], noisy, t)
print({k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)})
'''),
        md(r'''
## Step 4 - Diffusion training objective

The diffusion module learns to remove noise from atom coordinates conditioned on molecule features. We sample a noise level `t`, corrupt the true coordinates, and train the model to predict the injected noise. Later iterations should add AF3-style multi-step sampling, confidence calibration, bond/angle violation losses, clash losses, and molecule-class-specific metrics.

Scientifically, diffusion is useful because biomolecular complexes can have multiple plausible poses and conformations. Instead of predicting one deterministic coordinate set, the model learns a denoising process that can sample structures from a learned distribution.

Computationally, each training example draws random noise and a noise scale, which creates many supervised denoising tasks from the same structure. Mathematically, we sample `epsilon ~ N(0, I)`, form `x_t = x_0 + t epsilon`, and minimize `||epsilon_hat_theta(x_t, t, features) - epsilon||^2`. This is the score-matching core of the simplified diffusion objective.
'''),
        code(r'''
class AF3ToyComplexDataset(Dataset):
    def __init__(self, n: int = 8):
        self.rows = [
            ("MSTNPKPQRKTKRNTNRRPQDVKFPGG", "CCO"),
            ("ACDEFGHIKLMNPQRSTVWY", "CC(=O)O"),
            ("GGGGSGGGGSGGGGS", "c1ccccc1"),
            ("MEEPQSDPSVEPPLSQETFSDLWKLL", "CCN(CC)CC"),
        ] * ((n + 3) // 4)
        self.rows = self.rows[:n]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        sequence, smiles = self.rows[idx]
        return tokenize_complex(sequence, smiles)

def collate_one(batch):
    # Keep batch size 1 until padding/masking is added.
    return {k: v[None] for k, v in batch[0].items()}

toy = AF3ToyComplexDataset()
loader = DataLoader(toy, batch_size=1, shuffle=True, collate_fn=collate_one)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
model.train()
for step, item in enumerate(loader):
    item = {k: v.to(device()) for k, v in item.items()}
    t = torch.rand(item["true_coords"].shape[0], device=device()).clamp_min(0.05)
    noise = torch.randn_like(item["true_coords"])
    noisy = item["true_coords"] + noise * t[:, None, None]
    opt.zero_grad(set_to_none=True)
    outputs = model(item["single"], item["pair"], noisy, t)
    denoise_loss = F.mse_loss(outputs["pred_noise"], noise)
    opt.zero_grad(set_to_none=True)
    denoise_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    print(f"step={step} denoise_loss={float(denoise_loss.detach().cpu()):.4f}")
    if step >= 2:
        break
'''),
        md(r'''
## Step 5 - Sampling

A trained diffusion model starts from noisy coordinates and repeatedly denoises. This minimal sampler is Euler-style and intentionally simple. The important point is that generation is ours: PyTorch tensors in, PyTorch tensors out, with RDKit-derived ligand features.

Scientifically, sampling turns the learned distribution into concrete candidate complexes. Multiple samples are important for AF3-like tasks because ligand binding poses, flexible loops, and interfaces may be ambiguous from sequence and chemistry alone.

Computationally, the sampler runs the neural network several times while reducing noise. Mathematically, it approximates the reverse denoising trajectory with discrete updates `x_{t-dt} = x_t - dt * epsilon_hat_theta(...)`; later we can replace this with a better scheduler and sample reranking.
'''),
        code(r'''
@torch.no_grad()
def sample_complex(model, features, steps: int = 20):
    model.eval()
    single = features["single"][None].to(device())
    pair = features["pair"][None].to(device())
    coords = torch.randn_like(features["true_coords"][None].to(device()))
    for s in reversed(range(steps)):
        t = torch.full((1,), (s + 1) / steps, device=device())
        pred_noise = model(single, pair, coords, t)["pred_noise"]
        coords = coords - pred_noise / steps
    return coords[0].detach().cpu()

sampled = sample_complex(model, complex_features, steps=5)
print(tuple(sampled.shape), sampled[:3])
'''),
        md(r'''
## Step 6 - AF3 benchmark surfaces

For AF3, protein TM-score alone is insufficient. The score table must cover protein, ligand, nucleic acid, and interface quality. RDKit will also support chemistry validity checks for ligand predictions.

Scientifically, AF3's claim is about biomolecular interactions, so each molecular class needs metrics aligned with its biology. A protein backbone can be globally correct while the ligand pose is wrong, or a ligand can be close while the interface chemistry is invalid.

Computationally, we store a benchmark manifest that maps each target to required input files and metric families. Mathematically, the final evaluation is a vector-valued score, not a scalar: global fold terms, local/interface terms, ligand RMSD, clash/validity terms, and calibration terms should be tracked separately before any aggregate is reported.
'''),
        code(r'''
benchmark_manifest = [
    {
        "target_id": "protein_ligand_example",
        "inputs": {"protein_fasta": "data/abramson_2024/raw/protein_ligand_example.fasta", "ligand_sdf": "data/abramson_2024/ligands/protein_ligand_example.sdf"},
        "truth_cif": "data/abramson_2024/benchmarks/protein_ligand_example.cif",
        "metrics": ["protein_tm_score", "ligand_rmsd", "interface_lddt", "clash_count", "rdkit_sanitizes"],
        "faithful": True,
    },
    {
        "target_id": "protein_nucleic_acid_example",
        "inputs": {"complex_fasta": "data/abramson_2024/raw/protein_nucleic_acid_example.fasta"},
        "truth_cif": "data/abramson_2024/benchmarks/protein_nucleic_acid_example.cif",
        "metrics": ["interface_lddt", "nucleic_acid_rmsd", "clash_count"],
        "faithful": True,
    },
]
write_text(paths["benchmarks"] / "benchmark_manifest.example.json", json.dumps(benchmark_manifest, indent=2))
print(json.dumps(benchmark_manifest, indent=2))
'''),
        md(r'''
## Step 7 - Improvement track

The baseline is our own Pairformer/diffusion implementation. The first improvement arms should focus on places where AF3's benchmark is sensitive:

1. More diffusion samples and better confidence reranking.
2. RDKit ligand state ensembles: protonation, tautomer, stereochemistry, conformers.
3. Cross-molecule pair features for contacts, bonds, covalent links, and templates.
4. Chemistry validity losses and post-sampling restrained minimization.
5. Class-specific scoring so ligand improvements do not hide protein regressions.

Scientifically, the improvement loop tests hypotheses about what limits performance: sampling diversity, ligand chemistry states, cross-molecule features, or physical validity. Each arm should correspond to a biological or modeling reason, not just a parameter tweak.

Computationally, each experiment is a reproducible transformation of data, model, sampler, or scorer. Mathematically, we compare metric vectors per target and per molecule class, looking for consistent positive deltas rather than a single averaged number that could hide regressions.
'''),
        code(r'''
experiments = [
    {"name": "af3_lite_faithful_single_sample", "faithful": True, "change": "own Pairformer/diffusion baseline"},
    {"name": "af3_lite_8_sample_rerank", "faithful": False, "change": "more diffusion samples plus confidence/validity reranking"},
    {"name": "af3_lite_ligand_state_ensemble", "faithful": False, "change": "RDKit tautomer/protonation/conformer ensemble"},
    {"name": "af3_lite_chemistry_losses", "faithful": False, "change": "bond, clash, chirality, and validity losses"},
]
write_text(RUN_DIR / "experiment_registry.json", json.dumps(experiments, indent=2))
print(json.dumps(experiments, indent=2))
'''),
        md(SCORING_AND_CLUSTER),
        code(r'''
slurm = paths["slurm"] / "train_af3_lite.sbatch"
slurm.write_text(f"""#!/usr/bin/env bash
#SBATCH --job-name=af3-lite
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output={paths['slurm']}/%x-%j.out
set -euo pipefail
cd "{PROJECT_ROOT}"
jupyter nbconvert --to notebook --execute 03_Abramson_2024_AlphaFold3_Interaction_Reproduction.ipynb --output results/abramson_2024/executed.ipynb
""", encoding="utf-8")
print(slurm.read_text(encoding="utf-8"))
'''),
    ])


def main() -> None:
    notebooks = {
        "01_Senior_2020_AlphaFold_CASP13_Reproduction.ipynb": senior_notebook(),
        "02_Jumper_2021_AlphaFold2_CASP14_Reproduction.ipynb": jumper_notebook(),
        "03_Abramson_2024_AlphaFold3_Interaction_Reproduction.ipynb": abramson_notebook(),
    }
    for name, nb in notebooks.items():
        path = ROOT / name
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
