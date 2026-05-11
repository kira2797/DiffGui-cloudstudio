import argparse
import glob
import json
import os
import random
import sys

sys.path.append(".")

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem

from models.frequency_surrogate import FrequencySurrogate
from utils.transforms import get_index


def spectrum_vector(freqs, intensities, grid_min, grid_max, grid_step, sigma):
    grid = np.arange(grid_min, grid_max + grid_step, grid_step, dtype=float)
    vec = np.zeros_like(grid, dtype=float)
    if not freqs:
        return vec
    if not intensities or len(intensities) != len(freqs) or sum(intensities) <= 0:
        intensities = [1.0 for _ in freqs]
    for freq, intensity in zip(freqs, intensities):
        vec += float(intensity) * np.exp(-0.5 * ((grid - float(freq)) / sigma) ** 2)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def find_sdf_dir(run_dir):
    candidates = sorted(glob.glob(os.path.join(run_dir, "*_SDF")))
    if not candidates:
        raise FileNotFoundError("No *_SDF directory found under %s" % run_dir)
    if len(candidates) > 1:
        raise ValueError("Multiple *_SDF directories found: %s" % ", ".join(candidates))
    return candidates[0]


def numeric_sdf_key(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(stem)
    except ValueError:
        return stem


def iter_candidate_sdfs(sdf_dir):
    paths = []
    for path in glob.glob(os.path.join(sdf_dir, "*.sdf")):
        if os.path.basename(path).startswith("traj_"):
            continue
        paths.append(path)
    return sorted(paths, key=numeric_sdf_key)


def read_mol(path):
    mol = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)[0]
    if mol is None:
        raise ValueError("RDKit failed to read %s" % path)
    try:
        mol = Chem.RemoveHs(mol, sanitize=False)
    except TypeError:
        mol = Chem.RemoveHs(mol)
    if mol.GetNumConformers() == 0:
        raise ValueError("No conformer in %s" % path)
    return mol


def mol_to_tensors(path, atom_mode):
    mol = read_mol(path)
    conf = mol.GetConformer()
    atom_idx = []
    pos = []
    for atom in mol.GetAtoms():
        idx = get_index(
            atom.GetAtomicNum(),
            atom.GetHybridization(),
            atom.GetIsAromatic(),
            atom_mode,
        )
        atom_idx.append(idx)
        p = conf.GetAtomPosition(atom.GetIdx())
        pos.append([p.x, p.y, p.z])
    return torch.tensor(atom_idx, dtype=torch.long), torch.tensor(pos, dtype=torch.float32)


def load_frequency_json(path, grid_min, grid_max, grid_step, sigma):
    with open(path, "r") as f:
        data = json.load(f)
    if data.get("status") != "ok":
        return None
    freqs = data.get("frequencies", [])
    intensities = data.get("intensities", [])
    if not freqs:
        return None
    return torch.tensor(
        spectrum_vector(freqs, intensities, grid_min, grid_max, grid_step, sigma),
        dtype=torch.float32,
    )


def collect_examples(run_dirs, atom_mode, grid_min, grid_max, grid_step, sigma):
    examples = []
    skipped = 0
    for run_dir in run_dirs:
        sdf_dir = find_sdf_dir(run_dir)
        for sdf_path in iter_candidate_sdfs(sdf_dir):
            mol_id = os.path.splitext(os.path.basename(sdf_path))[0]
            freq_path = os.path.join(run_dir, "freq_work", "mol_" + mol_id, "frequencies.json")
            if not os.path.exists(freq_path):
                skipped += 1
                continue
            try:
                y = load_frequency_json(freq_path, grid_min, grid_max, grid_step, sigma)
                if y is None:
                    skipped += 1
                    continue
                atom_idx, pos = mol_to_tensors(sdf_path, atom_mode)
            except Exception:
                skipped += 1
                continue
            examples.append(
                {
                    "mol_id": mol_id,
                    "sdf_path": sdf_path,
                    "atom_idx": atom_idx,
                    "pos": pos,
                    "spectrum": y,
                }
            )
    return examples, skipped


def collate(examples, atom_feat_dim, device):
    atom_idx = torch.cat([ex["atom_idx"] for ex in examples], dim=0).to(device)
    node_h = F.one_hot(atom_idx, num_classes=atom_feat_dim).float()
    pos = torch.cat([ex["pos"] for ex in examples], dim=0).to(device)
    batch = torch.repeat_interleave(
        torch.arange(len(examples), dtype=torch.long),
        torch.tensor([ex["atom_idx"].numel() for ex in examples], dtype=torch.long),
    ).to(device)
    y = torch.stack([ex["spectrum"] for ex in examples], dim=0).to(device)
    return node_h, pos, batch, y


def atom_feat_dim(atom_mode):
    if atom_mode == "basic":
        return 11
    if atom_mode == "aromatic":
        return 16
    if atom_mode == "full":
        return 26
    raise ValueError(atom_mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", action="append", required=True)
    parser.add_argument("--out", type=str, default="ckpt/frequency_surrogate.pt")
    parser.add_argument("--atom_mode", type=str, default="aromatic", choices=["basic", "aromatic", "full"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--distance_dim", type=int, default=48)
    parser.add_argument("--distance_cutoff", type=float, default=12.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--grid_min", type=float, default=0.0)
    parser.add_argument("--grid_max", type=float, default=4000.0)
    parser.add_argument("--grid_step", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=50.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    examples, skipped = collect_examples(
        args.run_dir,
        args.atom_mode,
        args.grid_min,
        args.grid_max,
        args.grid_step,
        args.sigma,
    )
    if len(examples) < 5:
        raise ValueError(
            "Need at least 5 molecules with xTB frequencies; got %d, skipped %d."
            % (len(examples), skipped)
        )

    random.shuffle(examples)
    n_val = max(1, int(round(len(examples) * 0.15)))
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    model_config = {
        "atom_feat_dim": atom_feat_dim(args.atom_mode),
        "spectrum_dim": int(round((args.grid_max - args.grid_min) / args.grid_step)) + 1,
        "hidden_dim": args.hidden_dim,
        "distance_dim": args.distance_dim,
        "distance_cutoff": args.distance_cutoff,
    }
    model = FrequencySurrogate(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_examples)
        train_losses = []
        for start in range(0, len(train_examples), args.batch_size):
            batch_examples = train_examples[start : start + args.batch_size]
            node_h, pos, batch, y = collate(batch_examples, model_config["atom_feat_dim"], device)
            pred = model(node_h, pos, batch, num_graphs=len(batch_examples))
            loss = (1.0 - F.cosine_similarity(pred, y, dim=-1)).mean()
            loss = loss + 0.1 * F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            node_h, pos, batch, y = collate(val_examples, model_config["atom_feat_dim"], device)
            pred = model(node_h, pos, batch, num_graphs=len(val_examples))
            val_loss = (1.0 - F.cosine_similarity(pred, y, dim=-1)).mean()
            val_loss = val_loss + 0.1 * F.mse_loss(pred, y)
            val_loss = float(val_loss.cpu())

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                "epoch=%d train=%.6f val=%.6f best=%.6f"
                % (epoch, float(np.mean(train_losses)), val_loss, best_val)
            )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "model_config": model_config,
            "atom_mode": args.atom_mode,
            "grid": {
                "grid_min": args.grid_min,
                "grid_max": args.grid_max,
                "grid_step": args.grid_step,
                "sigma": args.sigma,
            },
            "num_train": len(train_examples),
            "num_val": len(val_examples),
            "best_val_loss": best_val,
        },
        args.out,
    )
    print("Saved:", args.out)
    print("Examples:", len(examples), "Skipped:", skipped)


if __name__ == "__main__":
    main()
