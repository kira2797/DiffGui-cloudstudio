import argparse
import csv
import glob
import os
import sys

sys.path.append(".")

import numpy as np
from rdkit import Chem

from utils.evaluation.docking_vina import VinaDockingTask
from utils.evaluation.scoring_func import get_basic, get_chem


def find_sdf_dir(run_dir):
    candidates = sorted(glob.glob(os.path.join(run_dir, "*_SDF")))
    if not candidates:
        raise FileNotFoundError("No *_SDF directory found under %s" % run_dir)
    if len(candidates) > 1:
        raise ValueError("Multiple *_SDF directories found: %s" % ", ".join(candidates))
    return candidates[0]


def numeric_sdf_key(path):
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(name)
    except ValueError:
        return name


def iter_sdf_files(sdf_dir):
    paths = []
    for path in glob.glob(os.path.join(sdf_dir, "*.sdf")):
        name = os.path.basename(path)
        if name.startswith("traj_"):
            continue
        paths.append(path)
    return sorted(paths, key=numeric_sdf_key)


def mean_median(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.median(values))


def pdb_box_center(pdb_path):
    coords = []
    with open(pdb_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                coords.append(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                )
            except ValueError:
                continue
    if not coords:
        raise ValueError("No atom coordinates found in %s" % pdb_path)
    coords = np.asarray(coords, dtype=float)
    return ((coords.min(axis=0) + coords.max(axis=0)) / 2.0).tolist()


def evaluate_one(path, protein_path, docking_mode, exhaustiveness, box_center):
    row = {
        "mol_id": os.path.splitext(os.path.basename(path))[0],
        "sdf_path": path,
        "valid": 0,
        "smiles": "",
        "n_atoms": "",
        "n_bonds": "",
        "n_rings": "",
        "mol_weight": "",
        "qed": "",
        "sa": "",
        "logp": "",
        "lipinski": "",
        "tpsa": "",
        "vina_score_only": "",
        "vina_min": "",
        "vina_dock": "",
        "error": "",
    }

    try:
        mol = Chem.SDMolSupplier(path, removeHs=False)[0]
        if mol is None:
            row["error"] = "RDKit failed to read molecule"
            return row

        Chem.SanitizeMol(mol)
        row["valid"] = 1
        row["smiles"] = Chem.MolToSmiles(mol)

        n_atoms, n_bonds, n_rings, weight = get_basic(mol)
        row["n_atoms"] = n_atoms
        row["n_bonds"] = n_bonds
        row["n_rings"] = n_rings
        row["mol_weight"] = weight

        chem = get_chem(mol)
        row["qed"] = chem["qed"]
        row["sa"] = chem["sa"]
        row["logp"] = chem["logp"]
        row["lipinski"] = chem["lipinski"]
        row["tpsa"] = chem["tpsa"]

        if docking_mode != "none":
            if not protein_path:
                raise ValueError("--protein is required when docking_mode is not none")
            center = pdb_box_center(protein_path) if box_center == "pocket" else None
            size_factor = None if box_center == "pocket" else 1.0
            task = VinaDockingTask.from_generated_mol(
                mol,
                protein_path,
                center=center,
                size_factor=size_factor,
            )
            if docking_mode in ("score_only", "all"):
                row["vina_score_only"] = task.run(
                    mode="score_only", exhaustiveness=exhaustiveness
                )[0]["affinity"]
            if docking_mode in ("minimize", "all"):
                row["vina_min"] = task.run(
                    mode="minimize", exhaustiveness=exhaustiveness
                )[0]["affinity"]
            if docking_mode in ("dock", "all_dock"):
                row["vina_dock"] = task.run(
                    mode="dock", exhaustiveness=exhaustiveness
                )[0]["affinity"]

    except Exception as exc:
        row["error"] = repr(exc)

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--sdf_dir", type=str, default=None)
    parser.add_argument("--protein", type=str, default=None)
    parser.add_argument(
        "--docking_mode",
        type=str,
        default="all",
        choices=["none", "score_only", "minimize", "all", "dock", "all_dock"],
    )
    parser.add_argument("--exhaustiveness", type=int, default=16)
    parser.add_argument(
        "--box_center",
        type=str,
        default="ligand",
        choices=["ligand", "pocket"],
        help="Use generated ligand coordinates or receptor pocket coordinates as docking box center.",
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.sdf_dir is None and args.run_dir is None:
        raise ValueError("Please provide either --sdf_dir or --run_dir")

    sdf_dir = args.sdf_dir or find_sdf_dir(args.run_dir)
    out = args.out or os.path.join(os.path.dirname(sdf_dir), "sdf_eval.csv")

    rows = [
        evaluate_one(path, args.protein, args.docking_mode, args.exhaustiveness, args.box_center)
        for path in iter_sdf_files(sdf_dir)
    ]

    fieldnames = [
        "mol_id",
        "sdf_path",
        "valid",
        "smiles",
        "n_atoms",
        "n_bonds",
        "n_rings",
        "mol_weight",
        "qed",
        "sa",
        "logp",
        "lipinski",
        "tpsa",
        "vina_score_only",
        "vina_min",
        "vina_dock",
        "error",
    ]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", out)
    print("Total:", len(rows))
    print("Valid:", sum(int(row["valid"]) for row in rows))

    for key in ["qed", "sa", "logp", "lipinski", "tpsa", "vina_score_only", "vina_min", "vina_dock"]:
        values = []
        for row in rows:
            if row[key] == "":
                continue
            values.append(float(row[key]))
        mean, median = mean_median(values)
        if mean is not None:
            print("%s mean: %.4f median: %.4f" % (key, mean, median))


if __name__ == "__main__":
    main()
