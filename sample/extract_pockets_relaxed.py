import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from rdkit import Chem

from utils.data import PDBProtein, parse_drug3d_mol


def read_ligand_without_sanitize(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".sdf":
        mol = Chem.SDMolSupplier(path, sanitize=False, removeHs=False)[0]
    elif ext == ".mol2":
        mol = Chem.MolFromMol2File(path, sanitize=False, removeHs=False)
    else:
        raise ValueError("Unknown ligand format: %s. Use .sdf or .mol2" % path)
    return mol


def parse_ligand_positions(path):
    try:
        return parse_drug3d_mol(path)
    except Exception as exc:
        print(
            "[extract_pockets_relaxed] Standard ligand parsing failed: %s" % repr(exc),
            file=sys.stderr,
        )
        print(
            "[extract_pockets_relaxed] Falling back to unsanitized ligand coordinates.",
            file=sys.stderr,
        )
        mol = read_ligand_without_sanitize(path)
        if mol is None:
            raise ValueError("RDKit failed to read ligand: %s" % path)
        if mol.GetNumConformers() == 0:
            raise ValueError("Ligand has no 3D conformer: %s" % path)
        conf = mol.GetConformer()
        element = []
        pos = []
        for atom in mol.GetAtoms():
            element.append(atom.GetAtomicNum())
            p = conf.GetAtomPosition(atom.GetIdx())
            pos.append([p.x, p.y, p.z])
        return {
            "element": element,
            "pos": pos,
        }


def main(args):
    with open(args.protein, "r") as f:
        protein_block = f.read()
    protein = PDBProtein(protein_block)
    ligand = parse_ligand_positions(args.ligand)
    pdb_pocket_block = protein.residues_to_pdb_block(
        protein.query_residues_ligand(ligand, args.radius)
    )
    with open(args.pocket, "w") as f:
        f.write(pdb_pocket_block)
    print("Saved pocket:", args.pocket)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein", type=str, required=True)
    parser.add_argument("--ligand", type=str, required=True)
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--pocket", type=str, required=True)
    args = parser.parse_args()
    main(args)
