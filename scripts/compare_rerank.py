import argparse
import csv
import glob
import os


def find_sdf_dir(run_dir):
    candidates = sorted(glob.glob(os.path.join(run_dir, "*_SDF")))
    if not candidates:
        raise FileNotFoundError("No *_SDF directory found under %s" % run_dir)
    if len(candidates) > 1:
        raise ValueError("Multiple *_SDF directories found: %s" % ", ".join(candidates))
    return candidates[0]


def to_float(value):
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def read_author_log(sdf_dir):
    path = os.path.join(sdf_dir, "log.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows = {}
    with open(path, "r", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]
    for idx, line in enumerate(lines[1:], 1):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        mol_id, smiles, sa, qed, vina, score = parts[:6]
        rows[mol_id] = {
            "mol_id": mol_id,
            "author_rank": idx,
            "author_smiles": smiles,
            "author_sa": sa,
            "author_qed": qed,
            "author_vina": vina,
            "author_score": score,
        }
    return rows


def read_frequency_rerank(path):
    rows = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[str(row["mol_id"])] = row
    return rows


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def summarize(rows, rank_key, k):
    top = sorted(rows, key=lambda r: int(r[rank_key]))[:k]
    return {
        "k": len(top),
        "mean_author_vina": mean([to_float(r.get("author_vina")) for r in top]),
        "mean_freq_similarity": mean([to_float(r.get("freq_similarity")) for r in top]),
        "mean_qed": mean([to_float(r.get("qed")) or to_float(r.get("author_qed")) for r in top]),
        "mean_sa": mean([to_float(r.get("sa")) or to_float(r.get("author_sa")) for r in top]),
    }


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "%.6f" % value
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--frequency_csv", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    sdf_dir = find_sdf_dir(args.run_dir)
    frequency_csv = args.frequency_csv or os.path.join(args.run_dir, "frequency_rerank.csv")
    out = args.out or os.path.join(args.run_dir, "rank_comparison.csv")

    author = read_author_log(sdf_dir)
    freq = read_frequency_rerank(frequency_csv)

    rows = []
    for mol_id, author_row in author.items():
        row = dict(author_row)
        row.update(freq.get(mol_id, {}))
        if "rank" in row and row["rank"] not in ("", None):
            row["frequency_rank"] = int(row["rank"])
            row["rank_shift"] = int(row["author_rank"]) - int(row["frequency_rank"])
        else:
            row["frequency_rank"] = ""
            row["rank_shift"] = ""
        rows.append(row)

    rows.sort(key=lambda r: int(r["frequency_rank"]) if r["frequency_rank"] != "" else 10**9)

    fieldnames = [
        "mol_id",
        "author_rank",
        "frequency_rank",
        "rank_shift",
        "author_score",
        "author_vina",
        "freq_similarity",
        "final_score",
        "qed",
        "sa",
        "logp",
        "tpsa",
        "smiles",
        "author_smiles",
    ]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", out)
    print("\nTop by author log.txt score:")
    author_summary = summarize(rows, "author_rank", min(args.topk, len(rows)))
    for key, value in author_summary.items():
        print("%s: %s" % (key, fmt(value)))

    print("\nTop by frequency rerank:")
    freq_summary = summarize(rows, "frequency_rank", min(args.topk, len(rows)))
    for key, value in freq_summary.items():
        print("%s: %s" % (key, fmt(value)))

    print("\nRank changes, positive means frequency rerank moved it earlier:")
    for row in rows:
        print(
            "mol=%s author=%s freq=%s shift=%s vina=%s freq_sim=%s"
            % (
                row.get("mol_id", ""),
                row.get("author_rank", ""),
                row.get("frequency_rank", ""),
                row.get("rank_shift", ""),
                row.get("author_vina", ""),
                row.get("freq_similarity", ""),
            )
        )


if __name__ == "__main__":
    main()
