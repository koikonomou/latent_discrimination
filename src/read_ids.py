import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

CSV_PATH = "colour_data/colours_labels.csv" 
SUBSET_TXT = "/home/katerina/codes/latent_discrimination/runs/20251027-141209_dataset=custom_method=kcenter_cosine_seed=42/subsets/keep_idx_50.txt"  # π.χ. keep_idx_50.txt
VAL_SPLIT = 0.2
SEED = 42
OUT_DIR = Path("runs/_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# --------------------------------------


df = pd.read_csv(CSV_PATH)

required = {"fname","n","label"}
missing = required - set(df.columns.str.lower())
if missing:
    raise SystemExit(f"CSV must contain columns {required}; missing: {missing}")

def norm_label(v):
    if isinstance(v, str):
        s = v.strip().lower()
        if s == "true": return 1
        if s == "false": return 0
    try:
        return int(v)
    except:
        raise ValueError(f"Unrecognized label: {v!r}")

labels = df["label"].apply(norm_label).astype(int).to_numpy()

idx_all = np.arange(len(df))
idx_tr, idx_te = train_test_split(
    idx_all, test_size=VAL_SPLIT, random_state=SEED, stratify=labels
)

keep_train_idx = np.loadtxt(SUBSET_TXT, dtype=int)
if keep_train_idx.ndim == 0:
    keep_train_idx = keep_train_idx[None]  

orig_idx = idx_tr[keep_train_idx]


df_sub = df.iloc[orig_idx].copy()


counts = df_sub["n"].value_counts().sort_index()
perc = (counts / counts.sum() * 100.0).round(2)

# print("\nDistilled subset size:", len(df_sub))
# print("\nValue counts of n:")
# print(counts.to_string())
# print("\nPercentages (%):")
# print(perc.to_string())

out_counts = OUT_DIR / (Path(SUBSET_TXT).stem + "_n_counts.csv")
out_rows   = OUT_DIR / (Path(SUBSET_TXT).stem + "_rows.csv")
counts.to_csv(out_counts, header=["count"])
df_sub.to_csv(out_rows, index=False)
print(f"\n[OK] Saved:\n - {out_counts}\n - {out_rows}")


n_vals = pd.to_numeric(df_sub["n"], errors="coerce")
mask = n_vals.notna()
n_vals = n_vals[mask]
df_b = df_sub.loc[mask].copy()

NBINS = 20
edges = np.linspace(n_vals.min(), n_vals.max(), NBINS + 1)

cats = pd.cut(n_vals, bins=edges, include_lowest=True, right=False)

bin_counts = cats.value_counts().sort_index()
bin_perc = (bin_counts / bin_counts.sum() * 100.0).round(2)

out_counts_bins = OUT_DIR / (Path(SUBSET_TXT).stem + "_n_binned_counts.csv")
out_perc_bins   = OUT_DIR / (Path(SUBSET_TXT).stem + "_n_binned_percent.csv")
bin_counts.to_csv(out_counts_bins, header=["count"])
bin_perc.to_csv(out_perc_bins, header=["percent"])

def bin_labels(idx):
    return [f"[{iv.left:.2f}, {iv.right:.2f})" for iv in idx]

plt.figure(figsize=(12,4))
bin_counts.index = bin_labels(bin_counts.index)
bin_counts.plot(kind="bar")
plt.title(f"{NBINS} bins")
plt.xlabel("bin(n)")
plt.ylabel("")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / (Path(SUBSET_TXT).stem + f"_n_binned_{NBINS}.png"), dpi=150)
plt.close()

plt.figure(figsize=(12,4))
bin_perc.index = bin_labels(bin_perc.index)
bin_perc.plot(kind="bar")
plt.title(f"{NBINS} bins (%)")
plt.xlabel("bin(n)")
plt.ylabel("(%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / (Path(SUBSET_TXT).stem + f"_n_binned_percent_{NBINS}.png"), dpi=150)
plt.close()

if "label" in df_b.columns:
    df_b["label_int"] = df_b["label"].apply(norm_label).astype(int)
    ct = pd.crosstab(cats, df_b["label_int"]).sort_index()
    ct.index = bin_labels(ct.index)

    plt.figure(figsize=(13,5))
    ct.plot(kind="bar", ax=plt.gca())
    plt.title(f"Κατανομή n σε {NBINS} bins ανά label")
    plt.xlabel("bin(n)")
    plt.ylabel("Πλήθος")
    plt.legend(title="label", labels=[0,1])
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / (Path(SUBSET_TXT).stem + f"_n_binned_by_label_{NBINS}.png"), dpi=150)
    plt.close()

print("[OK] ", OUT_DIR)
