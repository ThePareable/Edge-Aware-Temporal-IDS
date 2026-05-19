import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# =========================
# PATHS
# =========================

BASE = Path("C:\\Users\\bugramuhci\\Desktop\\tez\\gorsellik\\outputs")

PACKS = {
    "TGAT": BASE / "tgat_test.pkl",
    "GCLSTM": BASE / "gclstm_test.pkl",
    "HYBRID": BASE / "hybrid_test.pkl",
}

OUT = BASE / "visualizations_final"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "ARP-SPOOF",
    "BENIGN",
    "DDOS-ICMP",
    "EXPLOITING-FTP",
    "FTP-BRUTE-FORCE",
    "PORT-SCANNING",
    "SQL-INJECTION",
    "SSH-BRUTE-FORCE",
    "SYN-FLOOD",
    "XSS",
]

THRESHOLD = 0.5


# =========================
# HELPERS
# =========================

def load_pack(path):
    with open(path, "rb") as f:
        d = pickle.load(f)

    probs = np.asarray(d["probs"], dtype=np.float32)

    if "y" in d:
        y = np.asarray(d["y"], dtype=np.float32)
    elif "labels" in d:
        y = np.asarray(d["labels"], dtype=np.float32)
    else:
        raise KeyError(f"No labels/y in {path}")

    classes = d.get("classes", CLASSES)

    return {
        "probs": probs,
        "y": y,
        "pred": (probs >= THRESHOLD).astype(np.int32),
        "classes": classes,
        "ids": d.get("ids", None),
    }


def metrics(y, pred):
    return {
        "micro_f1": float(f1_score(y.reshape(-1), pred.reshape(-1), zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y.reshape(-1), pred.reshape(-1), zero_division=0)),
        "micro_recall": float(recall_score(y.reshape(-1), pred.reshape(-1), zero_division=0)),
        "exact_match": float((y == pred).all(axis=1).mean()),
    }


def per_class_metrics(y, pred, classes):
    rows = []
    for i, c in enumerate(classes):
        rows.append({
            "class": c,
            "support": int(y[:, i].sum()),
            "precision": float(precision_score(y[:, i], pred[:, i], zero_division=0)),
            "recall": float(recall_score(y[:, i], pred[:, i], zero_division=0)),
            "f1": float(f1_score(y[:, i], pred[:, i], zero_division=0)),
        })
    return rows


# =========================
# LOAD
# =========================

data = {}
summary = {}
perclass = {}

for name, path in PACKS.items():
    if not path.exists():
        print(f"[WARN] missing: {path}")
        continue

    d = load_pack(path)
    data[name] = d
    summary[name] = metrics(d["y"], d["pred"])
    perclass[name] = per_class_metrics(d["y"], d["pred"], d["classes"])

print(json.dumps(summary, indent=2))

with open(OUT / "summary_metrics.json", "w") as f:
    json.dump(summary, f, indent=2)

with open(OUT / "perclass_metrics.json", "w") as f:
    json.dump(perclass, f, indent=2)


# =========================
# 1) OVERALL METRICS BAR
# =========================

metric_names = ["micro_f1", "macro_f1", "exact_match", "micro_precision", "micro_recall"]
model_names = list(summary.keys())

x = np.arange(len(metric_names))
width = 0.8 / max(len(model_names), 1)

plt.figure(figsize=(12, 6))
for idx, model in enumerate(model_names):
    vals = [summary[model][m] for m in metric_names]
    plt.bar(x + idx * width, vals, width, label=model)

plt.xticks(x + width * (len(model_names) - 1) / 2, metric_names, rotation=20)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Overall Model Comparison: TGAT vs GCLSTM vs HYBRID")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "overall_metrics_comparison.png", dpi=200)
plt.close()


# =========================
# 2) PER-CLASS F1 COMPARISON
# =========================

x = np.arange(len(CLASSES))
width = 0.8 / max(len(model_names), 1)

plt.figure(figsize=(16, 7))
for idx, model in enumerate(model_names):
    f1s = [r["f1"] for r in perclass[model]]
    plt.bar(x + idx * width, f1s, width, label=model)

plt.xticks(x + width * (len(model_names) - 1) / 2, CLASSES, rotation=45, ha="right")
plt.ylim(0, 1.05)
plt.ylabel("F1-score")
plt.title("Per-Class F1 Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "perclass_f1_comparison.png", dpi=200)
plt.close()


# =========================
# 3) CLASS SUPPORT
# =========================

# support aynı label setinden geldiği için HYBRID varsa onu baz alıyoruz
ref_model = "HYBRID" if "HYBRID" in data else model_names[0]
support = data[ref_model]["y"].sum(axis=0)

plt.figure(figsize=(14, 6))
plt.bar(CLASSES, support)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Number of positive samples")
plt.title(f"Class Support in Test Set ({ref_model})")
plt.tight_layout()
plt.savefig(OUT / "class_support.png", dpi=200)
plt.close()


# =========================
# 4) PROBABILITY DISTRIBUTIONS
# =========================

for model, d in data.items():
    probs = d["probs"]

    plt.figure(figsize=(12, 6))
    for i, cls in enumerate(CLASSES):
        plt.hist(probs[:, i], bins=30, alpha=0.35, label=cls)

    plt.xlabel("Predicted probability")
    plt.ylabel("Frequency")
    plt.title(f"Probability Distribution per Class - {model}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / f"probability_distribution_{model}.png", dpi=200)
    plt.close()


# =========================
# 5) MULTILABEL ERROR HEATMAP
# TP FP FN TN per class
# =========================

for model, d in data.items():
    y = d["y"]
    p = d["pred"]

    tp = ((y == 1) & (p == 1)).sum(axis=0)
    fp = ((y == 0) & (p == 1)).sum(axis=0)
    fn = ((y == 1) & (p == 0)).sum(axis=0)
    tn = ((y == 0) & (p == 0)).sum(axis=0)

    mat = np.vstack([tp, fp, fn, tn])

    plt.figure(figsize=(14, 5))
    plt.imshow(mat, aspect="auto")
    plt.yticks([0, 1, 2, 3], ["TP", "FP", "FN", "TN"])
    plt.xticks(np.arange(len(CLASSES)), CLASSES, rotation=45, ha="right")
    plt.colorbar(label="Count")
    plt.title(f"Multilabel Error Breakdown - {model}")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, str(int(mat[i, j])), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT / f"error_heatmap_{model}.png", dpi=200)
    plt.close()


# =========================
# 6) CLASSIFICATION REPORT TXT
# =========================

with open(OUT / "classification_reports.txt", "w") as f:
    for model, d in data.items():
        f.write("=" * 80 + "\n")
        f.write(model + "\n")
        f.write("=" * 80 + "\n")
        f.write(classification_report(
            d["y"],
            d["pred"],
            target_names=CLASSES,
            zero_division=0
        ))
        f.write("\n\n")


print(f"\n✓ Visualizations written to: {OUT}")
print("Generated:")
for p in sorted(OUT.glob("*")):
    print(" -", p)