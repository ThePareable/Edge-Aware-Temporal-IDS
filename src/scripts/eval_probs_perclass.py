import argparse, json, pickle
from pathlib import Path
import numpy as np

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def f1_stats(y_true, y_pred, eps=1e-9):
    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1-y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1-y_pred)).sum(axis=0)
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2*prec*rec/(prec+rec+eps)
    sup  = y_true.sum(axis=0)
    return prec, rec, f1, sup

def micro_macro(y_true, y_pred, eps=1e-9):
    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1-y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1-y_pred)).sum(axis=0)
    f1_macro = (2*tp/(2*tp+fp+fn+eps)).mean()
    tpM = tp.sum(); fpM=fp.sum(); fnM=fn.sum()
    f1_micro = (2*tpM)/(2*tpM+fpM+fnM+eps)
    exact = (y_true==y_pred).all(axis=1).mean()
    return float(f1_micro), float(f1_macro), float(exact)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="pkl with keys: probs, y, classes, thresholds")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = load(args.pack)
    probs = np.asarray(d["probs"], dtype=np.float32)
    y = np.asarray(d["y"], dtype=np.float32)
    classes = d.get("classes", [f"c{i}" for i in range(probs.shape[1])])
    thr = np.asarray(d.get("thresholds", np.full((probs.shape[1],), 0.5, dtype=np.float32)), dtype=np.float32)

    pred = (probs >= thr[None,:]).astype(np.float32)

    f1_micro, f1_macro, exact = micro_macro(y, pred)
    prec, rec, f1, sup = f1_stats(y, pred)

    per = []
    for i,c in enumerate(classes):
        per.append({
            "class": c,
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": float(sup[i]),
        })

    out = {
        "metrics": {"f1_micro": f1_micro, "f1_macro": f1_macro, "exact": exact, "n": int(y.shape[0])},
        "thresholds": thr.tolist(),
        "classes": classes,
        "per_class": per,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("✓ wrote", args.out)
    print("micro", f1_micro, "macro", f1_macro, "exact", exact)

if __name__ == "__main__":
    main()