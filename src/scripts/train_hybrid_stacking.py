import argparse, json, pickle, re
from pathlib import Path

import numpy as np
import torch


def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def norm_id(x) -> str:
    """
    Hybrid join key.
    - TGAT ids: "w1.0_t1457" -> "1457"
    - GCLSTM ids: "t1457" -> "1457"
    - fallback: first number group -> "1457"
    """
    s = str(x)
    m = re.search(r"_t(\d+)", s)
    if m:
        return m.group(1)
    m = re.search(r"t(\d+)", s)
    if m:
        return m.group(1)
    m = re.search(r"(\d+)", s)
    if m:
        return m.group(1)
    return s


def first_present(pack: dict, keys):
    for k in keys:
        if k in pack and pack[k] is not None:
            return pack[k]
    return None


def extract_pack(pack: dict):
    """
    Returns:
      ids_raw(list[str]), ids_norm(list[str]), probs(np.float32 [N,C]), y(np.float32 hard {0,1} [N,C]), classes(list[str] or None)
    """
    ids = first_present(pack, ["ids", "graph_ids", "sample_ids"])
    probs = first_present(pack, ["probs", "p", "y_prob", "y_probs", "proba"])
    y = first_present(pack, ["labels", "y_true", "y", "targets"])
    classes = first_present(pack, ["classes"]) or (pack.get("meta", {}) or {}).get("classes")

    if ids is None or probs is None or y is None:
        raise KeyError(f"Pack missing keys. have={list(pack.keys())}")

    ids_raw = list(ids)


    # --- collapse duplicate ids (by NORM_ID) (mean probs; OR labels) ---
    # Align norm_id ile yapıldığı için, duplicate'leri norm_id üzerinden tekilleştiriyoruz.
    probs_arr = np.asarray(probs, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    if probs_arr.ndim != 2 or y_arr.ndim != 2:
        raise RuntimeError(f"Expected probs/y as 2D arrays. got probs{probs_arr.shape} y{y_arr.shape}")

    ids_norm_full = [norm_id(rid) for rid in ids_raw]

    seen = {}          # norm_id -> out index
    out_ids_raw = []   # representative raw id (first seen)
    out_ids_norm = []
    sum_probs = []
    or_y = []
    cnt = []

    for i, (rid, nid) in enumerate(zip(ids_raw, ids_norm_full)):
        if nid not in seen:
            seen[nid] = len(out_ids_norm)
            out_ids_raw.append(rid)
            out_ids_norm.append(nid)
            sum_probs.append(probs_arr[i].copy())
            or_y.append(y_arr[i].copy())
            cnt.append(1)
        else:
            j = seen[nid]
            sum_probs[j] += probs_arr[i]
            or_y[j] = np.maximum(or_y[j], y_arr[i])  # OR for hard multilabel
            cnt[j] += 1

    probs = (np.stack(sum_probs, axis=0) / np.asarray(cnt, dtype=np.float32)[:, None]).astype(np.float32)
    y = np.stack(or_y, axis=0).astype(np.float32)
    ids_raw = out_ids_raw
    ids_norm = out_ids_norm
    # --- end collapse duplicate ids ---
    ids_norm = [norm_id(z) for z in ids_raw]

    probs = np.asarray(probs, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    y = (y >= 0.5).astype(np.float32)

    if probs.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected probs/y 2D. probs={probs.shape} y={y.shape}")
    if probs.shape != y.shape:
        raise ValueError(f"probs/y shape mismatch: probs={probs.shape} y={y.shape}")

    return ids_raw, ids_norm, probs, y, classes


def align_two(ids1_norm, p1, y1, ids2_norm, p2, y2):
    """
    Aligns by intersection on normalized ids.
    Returns (common_ids, X, y) where:
      X = concat([p1, p2]) -> [N, 2C]
      y = y1
    """
    idx1 = {}
    for i, k in enumerate(ids1_norm):
        # if duplicates exist, keep first (or you can keep last); we just need deterministic
        if k not in idx1:
            idx1[k] = i
    idx2 = {}
    for i, k in enumerate(ids2_norm):
        if k not in idx2:
            idx2[k] = i

    common = sorted(set(idx1.keys()).intersection(set(idx2.keys())))
    if len(common) == 0:
        raise RuntimeError("No common ids after normalization. Check norm_id()")

    i1 = np.array([idx1[k] for k in common], dtype=np.int64)
    i2 = np.array([idx2[k] for k in common], dtype=np.int64)

    yA = y1[i1]
    yB = y2[i2]

    # Harden and compare (strict equality)
    yA = (yA >= 0.5).astype(np.float32)
    yB = (yB >= 0.5).astype(np.float32)

    # If still mismatch, warn but continue with yA (this can happen if packs are produced from different filtering)
    mismatch = np.mean(np.abs(yA - yB))
    if mismatch > 0:
        print(f"[WARN] label mismatch on common ids: mean abs diff={mismatch:.6f} (continuing with y from first pack)")

    X = np.concatenate([p1[i1], p2[i2]], axis=1).astype(np.float32)
    X = np.clip(X, 1e-6, 1.0 - 1e-6)
    X = np.log(X/(1.0 - X)).astype(np.float32)
    return common, X, yA


def f1_micro_macro(y_true, y_pred):
    eps = 1e-9
    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1 - y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1 - y_pred)).sum(axis=0)

    f1_c = (2 * tp) / (2 * tp + fp + fn + eps)
    f1_macro = float(np.mean(f1_c))

    tp_m = float(tp.sum())
    fp_m = float(fp.sum())
    fn_m = float(fn.sum())
    f1_micro = float((2 * tp_m) / (2 * tp_m + fp_m + fn_m + eps))

    exact = float(np.mean(np.all(y_true == y_pred, axis=1)))
    return f1_micro, f1_macro, exact, f1_c.astype(np.float32)


def tune_thresholds(probs, y_true, step=0.02):
    C = y_true.shape[1]
    thr = np.full((C,), 0.5, dtype=np.float32)
    for c in range(C):
        best = -1.0
        best_t = 0.5
        for t in np.arange(0.05, 0.951, step):
            y_pred = (probs[:, c] >= t).astype(np.float32)
            yt = y_true[:, c]
            # per-class F1
            tp = float(np.sum(yt * y_pred))
            fp = float(np.sum((1 - yt) * y_pred))
            fn = float(np.sum(yt * (1 - y_pred)))
            f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)
            if f1 > best:
                best = f1
                best_t = float(t)
        thr[c] = best_t
    return thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tgat-val", required=True)
    ap.add_argument("--gclstm-val", required=True)
    ap.add_argument("--tgat-test", required=True)
    ap.add_argument("--gclstm-test", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--tune-step", type=float, default=0.02)
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))

    # load
    t_val = load_pkl(args.tgat_val)
    g_val = load_pkl(args.gclstm_val)
    t_test = load_pkl(args.tgat_test)
    g_test = load_pkl(args.gclstm_test)

    # extract
    _, tvid, tvp, tvy, tcls = extract_pack(t_val)
    _, gvid, gvp, gvy, gcls = extract_pack(g_val)
    _, ttid, ttp, tty, tcls2 = extract_pack(t_test)
    _, gtid, gtp, gty, gcls2 = extract_pack(g_test)

    classes = tcls or tcls2 or gcls or gcls2
    if classes is not None and (gcls is not None) and (list(classes) != list(gcls)):
        print("[WARN] classes differ between packs. Using TGAT classes ordering.")

    # align by normalized id
    ids_v, Xv, yv = align_two(tvid, tvp, tvy, gvid, gvp, gvy)
    common_val = ids_v
    ids_t, Xt, yt = align_two(ttid, ttp, tty, gtid, gtp, gty)
    common_te = ids_t

    print(f"[hybrid] val: tgat={len(tvid)} gclstm={len(gvid)} common={len(common_val)} X={Xv.shape}")
    print(f"[hybrid] test: tgat={len(ttid)} gclstm={len(gtid)} common={len(common_te)} X={Xt.shape}")

    # model: simple stacking = linear layer -> logits for each class
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = yv.shape[1]
    D = Xv.shape[1]
    model = torch.nn.Linear(D, C).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = torch.nn.BCEWithLogitsLoss()

    Xv_t = torch.from_numpy(Xv).to(device)
    yv_t = torch.from_numpy(yv).to(device)
    Xt_t = torch.from_numpy(Xt).to(device)
    yt_t = torch.from_numpy(yt).to(device)

    best_macro = -1.0
    best_state = None

    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(Xv_t)
        loss = crit(logits, yv_t)
        loss.backward()
        opt.step()

        if ep % 10 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(Xv_t)).detach().cpu().numpy()
                thr = tune_thresholds(pv, yv, step=args.tune_step)
                yhat = (pv >= thr[None, :]).astype(np.float32)
                f1_micro, f1_macro, exact, _ = f1_micro_macro(yv, yhat)
            if f1_macro > best_macro:
                best_macro = f1_macro
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"epoch={ep:03d} loss={float(loss.item()):.4f} val micro={f1_micro:.4f} macro={f1_macro:.4f} exact={exact:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # final eval
    model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(model(Xv_t)).detach().cpu().numpy()
        pt = torch.sigmoid(model(Xt_t)).detach().cpu().numpy()

    thr = tune_thresholds(pv, yv, step=args.tune_step)
    yhat_v = (pv >= thr[None, :]).astype(np.float32)
    yhat_t = (pt >= thr[None, :]).astype(np.float32)

    v_micro, v_macro, v_exact, _ = f1_micro_macro(yv, yhat_v)
    t_micro, t_macro, t_exact, _ = f1_micro_macro(yt, yhat_t)

    out = {
        "val": {"f1_micro": v_micro, "f1_macro": v_macro, "exact": v_exact, "n": int(len(common_val))},
        "test": {"f1_micro": t_micro, "f1_macro": t_macro, "exact": t_exact, "n": int(len(common_te))},
        "thresholds": thr.astype(float).tolist(),
        "classes": list(classes) if classes is not None else None,
        "dims": {"D": int(D), "C": int(C)},
    }

    (out_dir / "hybrid_metrics.json").write_text(json.dumps(out, indent=2))

    # --- export hybrid probs (val/test) for per-class eval & deployment ---
    import pickle
    def _dump(name, ids, probs, y, classes, thresholds):
        out = {
            "ids": ids,
            "probs": probs.astype("float32"),
            "y": y.astype("float32"),
            "classes": classes,
            "thresholds": thresholds.astype("float32"),
        }
        with open(Path(args.out_dir) / name, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        _dump("hybrid_val.pkl", ids_v, pv, yv, classes, thr)
        _dump("hybrid_test.pkl", ids_t, pt, yt, classes, thr)
        print("✓ wrote:", Path(args.out_dir) / "hybrid_val.pkl")
        print("✓ wrote:", Path(args.out_dir) / "hybrid_test.pkl")
    except Exception as e:
        print("[warn] hybrid probs export failed:", e)
    torch.save(model.state_dict(), out_dir / "hybrid_stacker.pt")

    print("✓ wrote:", out_dir / "hybrid_metrics.json")
    print("val:", out["val"])
    print("test:", out["test"])


if __name__ == "__main__":
    main()