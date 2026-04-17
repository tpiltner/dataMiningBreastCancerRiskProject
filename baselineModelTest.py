import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, Callable, List

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

from datasetModel import CurrentOnlyDataset, RISK_COLS
from modelArchitecture import BaselineCurrentOnlyModel


DEFAULT_EXP_ROOT = Path("/local/scratch/tpiltne/models/baselineModel")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 123
CI_ALPHA = 0.05
MIN_VALID_BOOT = 50

TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 12


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def safe_auc_auprc(y: np.ndarray, s: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    y = y.astype(int)
    if (y == 1).sum() > 0 and (y == 0).sum() > 0:
        return float(roc_auc_score(y, s)), float(average_precision_score(y, s))
    return None, None


def bootstrap_metric_samples(
    y: np.ndarray,
    s: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], Optional[float]],
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    groups: Optional[np.ndarray] = None,
) -> np.ndarray:
    y = np.asarray(y)
    s = np.asarray(s)
    rng = np.random.default_rng(seed)
    samples: List[float] = []

    if len(y) < 2:
        return np.asarray(samples, dtype=float)

    if groups is None:
        n = len(y)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            val = metric_fn(y[idx], s[idx])
            if val is not None:
                samples.append(float(val))
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        if len(uniq) < 2:
            return np.asarray(samples, dtype=float)

        group_to_idx = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(n_boot):
            sampled_groups = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([group_to_idx[g] for g in sampled_groups], axis=0)
            val = metric_fn(y[idx], s[idx])
            if val is not None:
                samples.append(float(val))

    return np.asarray(samples, dtype=float)


def ci_from_samples(samples: np.ndarray, alpha: float = CI_ALPHA) -> Tuple[Optional[float], Optional[float], int]:
    n_valid = int(len(samples))
    if n_valid < MIN_VALID_BOOT:
        return None, None, n_valid
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi, n_valid


def metric_summary(
    y: np.ndarray,
    s: np.ndarray,
    label: str,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    auc_point, ap_point = safe_auc_auprc(y, s)

    if auc_point is not None:
        auc_samples = bootstrap_metric_samples(
            y, s, lambda yy, ss: safe_auc_auprc(yy.astype(int), ss)[0], groups=groups
        )
        auc_lo, auc_hi, auc_n = ci_from_samples(auc_samples)
        auc_ci = {"low": auc_lo, "high": auc_hi, "n_valid_boot": auc_n}
    else:
        auc_ci = None
        auc_samples = np.array([], dtype=float)

    if ap_point is not None:
        ap_samples = bootstrap_metric_samples(
            y, s, lambda yy, ss: safe_auc_auprc(yy.astype(int), ss)[1], groups=groups
        )
        ap_lo, ap_hi, ap_n = ci_from_samples(ap_samples)
        ap_ci = {"low": ap_lo, "high": ap_hi, "n_valid_boot": ap_n}
    else:
        ap_ci = None
        ap_samples = np.array([], dtype=float)

    return {
        "label": label,
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "auc": auc_point,
        "auc_ci95": auc_ci,
        "auc_samples": auc_samples,
        "auprc": ap_point,
        "auprc_ci95": ap_ci,
        "auprc_samples": ap_samples,
    }


def remap_legacy_cum_keys_for_baseline(state_dict: dict) -> dict:
    """
    Legacy checkpoint key shim:
      cum.base_hazard_fc -> cum.base
      cum.hazard_fc      -> cum.inc
    """
    sd = dict(state_dict)
    has_legacy = any(
        k.startswith("cum.hazard_fc.") or k.startswith("cum.base_hazard_fc.")
        for k in sd.keys()
    )
    if not has_legacy:
        return sd

    if "cum.base_hazard_fc.weight" in sd:
        sd["cum.base.weight"] = sd.pop("cum.base_hazard_fc.weight")
    if "cum.base_hazard_fc.bias" in sd:
        sd["cum.base.bias"] = sd.pop("cum.base_hazard_fc.bias")
    if "cum.hazard_fc.weight" in sd:
        sd["cum.inc.weight"] = sd.pop("cum.hazard_fc.weight")
    if "cum.hazard_fc.bias" in sd:
        sd["cum.inc.bias"] = sd.pop("cum.hazard_fc.bias")

    sd.pop("cum.upper_triangular_mask", None)
    return sd


@torch.inference_mode()
def collect_logits_labels_masks(
    model: torch.nn.Module,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    model.eval()

    all_logits = []
    all_labels = []
    all_masks = []
    all_groups = []

    for batch in loader:
        if len(batch) < 5:
            raise ValueError("Batch must have at least 5 elements: imgs, delta_feat, has_prior_views, y, m")

        imgs, delta_feat, has_prior_views, y, m = batch[:5]
        group_ids = batch[5] if len(batch) >= 6 else None

        imgs = imgs.to(DEVICE, dtype=torch.float32)
        delta_feat = delta_feat.to(DEVICE, dtype=torch.float32)
        has_prior_views = has_prior_views.to(DEVICE, dtype=torch.float32)
        y = y.to(DEVICE, dtype=torch.float32)
        m = m.to(DEVICE, dtype=torch.float32)

        out = model(imgs, delta_feat, has_prior_views)
        logits = out["risk_prediction"]["pred_fused"]

        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())
        all_masks.append(m.detach().cpu().numpy())

        if group_ids is not None:
            if torch.is_tensor(group_ids):
                group_ids = group_ids.detach().cpu().numpy()
            all_groups.append(np.asarray(group_ids))

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    groups = None
    if len(all_groups) > 0:
        groups = np.concatenate(all_groups, axis=0)
        if groups.shape[0] != logits.shape[0]:
            print("[WARN] group_ids shape mismatch; ignoring group bootstrap.")
            groups = None

    return logits, labels, masks, groups


def style_axes():
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.grid(True, alpha=0.25)


def plot_roc_all_horizons(labels, probs, masks, out_path: Path, title: str = "ROC Plot"):
    plt.figure(figsize=(11, 8.5))
    plt.plot([0, 1], [0, 1], "k--", linewidth=2)

    for t, name in enumerate(RISK_COLS):
        keep = masks[:, t] > 0.5
        y = labels[keep, t].astype(int)
        p = probs[keep, t].astype(float)

        if (y == 1).sum() == 0 or (y == 0).sum() == 0:
            continue

        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        plt.step(fpr, tpr, where="post", linewidth=2.5, label=f"{name} (AUC={auc:.3f})")

    plt.xlabel("False Positive Rate", fontsize=LABEL_FONTSIZE)
    plt.ylabel("True Positive Rate", fontsize=LABEL_FONTSIZE)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    style_axes()
    plt.legend(loc="lower right", fontsize=LEGEND_FONTSIZE, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_pr_all_horizons(labels, probs, masks, out_path: Path, title: str = "PR Plot"):
    plt.figure(figsize=(11, 8.5))

    for t, name in enumerate(RISK_COLS):
        keep = masks[:, t] > 0.5
        y = labels[keep, t].astype(int)
        p = probs[keep, t].astype(float)

        if y.size == 0 or (y == 1).sum() == 0 or (y == 0).sum() == 0:
            continue

        precision, recall, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)

        (line,) = plt.step(
            recall,
            precision,
            where="post",
            linewidth=2.5,
            label=f"{name} (AP={ap:.3f})",
        )

        prevalence = float((y == 1).sum() / len(y))
        plt.hlines(
            y=prevalence,
            xmin=0.0,
            xmax=1.0,
            colors=line.get_color(),
            linestyles="dashed",
            linewidth=2.0,
            label=f"{name} baseline (prev={prevalence:.3f})",
        )

    plt.xlabel("Recall", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Precision", fontsize=LABEL_FONTSIZE)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    style_axes()
    plt.legend(loc="upper right", fontsize=LEGEND_FONTSIZE, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def resolve_exp_dir(exp_dir: Optional[Path]) -> Path:
    if exp_dir is None:
        exp_dir = DEFAULT_EXP_ROOT
    exp_dir = exp_dir.expanduser().resolve()

    if exp_dir.is_dir():
        if (exp_dir / "final_best").exists() or (exp_dir / "best_config.json").exists():
            return exp_dir

    if exp_dir.name.startswith("baseline_grid_then_final_"):
        return exp_dir

    if not exp_dir.exists():
        raise FileNotFoundError(f"exp_dir does not exist: {exp_dir}")

    runs = sorted(exp_dir.glob("baseline_grid_then_final_*"), key=lambda p: p.name)
    if not runs:
        raise FileNotFoundError(f"No baseline_grid_then_final_* runs found under: {exp_dir}")
    return runs[-1]


def load_best_checkpoint_and_cfg(exp_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    ckpt_path = exp_dir / "final_best" / "best_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    run_cfg_path = exp_dir / "final_best" / "run_config.json"
    if run_cfg_path.exists():
        cfg = json.loads(run_cfg_path.read_text())
    else:
        best_cfg_path = exp_dir / "best_config.json"
        cfg = json.loads(best_cfg_path.read_text()) if best_cfg_path.exists() else {}

    return ckpt_path, cfg


def build_model_from_cfg(cfg: Dict[str, Any]) -> torch.nn.Module:
    return BaselineCurrentOnlyModel(
        pretrained_encoder=True,
        num_years=len(RISK_COLS),
        dim=int(cfg.get("dim", 512)),
        mlp_layers=int(cfg.get("num_layers", 1)),
        mlp_hidden=int(cfg.get("hidden_units", 256)),
        dropout=float(cfg.get("dropout", 0.0)),
        freeze_encoder=True,
    )


def make_loader(dataset: torch.utils.data.Dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default=None)
    args = parser.parse_args()

    exp_dir = resolve_exp_dir(Path(args.exp_dir) if args.exp_dir else None)
    ckpt_path, cfg = load_best_checkpoint_and_cfg(exp_dir)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = exp_dir / f"baseline_results_{run_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] EXP_DIR:", exp_dir, flush=True)
    print("[INFO] OUTPUT_DIR:", output_dir, flush=True)
    print("[INFO] CHECKPOINT:", ckpt_path, flush=True)

    test_ds = CurrentOnlyDataset(split="test")
    test_loader = make_loader(test_ds)

    print("[INFO] TEST size:", len(test_ds), flush=True)

    model = build_model_from_cfg(cfg).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
    state_dict = remap_legacy_cum_keys_for_baseline(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("[INFO] missing keys   :", len(missing), flush=True)
    print("[INFO] unexpected keys:", len(unexpected), flush=True)

    test_logits, test_labels, test_masks, test_groups = collect_logits_labels_masks(model, test_loader)

    if args.use_all_masks:
        test_masks = np.ones_like(test_labels, dtype=np.float32)

    test_probs = sigmoid_np(test_logits).astype(np.float64)

    roc_path = output_dir / "roc_plot.png"
    pr_path = output_dir / "pr_plot.png"

    plot_roc_all_horizons(test_labels, test_probs, test_masks, roc_path, title="ROC Plot")
    plot_pr_all_horizons(test_labels, test_probs, test_masks, pr_path, title="PR Plot")

    horizon_metrics = {}
    for t, h in enumerate(RISK_COLS):
        keep = test_masks[:, t] > 0.5
        y = test_labels[keep, t].astype(int)
        p = test_probs[keep, t].astype(float)
        g = test_groups[keep] if test_groups is not None else None

        horizon_metrics[h] = {
            k: v for k, v in metric_summary(y, p, h, groups=g).items()
            if not k.endswith("_samples")
        }

    results = {
        "run_tag": run_tag,
        "exp_dir": str(exp_dir),
        "checkpoint_used": str(ckpt_path),
        "n_test_total": int(len(test_ds)),
        "probabilities": "raw sigmoid(logits)",
        "per_horizon_metrics_with_ci": horizon_metrics,
        "paths": {
            "roc": str(roc_path),
            "pr": str(pr_path),
        },
    }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))

    print("\n[INFO] Saved outputs to:", output_dir, flush=True)
    print("[INFO] results json:", results_path, flush=True)


if __name__ == "__main__":
    main()