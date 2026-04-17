# datasetModel.py

import os
from pathlib import Path
from typing import Optional, Tuple
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


CSV_PATH = Path(os.environ.get(
    "EMBED_CSV_PATH",
    "/local/scratch/tpiltne/utils/trainingWithPNGS.csv"
))

HORIZONS = [1, 2, 3, 4, 5]
RISK_POS_COLS = [f"risk_{h}y_pos" for h in HORIZONS]
RISK_NEG_COLS = [f"risk_{h}y_neg" for h in HORIZONS]

EXAM_ID_COL = "acc_anon"
VIEW_ORDER = ["L-CC", "R-CC", "L-MLO", "R-MLO"]

CUR_PNG_COL = "cur_png"
LAT_COL = "ImageLateralityFinal"
VIEWPOS_COL = "ViewPosition"

FOLLOWUP_YEARS_CANDIDATES = ["followup_years_exam"]

# label semantics
LABEL_COLS_ARE_EVENT = False
AUTO_INFER_LABEL_FLIP = True
INFER_SAMPLE_ROWS = 5000

# image sizing
FIXED_HW: Optional[Tuple[int, int]] = (1664, 2048)  # (H, W)
USE_LANCZOS_RESIZE = True

# caching
ENABLE_IMAGE_CACHE = True
IMAGE_CACHE_MAX_ITEMS = 2048


# Helpers
def _to_int01(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    out = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    out[out != 0] = 1
    return out


def _resize_if_needed(img_u16: np.ndarray, fixed_hw: Optional[Tuple[int, int]]) -> np.ndarray:
    if fixed_hw is None:
        return img_u16
    Ht, Wt = fixed_hw
    if img_u16.shape[0] == Ht and img_u16.shape[1] == Wt:
        return img_u16
    pil = Image.fromarray(img_u16)
    resample = Image.Resampling.LANCZOS if USE_LANCZOS_RESIZE else Image.Resampling.BILINEAR
    pil = pil.resize((Wt, Ht), resample=resample)
    return np.array(pil)


def _normalize01(img_u16: np.ndarray) -> np.ndarray:
    return img_u16.astype(np.float32) / 65535.0


def _load_png_16bit_to_chw(
    png_path: Path,
    fixed_hw: Optional[Tuple[int, int]] = FIXED_HW
) -> np.ndarray:
    arr = np.array(Image.open(png_path))  # uint16 (H,W)

    # If image is (2048,1664) but fixed is (1664,2048), transpose first.
    if fixed_hw is not None:
        Ht, Wt = fixed_hw
        if arr.shape == (Wt, Ht):
            arr = arr.T

    arr = _resize_if_needed(arr, fixed_hw)
    img = _normalize01(arr)
    return np.expand_dims(img, axis=0)  # [1,H,W]


def _infer_hw_from_png_path(
    p: Path,
    fixed_hw: Optional[Tuple[int, int]] = FIXED_HW
) -> Tuple[int, int]:
    if fixed_hw is not None:
        return int(fixed_hw[0]), int(fixed_hw[1])
    arr = np.array(Image.open(p))
    return int(arr.shape[0]), int(arr.shape[1])


def _try_get_followup_years_row(row: pd.Series) -> Optional[float]:
    for col in FOLLOWUP_YEARS_CANDIDATES:
        if col in row.index:
            try:
                v = float(row[col])
            except Exception:
                continue
            if np.isnan(v):
                continue
            if "day" in col.lower():
                return v / 365.25
            return v
    return None


def _mask_from_followup(y_event: np.ndarray, followup_years: Optional[float]) -> Optional[np.ndarray]:
    """
    Observed at horizon t if:
      - followup_years >= t, OR
      - event occurred by t
    """
    if followup_years is None:
        return None
    follow = np.array([(followup_years >= t) for t in HORIZONS], dtype=np.float32)
    event = (y_event > 0).astype(np.float32)
    return np.maximum(follow, event)


def _mask_from_posneg(y_event: np.ndarray, y_neg: np.ndarray) -> np.ndarray:
    return np.maximum((y_neg > 0).astype(np.float32), (y_event > 0).astype(np.float32))


def _infer_need_flip(df: pd.DataFrame) -> bool:
    cols = [c for c in RISK_POS_COLS if c in df.columns]
    if not cols:
        return False
    sub = df[cols].head(INFER_SAMPLE_ROWS).copy()
    for c in cols:
        sub[c] = _to_int01(sub[c])
    c_use = cols[-1]  # prefer 5y
    mean_val = float(sub[c_use].mean())
    # if mean > 0.5, it's probably "negatives" not "events"
    return mean_val > 0.5


def _get_y_event_from_row(row0: pd.Series, flip: bool) -> np.ndarray:
    y_raw = np.array([float(row0[c]) for c in RISK_POS_COLS], dtype=np.float32)
    return (1.0 - y_raw) if flip else y_raw


def _build_exam_groups(df: pd.DataFrame, need_flip: bool):
    df = df.copy()
    df["side_view"] = (
        df[LAT_COL].astype(str).str.strip()
        + "-"
        + df[VIEWPOS_COL].astype(str).str.strip()
    )

    has_neg = all(c in df.columns for c in RISK_NEG_COLS)

    exam_groups = []
    for exam_id, df_exam in df.groupby(EXAM_ID_COL):
        df_exam = df_exam.reset_index(drop=True)
        row0 = df_exam.iloc[0]

        y_event = _get_y_event_from_row(row0, flip=need_flip)

        if has_neg:
            y_neg = np.array([float(row0[c]) for c in RISK_NEG_COLS], dtype=np.float32)
            mask = _mask_from_posneg(y_event, y_neg)
        else:
            followup_years = _try_get_followup_years_row(row0)
            mask_fu = _mask_from_followup(y_event, followup_years)
            mask = mask_fu if mask_fu is not None else np.ones_like(y_event, dtype=np.float32)

        p0 = None
        for _, r in df_exam.iterrows():
            v = r[CUR_PNG_COL]
            if isinstance(v, str) and v != "":
                p0 = Path(v)
                break
        if p0 is None:
            continue

        hw = _infer_hw_from_png_path(p0, FIXED_HW)
        exam_groups.append((str(exam_id), df_exam, y_event, mask, hw))

    return exam_groups


class _LRUCache:
    def __init__(self, max_items: int = 1024):
        self.max_items = max_items
        self._d: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get(self, key: str):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key: str, value: np.ndarray):
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.max_items:
            self._d.popitem(last=False)


class CurrentOnlyDataset(Dataset):
    """
    Baseline current-only dataset.

    Returns:
      imgs: [4,1,H,W] current exam only
      delta_feat: [2] zeros for compatibility
      has_prior_views: [4] zeros for compatibility
      y: [5]
      mask: [5]
    """
    def __init__(self, split: str = "train", csv_path: Path = CSV_PATH):
        df = pd.read_csv(csv_path, low_memory=False)
        df = df[df[CUR_PNG_COL].notna() & (df[CUR_PNG_COL] != "")].copy()

        if "split" not in df.columns:
            raise RuntimeError("CSV has no 'split' column. Run splitDataset.py first.")
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split '{split}'")
        df = df[df["split"] == split].copy()

        for col in [EXAM_ID_COL, LAT_COL, VIEWPOS_COL]:
            if col not in df.columns:
                raise RuntimeError(f"CSV must have '{col}' column")

        missing_pos = [c for c in RISK_POS_COLS if c not in df.columns]
        if missing_pos:
            raise RuntimeError(f"Missing required POS risk columns: {missing_pos}")

        for c in RISK_POS_COLS:
            df[c] = _to_int01(df[c])
        if all(c in df.columns for c in RISK_NEG_COLS):
            for c in RISK_NEG_COLS:
                df[c] = _to_int01(df[c])

        if AUTO_INFER_LABEL_FLIP:
            need_flip = _infer_need_flip(df)
        else:
            need_flip = (not LABEL_COLS_ARE_EVENT)

        self.flip_to_event = need_flip
        self.split = split
        print(f"[CurrentOnlyDataset:{split}] flip_to_event={self.flip_to_event} FIXED_HW={FIXED_HW}")

        self.exam_groups = _build_exam_groups(df, need_flip=self.flip_to_event)
        self.cache = _LRUCache(IMAGE_CACHE_MAX_ITEMS) if ENABLE_IMAGE_CACHE else None

    def __len__(self) -> int:
        return len(self.exam_groups)

    def _load_cached(self, path: str) -> np.ndarray:
        if self.cache is None:
            return _load_png_16bit_to_chw(Path(path), FIXED_HW)
        v = self.cache.get(path)
        if v is None:
            v = _load_png_16bit_to_chw(Path(path), FIXED_HW)
            self.cache.put(path, v)
        return v

    def __getitem__(self, idx: int):
        exam_id, df_exam, y_event, mask, (H, W) = self.exam_groups[idx]

        df_exam = df_exam.copy()
        df_exam["side_view"] = (
            df_exam[LAT_COL].astype(str).str.strip()
            + "-"
            + df_exam[VIEWPOS_COL].astype(str).str.strip()
        )

        cur_imgs = []
        for view_name in VIEW_ORDER:
            df_view = df_exam[df_exam["side_view"] == view_name]
            img = np.zeros((1, H, W), dtype=np.float32)

            if len(df_view) != 0:
                row = df_view.iloc[0]
                cur_path = row[CUR_PNG_COL]
                if isinstance(cur_path, str) and cur_path != "" and os.path.isfile(cur_path):
                    img = self._load_cached(cur_path)

            cur_imgs.append(img)

        imgs = np.stack(cur_imgs, axis=0)  # [4,1,H,W]

        # kept only for compatibility with older training loops
        delta_feat = np.array([0.0, 0.0], dtype=np.float32)
        has_prior_views = np.zeros(4, dtype=np.float32)

        return (
            torch.from_numpy(imgs),
            torch.from_numpy(delta_feat),
            torch.from_numpy(has_prior_views),
            torch.from_numpy(y_event.astype(np.float32)),
            torch.from_numpy(mask.astype(np.float32)),
        )
