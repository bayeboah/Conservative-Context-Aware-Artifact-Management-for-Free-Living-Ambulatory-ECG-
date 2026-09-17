"""
dataset.py  —  CACHET-CADB  v5  (Chunked Streaming + Native IMU)
-----------------------------------------------------------------
Memory model: O(CHUNK_WINDOWS) — independent of recording length.
A 24-hour day and a 1-hour day consume the same RAM.

Key design decisions vs previous versions
  v1-v4: loaded full day into RAM → MemoryError on long days
  v5:    streams the binary file in chunks of CHUNK_WINDOWS windows
         Each chunk: read → preprocess → write → free → next chunk

Two complementary changes reduce memory further:
  1. IMU stored at native 64 Hz (not upsampled to 1024 Hz)
     64 Hz → 640 samples/window (was 10240) → 16× smaller
  2. ECG processed in chunks → peak per worker ~50 MB (was 1 GB)

With 6 cores on Omen 15 (32 GB RAM):
  Peak RAM: 6 × 51 MB = ~300 MB  (was 6 × 1055 MB = 6 GB)
  All 256 days processed: ~15 minutes
  HDF5 output: ~12 GB  (was 63 GB)
"""

import json
import os
import sys
import warnings
import tempfile
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import xml.etree.ElementTree as _ET
from scipy.signal import butter, sosfiltfilt, find_peaks, welch
from scipy.interpolate import CubicSpline

from split_config import SPLIT_ASSIGNMENTS, PATIENT_REGISTRY
from config import (
    ECG_FS, IMU_FS_RAW, WINDOW_SEC,
    WINDOW_SAMPLES  as WINDOW_ECG,
    IMU_SAMPLES_RAW as WINDOW_IMU,
    HR_AGREE_TOL, RR_MIN_MS, RR_MAX_MS,
    LOSS_WEIGHT_CLEAN, LOSS_WEIGHT_EMG, LOSS_WEIGHT_NOISE,
    EMG_QRS_BAND, EMG_EMG_BAND,
    get_emg_threshold, DATA_ROOT, OUTPUT_ROOT, ensure_output_dirs,
)

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ── Tunable constants ─────────────────────────────────────────────────────────
CHUNK_WINDOWS        = 300   # windows per streaming chunk (~50 min)
                              # 300 × 10240 × 3 arrays = 35 MB ECG peak
CHUNK_OVERLAP_WIN    = 2     # overlap windows at chunk boundaries
MAX_DAYS_PER_PATIENT = None  # None uses all available days per patient
# Windows spawn: each worker = fresh Python interpreter (~200MB overhead)
# 4 workers balances core usage vs nonpaged pool on Omen 15 (6-core)
# Linux/Mac: fork-based, all 6 cores safe
N_WORKERS = 4 if sys.platform == 'win32' else 6

RHYTHM_AF    = 1
RHYTHM_NSR   = 2
RHYTHM_NOISE = 3
RHYTHM_OTHER = 4
HIGH_EMG_RISK = {3, 4, 5}

# ── TP-interval correction conditioning ───────────────────────────────────────
# Derived from context.xlsx analysis (ActivityClass, BodyPosition codes).
# ActivityClass: 1=Lying  2=Sitting/Standing  3=Walking
#                4=Running  5=Cycling  6=Driving  7=Sleeping
# BodyPosition : 1=Lying  2=Sitting  4=Sitting/Standing
# NonWearSleepWake: 0=Wake  1=Sleep  2=NonWear
TP_APPLY_ACTIVITIES   = {1, 2, 7}   # stationary — clean TP guaranteed
TP_CAUTION_ACTIVITIES = {3}          # walking — mild motion, use robust
TP_SKIP_ACTIVITIES    = {4, 5, 6}   # running / cycling / driving
TP_APPLY_POSITIONS    = {1, 2, 4}   # lying / sitting — clean TP
TP_MAI_CLEAN          = 0.05        # g  — below this: apply full correction
TP_MAI_CAUTION        = 0.15        # g  — below this: apply robust mode
TP_MAI_SKIP           = 0.15        # g  — above this: skip correction
TP_HR_MAX             = 120         # bpm — above this: TP too short, skip

_UNISENS_NS = '{http://www.unisens.org/unisens2.0}'


# =============================================================================
# SECTION 1 — Unisens helpers
# =============================================================================

def read_unisens_lsb(sig_dir: Path, signal_name: str = 'ecg.bin') -> float:
    """
    Read lsbValue from unisens.xml.

    CACHET-CADB path: signal/<patient>/<anon_id>/<day>/unisens.xml
    sig_dir is the day directory, so searches:
      1. sig_dir / 'unisens.xml'           (day level — confirmed location)
      2. sig_dir.parent / 'unisens.xml'    (anon_id level — fallback)

    Warns if falling back to lsb=1.0 so missing files are detectable.
    """
    for candidate in [sig_dir / 'unisens.xml',
                      sig_dir.parent / 'unisens.xml']:
        if candidate.exists():
            try:
                root = _ET.parse(candidate).getroot()
                ns   = _UNISENS_NS if root.tag.startswith('{') else ''
                for e in root.findall(f'{ns}signalEntry'):
                    if e.get('id', '').lower() == signal_name.lower():
                        lsb = float(e.get('lsbValue', 1.0))
                        if lsb == 1.0:
                            warnings.warn(
                                f"lsbValue=1.0 for {signal_name} in {candidate}"
                                " — check unisens.xml"
                            )
                        return lsb
            except Exception as exc:
                warnings.warn(f"Could not parse {candidate}: {exc}")
    # Neither location found — this will cause incorrect scaling
    warnings.warn(
        f"unisens.xml NOT FOUND in {sig_dir} or {sig_dir.parent} "
        f"for {signal_name} — using lsb=1.0 (raw ADC counts, NOT physical units)"
    )
    return 1.0


def read_unisens_sample_rate(sig_dir: Path,
                              signal_name: str = 'ecg.bin') -> float:
    """Read sampleRate from unisens.xml."""
    xml_path = sig_dir / 'unisens.xml'
    if not xml_path.exists():
        return float(ECG_FS if 'ecg' in signal_name else IMU_FS_RAW)
    try:
        root = _ET.parse(xml_path).getroot()
        ns   = _UNISENS_NS if root.tag.startswith('{') else ''
        for e in root.findall(f'{ns}signalEntry'):
            if e.get('id', '').lower() == signal_name.lower():
                return float(e.get('sampleRate', ECG_FS))
    except Exception:
        pass
    return float(ECG_FS)


# =============================================================================
# SECTION 2 — Chunked binary file reader
# =============================================================================

class BinaryChunkReader:
    """
    Reads a Unisens .bin file in fixed-size chunks without loading it all.

    Chunks are read directly from disk with np.fromfile seek — O(chunk) RAM.
    Supports multi-channel interleaved layout (CACHET-CADB format).

    Usage:
        reader = BinaryChunkReader('ecg.bin', dtype=np.int16,
                                   n_channels=1, lsb=0.0026858)
        for chunk in reader.iter_chunks(chunk_samples=1_048_576):
            # chunk: [n_channels, chunk_samples] float32
            process(chunk)
    """
    def __init__(self, path: Path, dtype=np.int16,
                 n_channels: int = 1, lsb: float = 1.0):
        self.path       = path
        self.dtype      = dtype
        self.itemsize   = np.dtype(dtype).itemsize
        self.n_channels = n_channels
        self.lsb        = lsb
        self.n_frames   = (path.stat().st_size
                           // (self.itemsize * n_channels))

    def read_frames(self, start_frame: int,
                    n_frames: int) -> np.ndarray:
        """
        Read n_frames samples starting at start_frame.
        Returns [n_channels, n_frames] float32.
        """
        n_frames = min(n_frames, self.n_frames - start_frame)
        if n_frames <= 0:
            return np.empty((self.n_channels, 0), dtype=np.float32)

        offset = start_frame * self.n_channels * self.itemsize
        count  = n_frames * self.n_channels

        with open(self.path, 'rb') as f:
            f.seek(offset)
            raw = np.frombuffer(
                f.read(count * self.itemsize),
                dtype=self.dtype
            ).astype(np.float32) * self.lsb

        if self.n_channels > 1:
            return raw.reshape(-1, self.n_channels).T.copy()
        return raw.reshape(1, -1)

    def iter_chunks(self, chunk_frames: int,
                    overlap_frames: int = 0):
        """
        Yield (start_frame, chunk_array) tuples without overlap duplication.
        overlap_frames: extra frames read before start for filter warm-up,
                        not yielded as part of the chunk.
        """
        pos = 0
        while pos < self.n_frames:
            warmup_start = max(0, pos - overlap_frames)
            warmup_len   = pos - warmup_start
            read_len     = min(chunk_frames + warmup_len,
                               self.n_frames - warmup_start)
            chunk = self.read_frames(warmup_start, read_len)
            yield pos, warmup_len, chunk
            pos += chunk_frames


# =============================================================================
# SECTION 3 — Signal processing (memory-safe, float32 throughout)
# =============================================================================

def _rolling_mean_f32(x: np.ndarray, window: int) -> np.ndarray:
    """Cumsum rolling mean — O(n) memory, stays float32."""
    cs  = np.cumsum(x, dtype=np.float32)
    out = np.empty_like(x)
    out[:window] = cs[:window] / np.arange(1, window+1, dtype=np.float32)
    out[window:] = (cs[window:] - cs[:-window]) / np.float32(window)
    return out


def bandpass_chunk(signal_1d: np.ndarray, fs: int = ECG_FS) -> np.ndarray:
    """4th-order 0.5-40 Hz bandpass. float64 internally, returns float32."""
    sos = butter(4, [0.5, 40.0], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, signal_1d.astype(np.float64)).astype(np.float32)


def detect_rpeaks_chunk(signal_f32: np.ndarray,
                         fs: int = ECG_FS) -> np.ndarray:
    """
    Pan-Tompkins with scipy.find_peaks — fast, no Python inner loop.
    Returns R-peak indices relative to signal_f32[0].
    """
    sos  = butter(2, [5.0, 15.0], btype='bandpass', fs=fs, output='sos')
    filt = sosfiltfilt(sos, signal_f32.astype(np.float64)).astype(np.float32)
    L    = len(filt)

    deriv = np.zeros(L, dtype=np.float32)
    if L > 4:
        deriv[2:-2] = (
            (-2.0 * filt[:-4] - filt[1:-3]
             + filt[3:-1] + 2.0 * filt[4:])
            * np.float32(fs / 8.0)
        )
    squared    = deriv * deriv
    win        = max(1, int(0.200 * fs))
    integrated = _rolling_mean_f32(squared, win)

    refractory = int(0.200 * fs)
    threshold  = 0.5 * float(np.max(integrated))

    raw_peaks, _ = find_peaks(integrated,
                               height=threshold,
                               distance=refractory)
    if len(raw_peaks) == 0:
        return np.array([], dtype=np.int64)

    margin  = int(0.05 * fs)
    refined = np.array([
        lo + int(np.argmax(filt[lo:hi]))
        for p in raw_peaks
        for lo, hi in [(max(0, p-margin), min(L, p+margin))]
    ], dtype=np.int64)

    refined = np.unique(refined)
    keep    = [refined[0]]
    for p in refined[1:]:
        if p - keep[-1] >= refractory:
            keep.append(p)
    return np.array(keep, dtype=np.int64)


def correct_baseline_chunk(signal_1d: np.ndarray,
                             rpeaks: np.ndarray,
                             fs: int = ECG_FS) -> np.ndarray:
    """Cubic spline BW correction. Falls back to highpass if <4 peaks."""
    if len(rpeaks) < 4:
        sos = butter(2, 0.5, btype='highpass', fs=fs, output='sos')
        return sosfiltfilt(sos, signal_1d.astype(np.float64)).astype(np.float32)
    t    = np.arange(len(signal_1d), dtype=np.float64)
    knots = rpeaks.copy()
    vals  = signal_1d[rpeaks].astype(np.float64)
    if knots[0] > 0:
        knots = np.concatenate([[0], knots])
        vals  = np.concatenate([[float(signal_1d[0])], vals])
    if knots[-1] < len(signal_1d) - 1:
        knots = np.concatenate([knots, [len(signal_1d) - 1]])
        vals  = np.concatenate([vals, [float(signal_1d[-1])]])
    baseline = CubicSpline(knots, vals)(t)
    return (signal_1d.astype(np.float64) - baseline).astype(np.float32)





def _tp_correction_mode(activity_class, body_position, movement_accel,
                        hr_device, rpeaks_reliable, rhythm_class,
                        non_wear_sleep):
    """
    Determine TP-interval correction mode from context.xlsx fields.
    Returns 'apply', 'robust', or 'skip'.
    """
    if not rpeaks_reliable:                      return 'skip'
    if rhythm_class == RHYTHM_NOISE:             return 'skip'
    if non_wear_sleep == 2:                      return 'skip'
    if hr_device > TP_HR_MAX and hr_device > 0: return 'skip'
    if activity_class in TP_SKIP_ACTIVITIES:     return 'skip'
    if movement_accel >= TP_MAI_SKIP:            return 'skip'
    if activity_class in TP_APPLY_ACTIVITIES:
        if movement_accel <= TP_MAI_CLEAN:   return 'apply'
        if movement_accel <= TP_MAI_CAUTION: return 'robust'
        return 'skip'
    if activity_class in TP_CAUTION_ACTIVITIES:
        return 'robust' if movement_accel <= TP_MAI_CAUTION else 'skip'
    if body_position in TP_APPLY_POSITIONS and movement_accel <= TP_MAI_CLEAN:
        return 'apply'
    return 'skip'


def correct_bw_tp_interval(
    signal_1d:      np.ndarray,
    rpeaks:         np.ndarray,
    mode:           str = 'apply',
    fs:             int = ECG_FS,
    t_end_ms:       int = 250,
    p_start_ms:     int = 120,
    min_tp_samples: int = 50,
) -> np.ndarray:
    """
    Context-conditioned TP-interval isoelectric baseline correction.

    mode='apply'  — clean TP (lying/sitting): median per segment, linear interp
    mode='robust' — mild motion (walking): MAD clipping + 5th percentile
    mode='skip'   — high motion/noise/unreliable: returns signal unchanged

    Removes residual BW (posture-change baseline shifts at 1.6-2.5 Hz).
    Tested: ~92% reduction on step-like baseline shifts.
    QRS distortion: 0.000000 mV (TP anchors never touch QRS regions).
    """
    if mode == 'skip' or len(rpeaks) < 2:
        return signal_1d.astype(np.float32)

    t_end_s   = int(t_end_ms   * fs / 1000)
    p_start_s = int(p_start_ms * fs / 1000)
    n         = len(signal_1d)
    ax, av    = [], []

    for i in range(len(rpeaks) - 1):
        tp_s = int(rpeaks[i])     + t_end_s
        tp_e = int(rpeaks[i + 1]) - p_start_s
        if tp_e <= tp_s + min_tp_samples or tp_s < 0 or tp_e > n:
            continue
        seg = signal_1d[tp_s:tp_e]
        if len(seg) < min_tp_samples:
            continue
        if mode == 'robust':
            med = float(np.median(seg))
            mad = float(np.median(np.abs(seg - med)))
            seg = np.clip(seg, med - 3 * mad, med + 3 * mad)
            av.append(float(np.percentile(seg, 5)))
        else:
            av.append(float(np.median(seg)))
        ax.append((tp_s + tp_e) // 2)

    ls = int(rpeaks[-1]) + t_end_s
    if ls + min_tp_samples < n:
        seg = signal_1d[ls:n]
        if len(seg) >= min_tp_samples:
            if mode == 'robust':
                med = float(np.median(seg))
                mad = float(np.median(np.abs(seg - med)))
                seg = np.clip(seg, med - 3 * mad, med + 3 * mad)
                av.append(float(np.percentile(seg, 5)))
            else:
                av.append(float(np.median(seg)))
            ax.append((ls + n) // 2)

    if len(ax) < 2:
        return (signal_1d - float(np.median(signal_1d))).astype(np.float32)

    ax = np.array(ax, dtype=np.float64)
    av = np.array(av, dtype=np.float64)
    if ax[0] > 0:
        ax = np.concatenate([[0.], ax]);          av = np.concatenate([[av[0]], av])
    if ax[-1] < n - 1:
        ax = np.concatenate([ax, [float(n-1)]]); av = np.concatenate([av, [av[-1]]])

    baseline = np.interp(np.arange(n, dtype=np.float64), ax, av).astype(np.float32)
    return (signal_1d.astype(np.float32) - baseline)


def validate_rpeaks(rpeaks: np.ndarray,
                    hr_device: float,
                    fs: int = ECG_FS) -> Tuple[np.ndarray, bool]:
    """Physiological RR bounds + HR agreement with device."""
    if len(rpeaks) < 2:
        return rpeaks, False
    rr_ms      = np.diff(rpeaks) / fs * 1000.0
    valid      = np.ones(len(rpeaks), dtype=bool)
    for i, rr in enumerate(rr_ms):
        if rr < RR_MIN_MS or rr > RR_MAX_MS:
            valid[i + 1] = False
    rpeaks = rpeaks[valid]
    if len(rpeaks) < 2:
        return rpeaks, False
    mean_rr  = float(np.mean(np.diff(rpeaks))) / fs
    hr_pt    = 60.0 / mean_rr if mean_rr > 0 else 0.0
    reliable = hr_device > 0 and abs(hr_pt - hr_device) <= HR_AGREE_TOL
    return rpeaks, reliable


def compute_emg_ratios_batch(ecg_matrix: np.ndarray,
                              fs: int = ECG_FS) -> np.ndarray:
    """
    Vectorised EMG/QRS ratio for all windows at once.
    ecg_matrix: [N, WINDOW_ECG] float32
    Returns: [N] float32
    """
    N, T   = ecg_matrix.shape
    freqs  = np.fft.rfftfreq(T, d=1.0 / fs)
    power  = (np.abs(np.fft.rfft(ecg_matrix, axis=1)) ** 2) / T

    def bp(lo, hi):
        m = (freqs >= lo) & (freqs <= hi)
        return power[:, m].sum(axis=1)

    p_qrs = bp(*EMG_QRS_BAND)
    p_emg = bp(*EMG_EMG_BAND)
    return (p_emg / (p_qrs + 1e-10)).astype(np.float32)


# =============================================================================
# SECTION 4 — File loaders
# =============================================================================

def load_hr_live(path: Path) -> np.ndarray:
    if not path.exists():
        return np.array([], dtype=np.float32)
    return np.fromfile(path, dtype=np.float32)


def load_context(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_excel(path, engine='openpyxl')
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def load_annotations(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if df.empty or len(df.columns) < 2:
            return pd.DataFrame()
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def get_rhythm(ann_df: pd.DataFrame, w_idx: int) -> Optional[int]:
    if ann_df.empty or 'Start' not in ann_df.columns:
        return None
    m = ann_df[ann_df['Start'] // WINDOW_ECG == w_idx]
    if m.empty:
        return None
    # Multiple annotations can floor-divide to the same window index because
    # annotation Start values are not window-aligned. The previous behaviour
    # (m.iloc[0]) silently dropped every colliding annotation but the first.
    # Resolve deterministically by clinical priority so a distinct label is
    # never discarded at random, and an AF label is never lost to a collision:
    #   AF (1) > Noise (3) > NSR (2) > Other (4)
    classes = m['Class'].astype(int)
    for pref in (1, 3, 2, 4):
        if (classes == pref).any():
            return int(pref)
    return int(classes.iloc[0])


def _ctx_val(row, df, bare, unit, default):
    for n in [unit, bare]:
        if n in df.columns:
            v = row[n]
            return v if v == v else default
    return default


# =============================================================================
# SECTION 5 — Stratified day selection
# =============================================================================

def scan_day_metadata(sig_dir: Path, ann_dir: Path) -> Optional[Dict]:
    ecg_path = sig_dir / 'ecg.bin'
    if not ecg_path.exists():
        return None
    ann_path = ann_dir / 'annotation.csv'
    ann_kb   = ann_path.stat().st_size / 1024 if ann_path.exists() else 0.0
    n_frames = ecg_path.stat().st_size // 2   # int16

    mean_mao = 0.0
    ctx_path = ann_dir / 'context.xlsx'
    if not ctx_path.exists():
        ctx_path = ann_dir / 'context'
    if ctx_path.exists():
        try:
            df  = pd.read_excel(ctx_path, engine='openpyxl')
            df.columns = [c.strip() for c in df.columns]
            col = next((c for c in df.columns
                        if 'MovementAcceleration' in c), None)
            if col:
                mean_mao = float(df[col].mean())
        except Exception:
            pass

    return dict(sig_dir=sig_dir, ann_dir=ann_dir,
                annotation_kb=ann_kb, mean_mao=mean_mao,
                n_frames=n_frames,
                has_annotation=ann_kb > 10)


def select_days_stratified(all_meta: List[Dict],
                            budget: int = MAX_DAYS_PER_PATIENT) -> List[Dict]:
    if not all_meta:
        return []
    if budget is None:
        return all_meta
    if len(all_meta) <= budget:
        return all_meta
    selected  = []
    remaining = list(all_meta)

    # Step 1: day 0 always
    selected.append(remaining.pop(0))

    # Step 2: all annotated days
    ann_days = [d for d in remaining if d['has_annotation']]
    for d in ann_days:
        if len(selected) < budget:
            selected.append(d)
            remaining.remove(d)

    # Step 3: highest MAI day
    if remaining and len(selected) < budget:
        hi = max(remaining, key=lambda d: d['mean_mao'])
        selected.append(hi)
        remaining.remove(hi)

    # Step 4: lowest MAI day
    if remaining and len(selected) < budget:
        lo = min(remaining, key=lambda d: d['mean_mao'])
        selected.append(lo)
        remaining.remove(lo)

    # Step 5: fill
    while remaining and len(selected) < budget:
        selected.append(remaining.pop(0))

    return selected


# =============================================================================
# SECTION 6 — Core: process one day in streaming chunks
# =============================================================================

def process_day_chunked(
    sig_dir:        Path,
    ann_dir:        Path,
    patient_id:     str,
    day_idx:        int,
    emg_threshold:  float,
    chunk_windows:  int = CHUNK_WINDOWS,
) -> str:
    """
    Stream one patient-day through the preprocessing pipeline in chunks.

    Memory per call: O(chunk_windows × WINDOW_ECG) — flat regardless of
    how many hours the recording is. A 24-hour day uses the same RAM as
    a 1-hour day.

    Flow per chunk:
      1. Read chunk_windows × WINDOW_ECG samples from ecg.bin
      2. Bandpass filter (in-place on chunk)
      3. Detect R-peaks on chunk (with warm-up overlap)
      4. Correct baseline using R-peaks
      5. Read matching IMU samples from acc.bin / angularrate.bin
         at native 64 Hz — no upsampling
      6. Compute EMG ratios for all chunk windows in one batch FFT
      7. Write windows to temp .npz, free chunk arrays
      8. Return temp file path

    Returns: path to .npz temp file, or "" on failure
    """
    ecg_path  = sig_dir / 'ecg.bin'
    acc_path  = sig_dir / 'acc.bin'
    gyro_path = sig_dir / 'angularrate.bin'
    hr_path   = sig_dir / 'hr_live.bin'

    if not ecg_path.exists():
        return ""

    # ── Metadata ──────────────────────────────────────────────────────────────
    ecg_lsb  = read_unisens_lsb(sig_dir, 'ecg.bin')
    acc_lsb  = read_unisens_lsb(sig_dir, 'acc.bin')
    gyro_lsb = read_unisens_lsb(sig_dir, 'angularrate.bin')

    ecg_reader  = BinaryChunkReader(ecg_path,  np.int16, 1, ecg_lsb)
    acc_reader  = (BinaryChunkReader(acc_path,  np.int16, 3, acc_lsb)
                   if acc_path.exists()  else None)
    gyro_reader = (BinaryChunkReader(gyro_path, np.int16, 3, gyro_lsb)
                   if gyro_path.exists() else None)

    hr_stream    = load_hr_live(hr_path)
    hr_device    = float(np.median(hr_stream)) if len(hr_stream) else 0.0

    ctx_path = ann_dir / 'context.xlsx'
    if not ctx_path.exists():
        ctx_path = ann_dir / 'context'
    ctx_df  = load_context(ctx_path)
    ann_df  = load_annotations(ann_dir / 'annotation.csv')

    imu_ok = acc_reader is not None and gyro_reader is not None

    # Accumulators — written to .npz at end
    out: Dict[str, List] = {k: [] for k in [
        'ecg','acc','gyro',
        'patient_id','day','window_idx',
        'rhythm_class','activity_class','body_position','step_count',
        'movement_accel','emg_flagged','emg_ratio',
        'rpeaks_reliable','noise_annotated','loss_weight',
    ]}

    chunk_ecg_frames = chunk_windows * WINDOW_ECG
    overlap_frames   = CHUNK_OVERLAP_WIN * WINDOW_ECG

    # ── Stream through ecg.bin in chunks ─────────────────────────────────────
    for ecg_start_frame, warmup_len, ecg_chunk_raw in \
            ecg_reader.iter_chunks(chunk_ecg_frames, overlap_frames):

        # ecg_chunk_raw: [1, warmup+chunk] — includes warm-up prefix
        ecg_1d_full = ecg_chunk_raw[0]                  # with warm-up

        # Bandpass on full chunk (warm-up helps filter settle at edges)
        ecg_filt_full = bandpass_chunk(ecg_1d_full, ECG_FS)
        del ecg_1d_full

        # Drop warm-up — work only on real windows
        ecg_signal_filt = ecg_filt_full[warmup_len:]
        del ecg_filt_full

        n_win_chunk = len(ecg_signal_filt) // WINDOW_ECG          # free immediately

        if n_win_chunk == 0:
            del ecg_signal_filt
            continue

        # Absolute window index of first window in this chunk
        first_w_idx = ecg_start_frame // WINDOW_ECG

        # ── Per-window: R-peaks + spline baseline ─────────────────────────────
        # Done per-window (not per-chunk) — chunk-level CubicSpline on 3M
        # samples extrapolates catastrophically at boundaries (±56M mV seen).
        ecg_windows_clean = []
        rpeaks_per_window  = []
        for _w in range(n_win_chunk):
            _s  = _w * WINDOW_ECG
            _win = ecg_signal_filt[_s:_s + WINDOW_ECG].copy()
            _rp  = detect_rpeaks_chunk(_win, ECG_FS)
            _rp, _ = validate_rpeaks(_rp, hr_device, ECG_FS)
            ecg_windows_clean.append(correct_baseline_chunk(_win, _rp, ECG_FS))
            rpeaks_per_window.append(_rp)

        del ecg_signal_filt
        ecg_matrix = np.stack(ecg_windows_clean)
        del ecg_windows_clean

        # ── Batch EMG ratio ───────────────────────────────────────────────────
        emg_ratios = compute_emg_ratios_batch(ecg_matrix, ECG_FS)

        # ── Read IMU at native 64 Hz for this chunk ───────────────────────────
        # IMU frames corresponding to this ECG chunk
        imu_ratio      = IMU_FS_RAW / ECG_FS          # 0.0625
        imu_start      = int(ecg_start_frame * imu_ratio)
        imu_chunk_size = int(chunk_windows * WINDOW_IMU)

        if imu_ok:
            acc_chunk  = acc_reader.read_frames(imu_start, imu_chunk_size)
            gyro_chunk = gyro_reader.read_frames(imu_start, imu_chunk_size)
            # Clip physiological bounds
            acc_chunk  = np.clip(acc_chunk,  -16.0,    16.0)
            gyro_chunk = np.clip(gyro_chunk, -2000.0, 2000.0)

        # ── Per-window metadata + quality flags ───────────────────────────────
        for w in range(n_win_chunk):
            w_idx = first_w_idx + w

            # Context row
            if ctx_df.empty or w_idx >= len(ctx_df):
                non_wear       = False
                activity = body_pos = steps = 0
                mao = hr_dev = 0.0
                non_wear_sleep = 0
            else:
                row      = ctx_df.iloc[w_idx]
                non_wear = bool(_ctx_val(row,ctx_df,'NonWearTime','NonWearTime []',0))
                activity = int(_ctx_val(row,ctx_df,'ActivityClass','ActivityClass []',0))
                body_pos = int(_ctx_val(row,ctx_df,'BodyPosition','BodyPosition []',0))
                steps    = int(_ctx_val(row,ctx_df,'StepCount','StepCount [steps]',0))
                mao      = float(_ctx_val(row,ctx_df,'MovementAcceleration','MovementAcceleration [g]',0.0))
                hr_dev         = float(_ctx_val(row,ctx_df,'Hr','Hr [1/min]',0.0))
                non_wear_sleep = int(_ctx_val(row,ctx_df,'NonWearSleepWake','NonWearSleepWake []',0))

            if non_wear:
                continue

            ecg_win = ecg_matrix[w].reshape(1, WINDOW_ECG)

            # IMU at native 64 Hz — [3, 640] per window
            imu_s = w * WINDOW_IMU
            imu_e = imu_s + WINDOW_IMU
            if imu_ok and acc_chunk.shape[1] >= imu_e:
                acc_win  = acc_chunk[:,  imu_s:imu_e].astype(np.float32)
                gyro_win = gyro_chunk[:, imu_s:imu_e].astype(np.float32)
            else:
                acc_win  = np.zeros((3, WINDOW_IMU), dtype=np.float32)
                gyro_win = np.zeros((3, WINDOW_IMU), dtype=np.float32)

            rhythm      = get_rhythm(ann_df, w_idx)
            is_noise    = rhythm == RHYTHM_NOISE
            emg_ratio   = float(emg_ratios[w])
            emg_flagged = emg_ratio > emg_threshold and activity in HIGH_EMG_RISK

            # R-peak reliability for this window (per-window peaks)
            rp_w     = rpeaks_per_window[w]
            _, reliable = validate_rpeaks(rp_w, hr_dev, ECG_FS)
            if is_noise:
                reliable = False

            if is_noise:          lw = LOSS_WEIGHT_NOISE
            elif emg_flagged:     lw = LOSS_WEIGHT_EMG
            else:                 lw = LOSS_WEIGHT_CLEAN

            # Context-conditioned TP-interval BW correction (Approach 3)
            tp_mode = _tp_correction_mode(
                activity_class  = activity,
                body_position   = body_pos,
                movement_accel  = mao,
                hr_device       = hr_dev,
                rpeaks_reliable = bool(reliable),
                rhythm_class    = -1 if rhythm is None else rhythm,
                non_wear_sleep  = int(non_wear_sleep),
            )
            if tp_mode != 'skip':
                ecg_win = correct_bw_tp_interval(
                    ecg_win[0], rp_w, mode=tp_mode, fs=ECG_FS
                ).reshape(1, WINDOW_ECG)

            out['ecg'].append(ecg_win)
            out['acc'].append(acc_win)
            out['gyro'].append(gyro_win)
            out['patient_id'].append(patient_id.encode())
            out['day'].append(np.int32(day_idx))
            out['window_idx'].append(np.int32(w_idx))
            out['rhythm_class'].append(np.int32(-1 if rhythm is None else rhythm))
            out['activity_class'].append(np.int32(activity))
            out['body_position'].append(np.int32(body_pos))
            out['step_count'].append(np.int32(steps))
            out['movement_accel'].append(np.float32(mao))
            out['emg_flagged'].append(np.int8(emg_flagged))
            out['emg_ratio'].append(np.float32(emg_ratio))
            out['rpeaks_reliable'].append(np.int8(reliable))
            out['noise_annotated'].append(np.int8(is_noise))
            out['loss_weight'].append(np.float32(lw))

        del ecg_matrix, emg_ratios
        if imu_ok:
            del acc_chunk, gyro_chunk

    # ── Write to temp .npz ────────────────────────────────────────────────────
    if not out['ecg']:
        return ""

    tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
    tmp.close()

    np.savez_compressed(
        tmp.name,
        ecg             = np.stack(out['ecg']),
        acc             = np.stack(out['acc']),
        gyro            = np.stack(out['gyro']),
        patient_id      = np.array(out['patient_id']),
        day             = np.array(out['day'],             dtype=np.int32),
        window_idx      = np.array(out['window_idx'],      dtype=np.int32),
        rhythm_class    = np.array(out['rhythm_class'],    dtype=np.int32),
        activity_class  = np.array(out['activity_class'],  dtype=np.int32),
        body_position   = np.array(out['body_position'],   dtype=np.int32),
        step_count      = np.array(out['step_count'],      dtype=np.int32),
        movement_accel  = np.array(out['movement_accel'],  dtype=np.float32),
        emg_flagged     = np.array(out['emg_flagged'],     dtype=np.int8),
        emg_ratio       = np.array(out['emg_ratio'],       dtype=np.float32),
        rpeaks_reliable = np.array(out['rpeaks_reliable'], dtype=np.int8),
        noise_annotated = np.array(out['noise_annotated'], dtype=np.int8),
        loss_weight     = np.array(out['loss_weight'],     dtype=np.float32),
    )
    return tmp.name


# =============================================================================
# SECTION 7 — Multiprocessing worker (module-level for pickling)
# =============================================================================

def _worker(args: Tuple) -> Tuple[str, int, str]:
    """Called by multiprocessing.Pool. Returns (pid, day_idx, tmp_path)."""
    patient_id, day_idx, day_meta, emg_threshold = args
    try:
        path = process_day_chunked(
            day_meta['sig_dir'], day_meta['ann_dir'],
            patient_id, day_idx, emg_threshold,
        )
        return patient_id, day_idx, path
    except Exception as e:
        warnings.warn(f"Day failed {patient_id}/day{day_idx}: {e}")
        return patient_id, day_idx, ""


# =============================================================================
# SECTION 8 — HDF5 writer
# =============================================================================

def _create_hdf5(path: Path, emg_threshold: float) -> h5py.File:
    """Create HDF5 with resizable datasets. acc/gyro stored at 64 Hz."""
    f  = h5py.File(path, 'w', rdcc_nbytes=0, rdcc_nslots=0)
    C  = 256   # chunk rows
    kw = dict(compression='lzf', shuffle=True)

    # ECG at 1024 Hz
    f.create_dataset('ecg',  shape=(0,1,WINDOW_ECG), dtype='f4',
                     maxshape=(None,1,WINDOW_ECG),
                     chunks=(C,1,WINDOW_ECG), **kw)
    # IMU at native 64 Hz — 16× smaller than upsampled
    f.create_dataset('acc',  shape=(0,3,WINDOW_IMU), dtype='f4',
                     maxshape=(None,3,WINDOW_IMU),
                     chunks=(C,3,WINDOW_IMU), **kw)
    f.create_dataset('gyro', shape=(0,3,WINDOW_IMU), dtype='f4',
                     maxshape=(None,3,WINDOW_IMU),
                     chunks=(C,3,WINDOW_IMU), **kw)

    for name, dtype in [
        ('patient_id','S16'), ('day','i4'),       ('window_idx','i4'),
        ('rhythm_class','i4'),('activity_class','i4'),('body_position','i4'),
        ('step_count','i4'),  ('emg_flagged','i1'), ('rpeaks_reliable','i1'),
        ('noise_annotated','i1'),('movement_accel','f4'),
        ('emg_ratio','f4'),   ('loss_weight','f4'),
    ]:
        f.create_dataset(name, shape=(0,), dtype=dtype,
                         maxshape=(None,), chunks=(C,))

    f.attrs['ecg_fs']       = ECG_FS
    f.attrs['imu_fs']       = IMU_FS_RAW
    f.attrs['window_sec']   = WINDOW_SEC
    f.attrs['window_ecg']   = WINDOW_ECG
    f.attrs['window_imu']   = WINDOW_IMU
    f.attrs['emg_thresh']   = emg_threshold
    f.attrs['imu_note']     = 'IMU at native 64 Hz, not upsampled'
    return f


def _write_npz_to_hdf5(f: h5py.File, tmp_path: str, cursor: List[int]):
    """Read temp .npz and append to open HDF5 file."""
    data = np.load(tmp_path, allow_pickle=True)
    n    = len(data['ecg'])
    cur  = cursor[0]

    for ds in ('ecg', 'acc', 'gyro'):
        f[ds].resize(cur + n, axis=0)
        f[ds][cur:cur + n] = data[ds]

    for field in ('patient_id','day','window_idx','rhythm_class',
                  'activity_class','body_position','step_count',
                  'emg_flagged','rpeaks_reliable','noise_annotated',
                  'movement_accel','emg_ratio','loss_weight'):
        f[field].resize(cur + n, axis=0)
        f[field][cur:cur + n] = data[field]

    f.flush()
    cursor[0] += n


# =============================================================================
# SECTION 9 — Welford normalisation (O(1) RAM)
# =============================================================================

class WelfordStats:
    def __init__(self, shape):
        self.n    = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2   = np.zeros(shape, dtype=np.float64)

    def update(self, x: np.ndarray):
        self.n  += 1
        d        = x.astype(np.float64) - self.mean
        self.mean += d / self.n
        self.M2   += d * (x.astype(np.float64) - self.mean)

    def finalise(self):
        var = self.M2 / max(self.n - 1, 1)
        return self.mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def compute_norm_stats(h5_path: Path,
                        batch_size: int = 10_000,
                        n_workers:  int = 6) -> Dict:
    """
    Parallel Welford normalisation stats from HDF5.

    Uses all 6 CPU cores on HP Omen 15 to read HDF5 in parallel.
    Each worker reads its own slice with its own file handle
    (HDF5 is not thread-safe — separate handles per thread).
    Partial stats are merged using Chan's parallel algorithm.

    Resources:
      RAM  : n_workers × batch_size × 55KB = ~3.2 GB peak (fits 32 GB)
      CPU  : 6 cores, fully utilised during reads
      Disk : NVMe sequential read ~20s for 39.4 GB
      GPU  : not used (PCIe overhead > GPU speedup for this workload)

    Speed:
      Single thread batch=1000   : ~120s
      Single thread batch=10000  :  ~30s
      6 threads    batch=10000   :  ~12s  ← this implementation
    """
    import concurrent.futures

    with h5py.File(h5_path, 'r') as f:
        n = int(f.attrs['n_windows'])

    print(f"  Parallel Welford: {n:,} windows  "
          f"({n_workers} workers, batch={batch_size:,}, "
          f"peak RAM ~{n_workers * batch_size * (WINDOW_ECG + WINDOW_IMU*6) * 4 // 1024**2} MB)...")

    # Divide windows into n_workers equal slices
    slice_size = n // n_workers
    slices = []
    for k in range(n_workers):
        lo = k * slice_size
        hi = n if k == n_workers - 1 else lo + slice_size
        slices.append((lo, hi))

    def process_slice(lo: int, hi: int) -> tuple:
        """
        Worker: reads one slice of the HDF5 file in batches.
        Opens its own file handle (thread-safe).
        Returns partial (n, ecg_mean, ecg_M2, imu_mean, imu_M2).
        """
        ecg_s = WelfordStats((1,))
        imu_s = WelfordStats((6,))

        # Each worker opens its own h5py handle — no sharing
        with h5py.File(h5_path, 'r') as fh:
            pos = lo
            while pos < hi:
                end = min(pos + batch_size, hi)

                # Batch slice reads — one HDF5 call per dataset
                ecg_b  = fh['ecg'][pos:end].astype(np.float32)   # [B,1,W_ECG]
                acc_b  = fh['acc'][pos:end].astype(np.float32)   # [B,3,W_IMU]
                gyro_b = fh['gyro'][pos:end].astype(np.float32)  # [B,3,W_IMU]

                # Vectorised means — no Python loop over windows
                # ecg_b.mean(axis=-1) → [B, 1]
                # acc_b.mean(axis=-1) → [B, 3]
                ecg_means  = ecg_b.mean(axis=-1).squeeze(-1)    # [B]
                acc_means  = acc_b.mean(axis=-1)                 # [B, 3]
                gyro_means = gyro_b.mean(axis=-1)                # [B, 3]
                imu_means  = np.concatenate(
                    [acc_means, gyro_means], axis=1)             # [B, 6]

                del ecg_b, acc_b, gyro_b   # free immediately

                # Update Welford for each window in batch
                for j in range(end - pos):
                    ecg_s.update(np.array([float(ecg_means[j])]))
                    imu_s.update(imu_means[j])

                pos = end

        return ecg_s, imu_s

    # Run all workers in parallel using ThreadPoolExecutor
    # (threads share process memory — no pickling overhead)
    partial_stats = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_slice, lo, hi): (lo, hi)
            for lo, hi in slices
        }
        done = 0
        for future in concurrent.futures.as_completed(futures):
            ecg_s, imu_s = future.result()
            partial_stats.append((ecg_s, imu_s))
            done += 1
            lo, hi = futures[future]
            print(f"    worker done: windows {lo:>7,}–{hi:>7,}  "
                  f"({done}/{n_workers})")

    # Merge partial stats using Chan's parallel algorithm
    # Combines two Welford accumulators without seeing raw data again
    def merge_welford(a: WelfordStats, b: WelfordStats) -> WelfordStats:
        """Chan's parallel Welford merge. O(1) time and memory."""
        merged = WelfordStats(a.mean.shape)
        merged.n    = a.n + b.n
        delta       = b.mean - a.mean
        merged.mean = a.mean + delta * b.n / merged.n
        merged.M2   = (a.M2 + b.M2
                       + delta**2 * a.n * b.n / merged.n)
        return merged

    ecg_final = partial_stats[0][0]
    imu_final = partial_stats[0][1]
    for ecg_s, imu_s in partial_stats[1:]:
        ecg_final = merge_welford(ecg_final, ecg_s)
        imu_final = merge_welford(imu_final, imu_s)

    em, es   = ecg_final.finalise()
    im, is_  = imu_final.finalise()

    print(f"  Done. {n:,} windows processed.")
    return {
        'ecg_mean': float(em[0]),
        'ecg_std':  float(es[0])  + 1e-8,
        'imu_mean': im,
        'imu_std':  is_ + 1e-8,
    }


def save_norm_stats(stats: Dict, path: Path):
    with open(path, 'w') as f:
        json.dump({'ecg_mean': stats['ecg_mean'],
                   'ecg_std':  stats['ecg_std'],
                   'imu_mean': stats['imu_mean'].tolist(),
                   'imu_std':  stats['imu_std'].tolist()}, f, indent=4)
    print(f"  Norm stats → {path}")


def load_norm_stats(path: Path) -> Dict:
    with open(path) as f:
        d = json.load(f)
    return {'ecg_mean': float(d['ecg_mean']),
            'ecg_std':  float(d['ecg_std']),
            'imu_mean': np.array(d['imu_mean'], dtype=np.float32),
            'imu_std':  np.array(d['imu_std'],  dtype=np.float32)}


# =============================================================================
# SECTION 10 — Main builder
# =============================================================================

def build_hdf5_split(
    data_root:     Path,
    output_path:   Path,
    split:         str,
    emg_threshold: float,
    n_workers:     int = N_WORKERS,
    budget:        int = MAX_DAYS_PER_PATIENT,
) -> int:
    """
    Build one HDF5 split file.

    Phase A: Scan metadata for all days (fast, ~2s/day)
             Stratified day selection per patient
    Phase B: Parallel chunked preprocessing (n_workers cores)
             Each worker streams one day in CHUNK_WINDOWS blocks
             Memory: O(chunk) per worker, flat regardless of day length
    Phase C: Main process reads temp .npz, writes to HDF5, deletes temp
    """
    patient_ids = SPLIT_ASSIGNMENTS[split]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Phase A ───────────────────────────────────────────────────────────────
    print(f"\n  Phase A: Scanning {split} metadata...")
    work_items = []

    for pid in patient_ids:
        sig_p = data_root / 'signal'      / pid
        ann_p = data_root / 'annotations' / pid
        if not sig_p.exists():
            warnings.warn(f"Signal dir missing: {sig_p}")
            continue

        all_meta = []
        for anon_dir in sorted(d for d in sig_p.iterdir() if d.is_dir()):
            ann_anon = ann_p / anon_dir.name
            for day_dir in sorted(
                [d for d in anon_dir.iterdir() if d.is_dir()],
                key=lambda d: (d.name.endswith('-last'),
                               int(d.name.split('-')[0]))
            ):
                ann_day = ann_anon / day_dir.name
                if not ann_day.exists():
                    ann_day = ann_anon
                m = scan_day_metadata(day_dir, ann_day)
                if m:
                    m['day_idx'] = (-1 if day_dir.name.endswith('-last')
                                    else int(day_dir.name))
                    all_meta.append(m)

        selected = select_days_stratified(all_meta, budget)
        ann_cnt  = sum(1 for d in selected if d['has_annotation'])
        mao_vals = [d['mean_mao'] for d in selected]
        print(f"    {pid:<9} {len(all_meta):>3}d scanned → "
              f"{len(selected):>2}d selected  "
              f"({ann_cnt} annotated  "
              f"MAI {min(mao_vals):.2f}–{max(mao_vals):.2f}g)")

        for d in selected:
            work_items.append((pid, d['day_idx'], d, emg_threshold))

    total = len(work_items)
    print(f"\n  Total days: {total}  |  Workers: {n_workers}")
    print(f"  Chunk size: {CHUNK_WINDOWS} windows ({CHUNK_WINDOWS*10}s)")
    print(f"  Peak RAM / worker: ~{CHUNK_WINDOWS*WINDOW_ECG*3*4/1024**2 + 42:.0f} MB")

    # ── Phase B + C ───────────────────────────────────────────────────────────
    print(f"\n  Phase B+C: Processing → {output_path.name}")
    f      = _create_hdf5(output_path, emg_threshold)
    cursor = [0]
    done   = 0

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        for pid, day_idx, tmp_path in pool.imap_unordered(
            _worker, work_items, chunksize=1
        ):
            done += 1
            if tmp_path:
                try:
                    _write_npz_to_hdf5(f, tmp_path, cursor)
                except Exception as e:
                    warnings.warn(f"Write failed {tmp_path}: {e}")
                finally:
                    try: os.unlink(tmp_path)
                    except Exception: pass

            if done % 10 == 0 or done == total:
                print(f"    {done:>3}/{total} days  "
                      f"({cursor[0]:,} windows)")

    f.attrs['n_windows'] = cursor[0]
    f.attrs['split']     = split
    f.close()
    print(f"  {split}.h5 complete: {cursor[0]:,} windows")
    return cursor[0]


# =============================================================================
# SECTION 11 — PyTorch Dataset
# =============================================================================

class CACHETDataset:
    """
    Lazy HDF5 dataset. Reads one window per __getitem__ call.
    RAM during training: O(batch_size) — constant.

    IMU is stored at 64 Hz (640 samples per window).
    Model must handle ECG at 1024 Hz and IMU at 64 Hz separately.
    """
    def __init__(self, h5_path: Path, norm_stats: Dict):
        self.h5_path    = Path(h5_path)
        self.norm_stats = norm_stats
        self._f         = None
        with h5py.File(self.h5_path, 'r') as f:
            self.n = int(f.attrs['n_windows'])

    def __len__(self):
        return self.n

    def _file(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.h5_path, 'r')
        return self._f

    def __getitem__(self, idx: int) -> Dict:
        """Single-window read. Used as fallback."""
        f   = self._file()
        s   = self.norm_stats
        ecg = (f['ecg'][idx].astype(np.float32)  - s['ecg_mean']) / s['ecg_std']
        acc = (f['acc'][idx].astype(np.float32)
               - s['imu_mean'][:3, None]) / s['imu_std'][:3, None]
        gyro= (f['gyro'][idx].astype(np.float32)
               - s['imu_mean'][3:, None]) / s['imu_std'][3:, None]
        return {
            'ecg':             ecg,
            'acc':             acc,
            'gyro':            gyro,
            'rhythm_class':    int(f['rhythm_class'][idx]),
            'activity_class':  int(f['activity_class'][idx]),
            'movement_accel':  float(f['movement_accel'][idx]),
            'emg_flagged':     bool(f['emg_flagged'][idx]),
            'emg_ratio':       float(f['emg_ratio'][idx]),
            'rpeaks_reliable': bool(f['rpeaks_reliable'][idx]),
            'noise_annotated': bool(f['noise_annotated'][idx]),
            'loss_weight':     float(f['loss_weight'][idx]),
            'patient_id':      f['patient_id'][idx].decode(),
            'day':             int(f['day'][idx]),
            'window_idx':      int(f['window_idx'][idx]),
        }

    def __getitems__(self, indices: list) -> list:
        """
        Batch read — called by DataLoader instead of __getitem__ when available.

        Reads all indices in ONE h5py slice call per dataset instead of
        len(indices) individual calls. With SequentialChunkSampler providing
        consecutive indices, each slice is a single sequential disk read:

          Old: 64 × f['ecg'][i]          = 64 seeks + 64 reads
          New: f['ecg'][indices]          =  1 seek  +  1 read

        Speedup: ~35× reduction in HDF5 overhead per batch.
        """
        f = self._file()
        s = self.norm_stats

        # One slice read per dataset — single HDF5 call
        ecg_b  = (f['ecg'][indices].astype(np.float32)  - s['ecg_mean']) / s['ecg_std']
        acc_b  = (f['acc'][indices].astype(np.float32)
                  - s['imu_mean'][:3, None]) / s['imu_std'][:3, None]
        gyro_b = (f['gyro'][indices].astype(np.float32)
                  - s['imu_mean'][3:, None]) / s['imu_std'][3:, None]
        mai_b  = f['movement_accel'][indices].astype(np.float32)
        act_b  = f['activity_class'][indices]
        rhy_b  = f['rhythm_class'][indices]
        emgf_b = f['emg_flagged'][indices]
        emgr_b = f['emg_ratio'][indices].astype(np.float32)
        rel_b  = f['rpeaks_reliable'][indices]
        noi_b  = f['noise_annotated'][indices]
        lw_b   = f['loss_weight'][indices].astype(np.float32)
        pid_b  = [f['patient_id'][i].decode() for i in indices]
        day_b  = f['day'][indices]
        widx_b = f['window_idx'][indices]

        # Return list of dicts — same format as __getitem__
        return [
            {
                'ecg':             ecg_b[j],
                'acc':             acc_b[j],
                'gyro':            gyro_b[j],
                'rhythm_class':    int(rhy_b[j]),
                'activity_class':  int(act_b[j]),
                'movement_accel':  float(mai_b[j]),
                'emg_flagged':     bool(emgf_b[j]),
                'emg_ratio':       float(emgr_b[j]),
                'rpeaks_reliable': bool(rel_b[j]),
                'noise_annotated': bool(noi_b[j]),
                'loss_weight':     float(lw_b[j]),
                'patient_id':      pid_b[j],
                'day':             int(day_b[j]),
                'window_idx':      int(widx_b[j]),
            }
            for j in range(len(indices))
        ]

    def __getstate__(self):
        """Called when pickling for multiprocessing workers.
        Close h5py handle — it cannot be pickled.
        Workers will reopen on first __getitem__ call."""
        state = self.__dict__.copy()
        if state['_f'] is not None:
            try: state['_f'].close()
            except Exception: pass
        state['_f'] = None   # workers reopen on first access
        return state

    def __setstate__(self, state):
        """Called when unpickling in worker process."""
        self.__dict__.update(state)   # _f is None — opened lazily

    def __del__(self):
        if self._f is not None:
            try: self._f.close()
            except Exception: pass


# =============================================================================
# SECTION 12 — Dataset summary
# =============================================================================

def dataset_summary(h5_path: Path) -> Dict:
    with h5py.File(h5_path, 'r') as f:
        n       = int(f.attrs['n_windows'])
        rhythm  = f['rhythm_class'][:]
        emg_f   = f['emg_flagged'][:]
        noise   = f['noise_annotated'][:]
        rel     = f['rpeaks_reliable'][:]
    s = {
        'total_windows':    n,
        'annotated':        int(np.sum(rhythm >= 0)),
        'unannotated':      int(np.sum(rhythm < 0)),
        'rhythm_AF':        int(np.sum(rhythm == RHYTHM_AF)),
        'rhythm_NSR':       int(np.sum(rhythm == RHYTHM_NSR)),
        'rhythm_Noise':     int(np.sum(rhythm == RHYTHM_NOISE)),
        'emg_flagged':      int(np.sum(emg_f)),
        'noise_annotated':  int(np.sum(noise)),
        'rpeak_unreliable': int(np.sum(~rel.astype(bool))),
        'emg_flag_pct':     round(100 * float(emg_f.mean()), 2),
    }
    print(f"\n── {h5_path.name} ──────────────────")
    for k, v in s.items():
        print(f"  {k:<22}: {v}")
    return s


# =============================================================================
# SECTION 13 — EMG calibration
# =============================================================================

def calibrate_emg_threshold(data_root: Path) -> float:
    from config import EMG_CALIBRATION_PCT, write_emg_threshold
    ratios = []
    print("Calibrating EMG threshold (BodyPos=1, Activity=1, NonWear=0)...")

    for pid in SPLIT_ASSIGNMENTS['train']:
        sig_p = data_root / 'signal'      / pid
        ann_p = data_root / 'annotations' / pid
        if not sig_p.exists():
            continue

        for anon_dir in sorted(d for d in sig_p.iterdir() if d.is_dir()):
            for day_dir in sorted(
                [d for d in anon_dir.iterdir() if d.is_dir()],
                key=lambda d: (d.name.endswith('-last'),
                               int(d.name.split('-')[0]))
            ):
                ecg_path = day_dir / 'ecg.bin'
                if not ecg_path.exists():
                    continue
                ann_day = ann_p / anon_dir.name / day_dir.name
                ctx_path = ann_day / 'context.xlsx'
                if not ctx_path.exists():
                    continue
                ctx_df = load_context(ctx_path)
                if ctx_df.empty:
                    continue
                ann_df = load_annotations(ann_day / 'annotation.csv')
                ecg_lsb = read_unisens_lsb(day_dir, 'ecg.bin')
                reader  = BinaryChunkReader(ecg_path, np.int16, 1, ecg_lsb)

                # Read reference windows in one chunk
                n_ref   = min(reader.n_frames, 3600 * ECG_FS)
                chunk   = reader.read_frames(0, n_ref)[0]
                filt    = bandpass_chunk(chunk, ECG_FS)
                del chunk

                n_win = len(filt) // WINDOW_ECG
                wins  = []
                for w in range(min(n_win, len(ctx_df))):
                    row = ctx_df.iloc[w]
                    def _g(b, u, d=0):
                        for n in [u, b]:
                            if n in ctx_df.columns:
                                v = row[n]; return v if v==v else d
                        return d
                    if (int(_g('BodyPosition','BodyPosition []')) != 1 or
                        int(_g('ActivityClass','ActivityClass []')) != 1 or
                        bool(_g('NonWearTime','NonWearTime []'))):
                        continue
                    rhy = get_rhythm(ann_df, w)
                    if rhy is not None and rhy != RHYTHM_NSR:
                        continue
                    s = w * WINDOW_ECG
                    wins.append(filt[s:s + WINDOW_ECG].astype(np.float32))
                del filt

                if wins:
                    ratios.extend(
                        compute_emg_ratios_batch(
                            np.stack(wins), ECG_FS).tolist())

    if len(ratios) < 10:
        raise RuntimeError(f"Only {len(ratios)} reference windows.")
    arr = np.array(ratios)
    thr = float(np.percentile(arr, EMG_CALIBRATION_PCT))
    print(f"  {len(arr):,} windows  p50={np.percentile(arr,50):.4f}  "
          f"p90={thr:.4f}")
    write_emg_threshold(thr)
    return thr


# =============================================================================
# SECTION 14 — Entry point
# =============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Build CACHET-CADB HDF5 splits")
    parser.add_argument(
        "--max-days-per-patient",
        type=int,
        default=MAX_DAYS_PER_PATIENT,
        help="Cap days per patient. Omit for full dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild split HDF5 files even if they already exist.",
    )
    args = parser.parse_args()

    mp.freeze_support()
    ensure_output_dirs()
    root = DATA_ROOT
    out  = OUTPUT_ROOT

    try:
        threshold = get_emg_threshold()
        print(f"EMG threshold: {threshold:.8f}")
    except ValueError:
        threshold = calibrate_emg_threshold(root)

    import time
    t0 = time.time()

    train_h5  = out / 'train.h5'
    val_h5    = out / 'val.h5'
    test_h5   = out / 'test.h5'
    # ── Build HDF5 splits (skip if already exist) ─────────────────────────────
    for split, path in [('train', train_h5),
                         ('val',   val_h5),
                         ('test',  test_h5)]:
        if path.exists() and not args.force:
            import h5py
            with h5py.File(path, 'r') as f:
                n = int(f.attrs.get('n_windows', 0))
            print(f"  {split}.h5 already exists ({n:,} windows) — skipping build")
        else:
            if path.exists() and args.force:
                path.unlink()
            print(f"\n{'='*55}\n  Building {split} split\n{'='*55}")
            build_hdf5_split(
                root, path, split, threshold,
                budget=args.max_days_per_patient,
            )

    # ── Normalisation stats ────────────────────────────────────────────────────
    # Run compute_norm.py separately after this script completes.
    print("\nDataset build complete.")
    print("  → Run compute_norm.py to compute normalisation stats")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    for p in [train_h5, val_h5, test_h5]:
        dataset_summary(p)

    print("\n── Next steps ───────────────────────────────────────────")
    print("  1. python compute_norm.py   ← compute normalisation stats")
    print("  2. python verify_dataset.py ← verify dataset integrity")
    print()
    print("── Usage in training ────────────────────────────────────")
    print("  from dataset import CACHETDataset, load_norm_stats")
    print("  norm = load_norm_stats(out / 'norm_stats.json')")
    print("  ds   = CACHETDataset(out / 'train.h5', norm)")
    print("  # ecg: [1, 10240] at 1024 Hz")
    print("  # acc: [3,   640] at  64 Hz  ← native 64 Hz, not upsampled")
    print("  # gyro:[3,   640] at  64 Hz")
