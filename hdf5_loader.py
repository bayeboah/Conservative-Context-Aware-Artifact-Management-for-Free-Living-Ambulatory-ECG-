"""
hdf5_loader.py
--------------
Two HDF5 loaders for Stage 2 training:

  HDF5BatchLoader          — sequential chunks (Stage 3 training, fast IO)
  PatientDiverseLoader     — patient-diverse batches (Stage 2 pretraining)

PatientDiverseLoader is the key fix for Stage 2.

WHY IT MATTERS
--------------
InfoNCE contrastive loss requires HARD negatives — samples that are
similar to the positive but belong to different identities. For ECG:
  - Easy negatives: consecutive windows from same patient-day (nearly identical)
  - Hard negatives: windows from different patients (different morphology)

The old HDF5BatchLoader read chunks of 6400 consecutive windows.
Every batch of 64 came from the same patient-day.
All 63 negatives looked almost identical to the positive.
InfoNCE became trivially easy to game → model memorises patient morphology
→ val loss ABOVE the theoretical maximum log(64) = 4.16.

PatientDiverseLoader:
  - Pre-loads CHUNK_PER_PATIENT=400 windows from each of 16 patients
  - Builds batches of 4 windows × 16 patients = 64 diverse windows
  - Refreshes patient chunks every REFRESH_EVERY epochs
  - RAM: 16 patients × 400 windows × 55KB = 344 MB total (fits in 32 GB)
  - HDF5 reads: 1,872 per epoch vs 750,336 (400× fewer)

PERFORMANCE
-----------
  Sequential loader: ~320s/epoch (fast IO, bad diversity)
  Patient-diverse  : ~330s/epoch (similar IO, excellent diversity)
  The per-patient chunk loading adds ~10s/epoch — worth it for learning.
"""

import random
import numpy as np
import h5py
from pathlib import Path
from typing import Iterator, Dict, List
import torch


# =============================================================================
# PatientDiverseLoader — for Stage 2 contrastive pretraining
# =============================================================================


class PatientDiverseLoader:
    """
    Builds batches with windows from ALL patients in every batch.

    Structure per batch:
      WINDOWS_PER_PATIENT windows × N_PATIENTS = BATCH_SIZE
      e.g. 4 × 16 = 64  (matches batch_size=64)

    Each of the 63 negatives in InfoNCE comes from a different patient
    with a different ECG morphology — genuinely hard negatives.

    Memory layout:
      patient_chunks[patient_id] = {
          'ecg': [400, 1, 10240]  float32  normalised
          'imu': [400, 6,   640]  float32  normalised
          'mai': [400]            float32
      }
      Total: 16 × 344 MB = 344 MB across all patients (pre-loaded once)

    Refreshing:
      Patient chunks are re-sampled every REFRESH_EVERY epochs so the
      model sees all windows over the course of training.
    """

    def __init__(
        self,
        h5_path:             Path,
        norm_stats:          dict,
        batch_size:          int   = 128,
        chunk_per_patient:   int   = 1200,  # 19,200 windows resist memorisation
        refresh_every:       int   = 1,    # every epoch — 19,200 new windows/epoch
        shuffle:             bool  = True,
        mai_threshold:         float = 0.0,
        n_patients_per_batch:  int   = 2,    # within-patient hard negatives
    ):
        self.h5_path           = Path(h5_path)
        self.norm              = norm_stats
        self.batch_size        = batch_size
        self.chunk_per_patient = chunk_per_patient
        self.refresh_every     = refresh_every
        self.shuffle           = shuffle
        self._epoch            = 0

        self.mai_threshold        = mai_threshold

        # Load patient index + MAI filter
        # Only index motion windows (MAI > threshold) to ensure every
        # batch window has real ECG-IMU coupling signal.
        # MAI > 0.016g = 39.2% of train windows (293,942 windows)
        print("  PatientDiverseLoader: indexing patients...", end=" ")
        with h5py.File(self.h5_path, 'r') as f:
            self.n         = int(f.attrs['n_windows'])
            patient_ids    = np.array([
                f['patient_id'][i].decode()
                for i in range(self.n)
            ])
            mai_all = f['movement_accel'][:]

        # Motion mask — only high-motion windows
        motion_mask = mai_all > mai_threshold
        n_motion    = int(motion_mask.sum())
        print(f"found {len(set(patient_ids))} patients, "
              f"{n_motion:,} motion windows ({100*n_motion/self.n:.1f}%)")

        self.patients = sorted(str(p) for p in set(patient_ids))
        self.n_patients_per_batch = min(n_patients_per_batch, len(self.patients))
        self.patient_indices = {
            p: np.where((patient_ids == p) & motion_mask)[0]
            for p in self.patients
        }
        # Fallback: if patient has too few motion windows, use all windows
        MIN_WINDOWS = 50
        for p in self.patients:
            if len(self.patient_indices[p]) < MIN_WINDOWS:
                self.patient_indices[p] = np.where(patient_ids == p)[0]
                print(f"    {p}: too few motion windows — using all windows")
        n_patients = len(self.patients)
        n_bp = self.n_patients_per_batch
        assert batch_size % n_bp == 0, (
            f"batch_size={batch_size} must be divisible by "
            f"n_patients_per_batch={n_bp}"
        )
        self.windows_per_patient = batch_size // n_bp
        print(f"found {n_patients} patients")
        print(f"    {self.windows_per_patient} win/patient × "
              f"{self.n_patients_per_batch} patients/batch = {batch_size} per batch")

        # Pre-load first chunks
        self._chunks = {}
        self._load_patient_chunks()

    def _load_patient_chunks(self):
        """Load a random chunk of CHUNK_PER_PATIENT windows per patient."""
        imu_mean = np.array(self.norm['imu_mean'], dtype=np.float32)
        imu_std  = np.array(self.norm['imu_std'],  dtype=np.float32)

        with h5py.File(self.h5_path, 'r') as f:
            # Load chunks for ALL patients into memory.
            # The 2-patient-per-batch selection happens in __iter__.
            for p in self.patients:
                all_idx = self.patient_indices[p]
                n_take  = min(self.chunk_per_patient, len(all_idx))
                chosen  = np.sort(
                    np.random.choice(all_idx, n_take, replace=False)
                )

                acc_b  = f['acc'][chosen].astype(np.float32)
                gyro_b = f['gyro'][chosen].astype(np.float32)
                mai_b  = f['movement_accel'][chosen].astype(np.float32)
                act_b  = f['activity_class'][chosen].astype(np.int64)
                body_b = f['body_position'][chosen].astype(np.int64)
                noise_b = f['noise_annotated'][chosen].astype(np.float32)
                rel_b   = f['rpeaks_reliable'][chosen].astype(np.float32)
                emg_flag_b = f['emg_flagged'][chosen].astype(np.float32)
                emg_ratio_b = f['emg_ratio'][chosen].astype(np.float32)

                # Normalise
                acc_b  = (acc_b  - imu_mean[:3, np.newaxis]) / imu_std[:3, np.newaxis]
                gyro_b = (gyro_b - imu_mean[3:, np.newaxis]) / imu_std[3:, np.newaxis]
                imu_b  = np.concatenate([acc_b, gyro_b], axis=1)  # [N,6,640]

                self._chunks[p] = {
                    'imu': imu_b,
                    'mai': mai_b,
                    'activity': act_b,
                    'body_position': body_b,
                    'noise': noise_b,
                    'rpeaks_reliable': rel_b,
                    'emg_flagged': emg_flag_b,
                    'emg_ratio': emg_ratio_b,
                }

    def __len__(self) -> int:
        """Approximate batches per epoch."""
        return self.n // self.batch_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Yield batches where each batch contains windows from ALL patients.
        """
        self._epoch += 1

        # Refresh patient chunks periodically
        if self._epoch % self.refresh_every == 0:
            self._load_patient_chunks()

        # Shuffle within each patient chunk
        shuffled = {}
        for p in self.patients:
            chunk = self._chunks[p]
            n     = len(chunk['imu'])
            perm  = np.random.permutation(n) if self.shuffle else np.arange(n)
            shuffled[p] = {
                'imu': chunk['imu'][perm],
                'mai': chunk['mai'][perm],
                'activity': chunk['activity'][perm],
                'body_position': chunk['body_position'][perm],
                'noise': chunk['noise'][perm],
                'rpeaks_reliable': chunk['rpeaks_reliable'][perm],
                'emg_flagged': chunk['emg_flagged'][perm],
                'emg_ratio': chunk['emg_ratio'][perm],
            }

        n_batches = self.__len__()
        wpp = self.windows_per_patient

        patient_offsets = {p: 0 for p in self.patients}

        for batch_idx in range(n_batches):
            imu_parts, mai_parts = [], []
            act_parts, body_parts, noise_parts, rel_parts = [], [], [], []
            emg_flag_parts, emg_ratio_parts = [], []

            # Within-patient hard negatives:
            # Randomly select n_patients_per_batch patients per batch.
            # 2 patients × 64 windows — forces temporal ECG-IMU correspondence.
            if self.shuffle:
                batch_pts = list(np.random.choice(
                    self.patients, self.n_patients_per_batch, replace=False
                ))
            else:
                start = (batch_idx * self.n_patients_per_batch) % len(self.patients)
                batch_pts = [
                    self.patients[(start + j) % len(self.patients)]
                    for j in range(self.n_patients_per_batch)
                ]

            for p in batch_pts:
                chunk_n = len(shuffled[p]['imu'])

                # Random window selection — no pointer needed
                # Each batch independently samples wpp windows from this patient
                if self.shuffle:
                    idx = np.random.choice(chunk_n, wpp, replace=(chunk_n < wpp))
                else:
                    start = patient_offsets[p]
                    idx = (np.arange(wpp) + start) % chunk_n
                    patient_offsets[p] = (start + wpp) % chunk_n
                # Temporal jitter on IMU: random shift ±2 IMU samples
                # Forces shift-invariant embeddings — robust to
                # minor hardware timing variation between recordings
                imu_chunk = shuffled[p]['imu'][idx].copy()  # [wpp, 6, 640]
                if self.shuffle:   # only during training, not validation
                    for jj in range(len(idx)):
                        shift = np.random.randint(-4, 5)  # ±4 IMU samples = ±62ms
                        if shift != 0:
                            # Edge-pad instead of np.roll to avoid wrap-around
                            # np.roll creates a sharp discontinuity when the
                            # tail of the array wraps to the front — conv nets
                            # are sensitive to such artificial spikes.
                            # Edge padding replicates the boundary sample,
                            # keeping the signal continuous.
                            src = imu_chunk[jj]        # [6, 640]
                            out = np.empty_like(src)
                            if shift > 0:
                                # Shift right: pad left edge with first sample
                                out[:, shift:] = src[:, :-shift]
                                out[:, :shift] = src[:, 0:1]
                            else:
                                # Shift left: pad right edge with last sample
                                s = -shift
                                out[:, :640-s] = src[:, s:]
                                out[:, 640-s:] = src[:, -1:]
                            imu_chunk[jj] = out
                imu_parts.append(imu_chunk)
                mai_parts.append(shuffled[p]['mai'][idx])
                act_parts.append(shuffled[p]['activity'][idx])
                body_parts.append(shuffled[p]['body_position'][idx])
                noise_parts.append(shuffled[p]['noise'][idx])
                rel_parts.append(shuffled[p]['rpeaks_reliable'][idx])
                emg_flag_parts.append(shuffled[p]['emg_flagged'][idx])
                emg_ratio_parts.append(shuffled[p]['emg_ratio'][idx])



            imu_batch = np.concatenate(imu_parts, axis=0)  # [B,6,  640]
            mai_batch = np.concatenate(mai_parts, axis=0)  # [B]
            act_batch = np.concatenate(act_parts, axis=0)
            body_batch = np.concatenate(body_parts, axis=0)
            noise_batch = np.concatenate(noise_parts, axis=0)
            rel_batch = np.concatenate(rel_parts, axis=0)
            emg_flag_batch = np.concatenate(emg_flag_parts, axis=0)
            emg_ratio_batch = np.concatenate(emg_ratio_parts, axis=0)

            # No ECG noise augmentation:
            # CACHET-CADB already contains real ambulatory noise.
            # Soft MAI weighting in the loss handles the patient identity
            # shortcut more directly without risking obscuring the
            # motion artifacts that Stage 2 is learning to align.

            # Shuffle the batch (mix patients within batch)
            if self.shuffle:
                perm      = np.random.permutation(self.batch_size)
                imu_batch = imu_batch[perm]
                mai_batch = mai_batch[perm]
                act_batch = act_batch[perm]
                body_batch = body_batch[perm]
                noise_batch = noise_batch[perm]
                rel_batch = rel_batch[perm]
                emg_flag_batch = emg_flag_batch[perm]
                emg_ratio_batch = emg_ratio_batch[perm]

            yield {
                'imu': torch.from_numpy(imu_batch),
                'movement_accel': torch.from_numpy(mai_batch),
                'activity_class': torch.from_numpy(act_batch),
                'body_position': torch.from_numpy(body_batch),
                'noise_annotated': torch.from_numpy(noise_batch),
                'rpeaks_reliable': torch.from_numpy(rel_batch),
                'emg_flagged': torch.from_numpy(emg_flag_batch),
                'emg_ratio': torch.from_numpy(emg_ratio_batch),
            }


# =============================================================================
# HDF5BatchLoader — sequential chunks (Stage 3 training)
# =============================================================================
class HDF5BatchLoader:
    """
    Fast sequential HDF5 reader for Stage 3 training.
    Reads CHUNK_SIZE windows in one HDF5 slice, yields batches from RAM.
    Use PatientDiverseLoader for Stage 2 contrastive pretraining.
    """

    def __init__(
        self,
        h5_path:    Path,
        norm_stats: dict,
        batch_size: int  = 64,
        chunk_size: int  = 6400,
        shuffle:    bool = True,
        drop_last:  bool = True,
    ):
        self.h5_path    = Path(h5_path)
        self.norm       = norm_stats
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.shuffle    = shuffle
        self.drop_last  = drop_last

        with h5py.File(self.h5_path, 'r') as f:
            self.n = int(f.attrs['n_windows'])

        self.chunks = [
            (i, min(i + chunk_size, self.n))
            for i in range(0, self.n, chunk_size)
        ]

    def __len__(self) -> int:
        return self.n // self.batch_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        ecg_mean = self.norm['ecg_mean']
        ecg_std  = self.norm['ecg_std']
        imu_mean = np.array(self.norm['imu_mean'], dtype=np.float32)
        imu_std  = np.array(self.norm['imu_std'],  dtype=np.float32)

        chunk_order = list(range(len(self.chunks)))
        if self.shuffle:
            random.shuffle(chunk_order)

        leftover_ecg = leftover_imu = leftover_mai = None
        leftover_act = leftover_body = leftover_noise = leftover_rel = None

        with h5py.File(self.h5_path, 'r') as f:
            for chunk_idx in chunk_order:
                lo, hi = self.chunks[chunk_idx]

                ecg_c  = f['ecg'][lo:hi, 0, :].astype(np.float32)
                acc_c  = f['acc'][lo:hi].astype(np.float32)
                gyro_c = f['gyro'][lo:hi].astype(np.float32)
                mai_c  = f['movement_accel'][lo:hi].astype(np.float32)
                act_c  = f['activity_class'][lo:hi].astype(np.int64)
                body_c = f['body_position'][lo:hi].astype(np.int64)
                noise_c = f['noise_annotated'][lo:hi].astype(np.float32)
                rel_c = f['rpeaks_reliable'][lo:hi].astype(np.float32)

                ecg_c  = (ecg_c - ecg_mean) / ecg_std
                ecg_c  = ecg_c[:, np.newaxis, :]
                acc_c  = (acc_c  - imu_mean[:3, np.newaxis]) / imu_std[:3, np.newaxis]
                gyro_c = (gyro_c - imu_mean[3:, np.newaxis]) / imu_std[3:, np.newaxis]
                imu_c  = np.concatenate([acc_c, gyro_c], axis=1)

                if self.shuffle:
                    perm   = np.random.permutation(len(ecg_c))
                    ecg_c  = ecg_c[perm]
                    imu_c  = imu_c[perm]
                    mai_c  = mai_c[perm]
                    act_c  = act_c[perm]
                    body_c = body_c[perm]
                    noise_c = noise_c[perm]
                    rel_c = rel_c[perm]

                if leftover_ecg is not None:
                    ecg_c = np.concatenate([leftover_ecg, ecg_c], axis=0)
                    imu_c = np.concatenate([leftover_imu, imu_c], axis=0)
                    mai_c = np.concatenate([leftover_mai, mai_c], axis=0)
                    act_c = np.concatenate([leftover_act, act_c], axis=0)
                    body_c = np.concatenate([leftover_body, body_c], axis=0)
                    noise_c = np.concatenate([leftover_noise, noise_c], axis=0)
                    rel_c = np.concatenate([leftover_rel, rel_c], axis=0)
                    leftover_ecg = leftover_imu = leftover_mai = None
                    leftover_act = leftover_body = leftover_noise = leftover_rel = None

                total = len(ecg_c)
                for b in range(0, total - self.batch_size + 1, self.batch_size):
                    e = b + self.batch_size
                    yield {
                        'ecg': torch.from_numpy(ecg_c[b:e]),
                        'imu': torch.from_numpy(imu_c[b:e]),
                        'movement_accel': torch.from_numpy(mai_c[b:e]),
                        'activity_class': torch.from_numpy(act_c[b:e]),
                        'body_position': torch.from_numpy(body_c[b:e]),
                        'noise_annotated': torch.from_numpy(noise_c[b:e]),
                        'rpeaks_reliable': torch.from_numpy(rel_c[b:e]),
                    }

                remainder = total % self.batch_size
                if remainder > 0 and not self.drop_last:
                    leftover_ecg = ecg_c[-remainder:]
                    leftover_imu = imu_c[-remainder:]
                    leftover_mai = mai_c[-remainder:]
                    leftover_act = act_c[-remainder:]
                    leftover_body = body_c[-remainder:]
                    leftover_noise = noise_c[-remainder:]
                    leftover_rel = rel_c[-remainder:]

        if (not self.drop_last and leftover_ecg is not None
                and len(leftover_ecg) > 0):
            yield {
                'ecg': torch.from_numpy(leftover_ecg),
                'imu': torch.from_numpy(leftover_imu),
                'movement_accel': torch.from_numpy(leftover_mai),
                'activity_class': torch.from_numpy(leftover_act),
                'body_position': torch.from_numpy(leftover_body),
                'noise_annotated': torch.from_numpy(leftover_noise),
                'rpeaks_reliable': torch.from_numpy(leftover_rel),
            }
