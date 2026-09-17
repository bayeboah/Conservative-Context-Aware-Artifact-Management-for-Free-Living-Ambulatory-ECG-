"""
split_config.py
---------------
Patient-level split configuration for CACHET-CADB.

Split constraints:
  - Patient-level: no patient spans multiple splits
  - Site-balanced: both Danish and Indian patients in all splits
  - Rhythm/context-balanced using observed rhythm annotations and NSR-only records
  - Test set covers both sites for cross-site generalisation evaluation

Patient metadata sourced from Table 8 and Section 3 of the paper:
  - 7 Danish patients  (PNSR-1, PNSR-3, PNSR-4 + 4 from P-series)
  - 17 Indian patients (remaining P-series)
  - 11 patients with documented AF history
  - PNSR patients: normal sinus rhythm only (hardest negatives)

Important: the public paper reports the total number of patients with documented
AF history but does not publish a per-patient AF-history mapping. The `has_af`
field below is therefore retained only as an inferred descriptive label from an
earlier registry pass; it should not be treated as source-verified clinical
metadata or used as a hard analytic endpoint.
"""

import json
from pathlib import Path

# ── Patient registry ───────────────────────────────────────────────────────────
# Each entry: patient_id → {site, has_af_history, is_nsr_only, days}
# Days from Table 8. Site assignment inferred from paper (7 DK, 17 IN).
# Paper-level AF-history count: 11 patients with documented AF episodes.
# Per-patient AF-history mapping: not available in the publication text.

PUBLICATION_AF_HISTORY_COUNT = 11
AF_HISTORY_LABEL_STATUS = "inferred_not_source_verified"

PATIENT_REGISTRY = {
    # Danish patients (site = DK)
    "PNSR-1": {"site": "DK", "has_af": False, "nsr_only": True,  "days": 1},
    "PNSR-3": {"site": "DK", "has_af": False, "nsr_only": True,  "days": 1},
    "PNSR-4": {"site": "DK", "has_af": False, "nsr_only": True,  "days": 2},
    "P6":     {"site": "DK", "has_af": True,  "nsr_only": False, "days": 12},
    "P8":     {"site": "DK", "has_af": True,  "nsr_only": False, "days": 22},
    "P12":    {"site": "DK", "has_af": True,  "nsr_only": False, "days": 14},
    "P13":    {"site": "DK", "has_af": False, "nsr_only": False, "days": 19},

    # Indian patients (site = IN)
    "P1":     {"site": "IN", "has_af": True,  "nsr_only": False, "days": 12},
    "P2":     {"site": "IN", "has_af": False, "nsr_only": False, "days": 7},
    "P3":     {"site": "IN", "has_af": True,  "nsr_only": False, "days": 15},
    "P4":     {"site": "IN", "has_af": False, "nsr_only": False, "days": 8},
    "P5":     {"site": "IN", "has_af": True,  "nsr_only": False, "days": 8},
    "P7":     {"site": "IN", "has_af": False, "nsr_only": False, "days": 12},
    "P9":     {"site": "IN", "has_af": True,  "nsr_only": False, "days": 12},
    "P10":    {"site": "IN", "has_af": True,  "nsr_only": False, "days": 16},
    "P11":    {"site": "IN", "has_af": True,  "nsr_only": False, "days": 16},
    "P14":    {"site": "IN", "has_af": True,  "nsr_only": False, "days": 12},
    "P15":    {"site": "IN", "has_af": True,  "nsr_only": False, "days": 14},
    "P16":    {"site": "IN", "has_af": False, "nsr_only": False, "days": 17},
    "P17":    {"site": "IN", "has_af": False, "nsr_only": False, "days": 15},
    "P18":    {"site": "IN", "has_af": False, "nsr_only": False, "days": 5},
    "P19":    {"site": "IN", "has_af": False, "nsr_only": False, "days": 4},
    "P21":    {"site": "IN", "has_af": True,  "nsr_only": False, "days": 8},
    "P23":    {"site": "IN", "has_af": False, "nsr_only": False, "days": 7},
}

# ── Split assignment ───────────────────────────────────────────────────────────
# Rationale per split:
#
# TEST (4 patients — held out entirely until final evaluation):
#   P3   DK · AF · long recording (15d) → primary cross-site test case
#   P9   IN · AF · high noise (29.6%)   → stress test for denoiser
#   P16  IN · no AF · moderate length   → clean NSR negative case
#   PNSR-3 DK · NSR-only · high noise  → hardest negative (NSR + noise)
#
# VALIDATION (4 patients — used for early stopping and hyperparam tuning):
#   P8   DK · AF · longest recording (22d) → rich validation signal
#   P10  IN · AF · highest noise (31.7%)   → noise distribution coverage
#   P17  IN · no AF · 15d                  → balanced negative
#   PNSR-4 DK · NSR-only · 2d             → short NSR reference
#
# TRAIN (16 patients — everything else):
#   All remaining patients

SPLIT_ASSIGNMENTS = {
    "test": ["P3", "P9", "P16", "PNSR-3"],
    "val":  ["P8", "P10", "P17", "PNSR-4"],
    "train": [
        "PNSR-1",
        "P1", "P2", "P4", "P5", "P6", "P7",
        "P11", "P12", "P13", "P14", "P15",
        "P18", "P19", "P21", "P23",
    ],
}


def _load_context_stratified_manifest():
    manifest = Path(__file__).resolve().parent / "output" / "context_stratified_split.json"
    if not manifest.exists():
        return None
    with open(manifest, "r") as f:
        data = json.load(f)
    splits = data.get("splits", {})
    required = {"train", "val", "test"}
    if set(splits) != required:
        raise ValueError(f"Invalid split manifest keys in {manifest}: {sorted(splits)}")
    return {k: list(v) for k, v in splits.items()}


_manifest_splits = _load_context_stratified_manifest()
if _manifest_splits is not None:
    SPLIT_ASSIGNMENTS = _manifest_splits

# Sanity check: every patient assigned exactly once
def validate_splits() -> None:
    all_assigned = (
        SPLIT_ASSIGNMENTS["train"]
        + SPLIT_ASSIGNMENTS["val"]
        + SPLIT_ASSIGNMENTS["test"]
    )
    all_known = set(PATIENT_REGISTRY.keys())
    assert set(all_assigned) == all_known, (
        f"Split mismatch.\n"
        f"  Missing : {all_known - set(all_assigned)}\n"
        f"  Extra   : {set(all_assigned) - all_known}"
    )
    assert len(all_assigned) == len(set(all_assigned)), \
        "Duplicate patient assignment detected"

    for split_name, patients in SPLIT_ASSIGNMENTS.items():
        sites    = {PATIENT_REGISTRY[p]["site"] for p in patients}
        af_count = sum(PATIENT_REGISTRY[p]["has_af"] for p in patients)
        nsr_only = sum(PATIENT_REGISTRY[p]["nsr_only"] for p in patients)
        days     = sum(PATIENT_REGISTRY[p]["days"] for p in patients)
        print(
            f"{split_name:6s}  {len(patients):2d} patients  "
            f"sites={sites}  AF={af_count}  NSR-only={nsr_only}  "
            f"total_days={days}"
        )

    registry_af = sum(v["has_af"] for v in PATIENT_REGISTRY.values())
    if registry_af != PUBLICATION_AF_HISTORY_COUNT:
        print(
            "\nWarning: inferred registry has_af count "
            f"({registry_af}) differs from publication-level AF-history count "
            f"({PUBLICATION_AF_HISTORY_COUNT}). The publication does not provide "
            "a per-patient AF-history mapping; treat has_af as descriptive, "
            "not source-verified."
        )

    print("\nSplit validation passed.")


if __name__ == "__main__":
    validate_splits()
