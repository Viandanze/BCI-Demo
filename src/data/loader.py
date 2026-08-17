"""
PhysioNet Motor Movement/Imagery Dataset Loader
Downloads and loads the PhysioNet MI dataset using MNE
"""

import os
import logging
from typing import Tuple, List, Optional, Union, Dict
from pathlib import Path

import mne
import numpy as np

logger = logging.getLogger(__name__)


# PhysioNet Motor Imagery event mapping
EVENT_ID = {
    'T1': 1,  # Left hand (motor imagery)
    'T2': 2,  # Right hand
    'T3': 3,  # Feet
    'T4': 4,  # Rest
}

BINARY_EVENT_ID = {
    'left_hand': 1,
    'right_hand': 2,
}

TERNARY_EVENT_ID = {
    'left_hand': 1,
    'right_hand': 2,
    'feet': 3,
}


def load_physionet_data(
    data_path: Optional[str] = None,
    subjects: Optional[List[int]] = None,
    runs: Optional[List[int]] = None,
    verbose: bool = False,
) -> Dict[int, mne.io.Raw]:
    """Load PhysioNet Motor Movement/Imagery dataset using mne.datasets.eegbci."""
    if data_path is None:
        data_path = "./data/"
    if subjects is None:
        subjects = list(range(1, 109))
    if runs is None:
        runs = [4, 5, 6]
    if not all(1 <= s <= 109 for s in subjects):
        raise ValueError(f"Subject IDs must be between 1 and 109, got: {subjects}")
    if not all(1 <= r <= 14 for r in runs):
        raise ValueError(f"Run numbers must be between 1 and 14, got: {runs}")
    logger.info(f"Loading PhysioNet data for {len(subjects)} subjects, runs {runs}")
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    raw_dict = {}
    for subject_id in subjects:
        try:
            logger.info(f"Loading subject {subject_id:03d}...")
            try:
                raw = _load_subject_local(data_path, subject_id, runs)
            except FileNotFoundError:
                logger.info(f"Subject {subject_id} not found locally, downloading...")
                raw = _download_and_load_subject(data_path, subject_id, runs, verbose)
            if raw is not None and len(raw) > 0:
                raw_dict[subject_id] = raw
                logger.info(f"Subject {subject_id} loaded successfully, "
                           f"duration: {raw.times[-1]:.1f}s, "
                           f"channels: {len(raw.ch_names)}")
            else:
                logger.warning(f"Subject {subject_id} returned empty data, skipping")
        except Exception as e:
            logger.warning(f"Failed to load subject {subject_id}: {e}")
            continue
    logger.info(f"Successfully loaded {len(raw_dict)} subjects")
    if len(raw_dict) == 0:
        raise RuntimeError("No subjects could be loaded. Check your data path and internet connection.")
    return raw_dict


def _load_subject_local(data_path: Path, subject_id: int, runs: List[int]) -> mne.io.Raw:
    """Try to load subject from local EDF files (eegbci cache format)."""
    from mne.datasets import eegbci
    raw_fnames = eegbci.load_data(
        subject_id, runs, path=str(data_path), update_path=False, verbose=False
    )
    if not raw_fnames:
        raise FileNotFoundError(f"No local files for subject {subject_id}")
    raw_list = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
    if len(raw_list) == 1:
        return raw_list[0]
    return mne.concatenate_raws(raw_list, preload=True)


def _download_and_load_subject(
    data_path: Path, subject_id: int, runs: List[int], verbose: bool = False
) -> mne.io.Raw:
    """Download and load subject data from PhysioNet via mne.datasets.eegbci."""
    from mne.datasets import eegbci
    from mne.io import concatenate_raws, read_raw_edf
    from mne.channels import make_standard_montage
    logger.info(f"Downloading subject {subject_id} via eegbci...")
    raw_fnames = eegbci.load_data(
        subject_id, runs, path=str(data_path), update_path=True, verbose=verbose
    )
    if not raw_fnames:
        raise FileNotFoundError(f"No files downloaded for subject {subject_id}")
    raw_list = [read_raw_edf(f, preload=True, verbose=verbose) for f in raw_fnames]
    raw = concatenate_raws(raw_list) if len(raw_list) > 1 else raw_list[0]
    eegbci.standardize(raw)
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing='ignore')
    logger.info(f"Subject {subject_id} downloaded: {len(raw.ch_names)} ch, {raw.times[-1]:.1f}s")
    return raw


def get_subject_data(
    raw: mne.io.Raw,
    tmin: float = -1.0,
    tmax: float = 4.0,
    event_id: Optional[Dict[str, int]] = None,
    baseline: Optional[Tuple[float, float]] = (-1.0, 0.0),
    picks: Optional[Union[str, List[str]]] = None,
) -> Tuple[mne.Epochs, np.ndarray, np.ndarray]:
    """Extract epochs from Raw object for motor imagery classification."""
    if event_id is None:
        event_id = BINARY_EVENT_ID
    # events_from_annotations always returns (events, event_id_map)
    events, event_id_map = mne.events_from_annotations(raw, verbose=False)
    # Map annotation names to our standard IDs
    new_events = events.copy()
    for annot_name, raw_id in event_id_map.items():
        if annot_name in ('T1', 'left_hand'):
            new_events[events[:, 2] == raw_id, 2] = 1
        elif annot_name in ('T2', 'right_hand'):
            new_events[events[:, 2] == raw_id, 2] = 2
        elif annot_name in ('T3', 'feet'):
            new_events[events[:, 2] == raw_id, 2] = 3
    # Filter to only the events we want
    valid_ids = set(event_id.values())
    mask = np.isin(new_events[:, 2], list(valid_ids))
    new_events = new_events[mask]
    if len(new_events) == 0:
        raise ValueError(f"No matching events found. Available: {event_id_map}, Looking for: {event_id}")
    if picks is not None:
        try:
            raw = raw.pick(picks)
        except Exception as e:
            logger.warning(f"Channel selection failed: {e}, using all channels")
    epochs = mne.Epochs(
        raw, events=new_events, event_id=event_id,
        tmin=tmin, tmax=tmax, baseline=baseline,
        preload=True, reject=None, verbose=False,
    )
    X = epochs.get_data()
    y = epochs.events[:, -1]
    unique_labels = sorted(list(event_id.values()))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in y])
    logger.info(f"Created {len(X)} epochs: shape {X.shape}, labels distribution: {np.bincount(y)}")
    return epochs, X, y


def create_train_test_split(
    X: np.ndarray, y: np.ndarray,
    test_size: float = 0.2, random_state: int = 42, stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training and testing sets."""
    from sklearn.model_selection import train_test_split
    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param,
    )
    logger.info(f"Train/test split: {len(X_train)}/{len(X_test)} samples")
    logger.info(f"Train labels: {np.bincount(y_train)}, Test labels: {np.bincount(y_test)}")
    return X_train, X_test, y_train, y_test


__all__ = [
    "load_physionet_data", "get_subject_data", "create_train_test_split",
    "EVENT_ID", "BINARY_EVENT_ID", "TERNARY_EVENT_ID",
]
