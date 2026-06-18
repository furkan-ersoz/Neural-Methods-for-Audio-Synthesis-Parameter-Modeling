"""
test_sequenced_dataset.py
sequenced_dataset.py icindeki prev_params encoding mantigi icin birim testleri.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sequenced_dataset import (
    build_global_cat_map,
    compute_prev_params_dim,
    _build_prev_params,
    _is_frame_active,
)


# Exp003 config.yaml'deki osc_a / osc_b / noise / filter frame'lerinin
# bu testi calistirmaya yetecek kadar daraltilmis hali.
FRAMES = [
    {
        "name": "osc_a",
        "always_active": True,
        "params": [
            "osc_a_waveform", "osc_a_warp_amt", "osc_a_warp_type",
            "osc_a_detune", "osc_a_blend", "osc_a_pitch_coarse",
            "osc_a_pitch_fine", "osc_a_pan", "osc_a_level",
        ],
        "categorical_params": {
            "osc_a_warp_type": ["sync", "fm", "rm", "fold"],
        },
    },
    {
        "name": "osc_b",
        "bypass_column": "bypass_osc_b",
        "params": [
            "osc_b_waveform", "osc_b_warp_amt", "osc_b_warp_type",
            "osc_b_detune", "osc_b_blend", "osc_b_pitch_coarse",
            "osc_b_pitch_fine", "osc_b_pan", "osc_b_level",
            "osc_b_phase", "osc_b_semitone_ofs",
        ],
        "categorical_params": {
            "osc_b_warp_type": ["sync", "fm", "rm", "fold"],
        },
    },
    {
        "name": "noise",
        "bypass_column": "bypass_noise",
        "params": ["noise_color", "noise_level", "noise_stereo"],
        "categorical_params": {
            "noise_color": ["white", "pink", "brown"],
        },
    },
    {
        "name": "filter",
        "bypass_column": "bypass_filter",
        "params": [
            "filter_cutoff", "filter_resonance", "filter_type",
            "filter_drive", "filter_keytrack", "filter_env_amt",
        ],
        "categorical_params": {
            "filter_type": ["lp12", "lp24", "hp12", "hp24", "bp"],
        },
    },
]


# Sentetik 2 satirlik "CSV" — filter frame'i icin prev_params encoding'i test eder.
ROWS = [
    {
        "osc_a_waveform": "0.5", "osc_a_warp_amt": "0.2", "osc_a_warp_type": "rm",
        "osc_a_detune": "0.1", "osc_a_blend": "0.5", "osc_a_pitch_coarse": "0.5",
        "osc_a_pitch_fine": "0.5", "osc_a_pan": "0.5", "osc_a_level": "0.8",
        "bypass_osc_b": "1",
        "osc_b_waveform": "0", "osc_b_warp_amt": "0", "osc_b_warp_type": "sync",
        "osc_b_detune": "0", "osc_b_blend": "0", "osc_b_pitch_coarse": "0",
        "osc_b_pitch_fine": "0", "osc_b_pan": "0", "osc_b_level": "0",
        "osc_b_phase": "0", "osc_b_semitone_ofs": "0",
        "bypass_noise": "0",
        "noise_color": "pink", "noise_level": "0.3", "noise_stereo": "0.5",
        "bypass_filter": "0",
        "filter_cutoff": "0.7", "filter_resonance": "0.2", "filter_type": "hp24",
        "filter_drive": "0.1", "filter_keytrack": "0.0", "filter_env_amt": "0.5",
    },
    {
        "osc_a_waveform": "0.1", "osc_a_warp_amt": "0.0", "osc_a_warp_type": "fold",
        "osc_a_detune": "0.0", "osc_a_blend": "0.0", "osc_a_pitch_coarse": "0.5",
        "osc_a_pitch_fine": "0.5", "osc_a_pan": "0.5", "osc_a_level": "0.6",
        "bypass_osc_b": "0",
        "osc_b_waveform": "0.3", "osc_b_warp_amt": "0.1", "osc_b_warp_type": "fm",
        "osc_b_detune": "0.1", "osc_b_blend": "0.2", "osc_b_pitch_coarse": "0.5",
        "osc_b_pitch_fine": "0.5", "osc_b_pan": "0.5", "osc_b_level": "0.7",
        "osc_b_phase": "0.0", "osc_b_semitone_ofs": "0.5",
        "bypass_noise": "1",
        "noise_color": "white", "noise_level": "0.0", "noise_stereo": "0.0",
        "bypass_filter": "0",
        "filter_cutoff": "0.4", "filter_resonance": "0.6", "filter_type": "lp12",
        "filter_drive": "0.0", "filter_keytrack": "1.0", "filter_env_amt": "0.2",
    },
]


def test_global_cat_map_merges_per_frame_categoricals():
    cat_map = build_global_cat_map(FRAMES)
    assert cat_map["osc_a_warp_type"] == ["sync", "fm", "rm", "fold"]
    assert cat_map["osc_b_warp_type"] == ["sync", "fm", "rm", "fold"]
    assert cat_map["noise_color"] == ["white", "pink", "brown"]
    assert cat_map["filter_type"] == ["lp12", "lp24", "hp12", "hp24", "bp"]


def test_prev_params_dim_matches_export_computation():
    # osc_a: 8 continuous (1 each) + osc_a_warp_type one-hot(4) = 12
    # osc_b: 10 continuous (1 each) + osc_b_warp_type one-hot(4) = 14
    # noise: 2 continuous (1 each) + noise_color one-hot(3) = 5
    assert compute_prev_params_dim(FRAMES, "filter") == 12 + 14 + 5


def test_osc_a_warp_type_onehot_encoding_in_prev_params():
    cat_map = build_global_cat_map(FRAMES)
    prev_frames = FRAMES[:3]  # osc_a, osc_b, noise (frames preceding "filter")

    row = ROWS[0]
    prev_params = _build_prev_params(row, prev_frames, cat_map)

    expected_dim = compute_prev_params_dim(FRAMES, "filter")
    assert prev_params.shape[0] == expected_dim

    # osc_a params order: waveform, warp_amt, warp_type, detune, blend, pitch_coarse,
    # pitch_fine, pan, level -> warp_type is the 3rd param, occupying indices 2..5
    # (one-hot, width 4). "rm" -> index 2 of [sync, fm, rm, fold].
    osc_a_warp_onehot = prev_params[2:6]
    assert (osc_a_warp_onehot == [0.0, 0.0, 1.0, 0.0]).all()


def test_osc_b_bypassed_zeroes_full_block_including_categorical():
    cat_map = build_global_cat_map(FRAMES)
    prev_frames = FRAMES[:3]

    row = ROWS[0]  # bypass_osc_b == "1"
    assert not _is_frame_active(row, FRAMES[1])

    prev_params = _build_prev_params(row, prev_frames, cat_map)
    # osc_b block occupies indices 12..26 (14-dim)
    osc_b_block = prev_params[12:26]
    assert (osc_b_block == 0.0).all()


def test_osc_b_active_warp_type_onehot():
    cat_map = build_global_cat_map(FRAMES)
    prev_frames = FRAMES[:3]

    row = ROWS[1]  # bypass_osc_b == "0", osc_b_warp_type == "fm" -> index 1
    prev_params = _build_prev_params(row, prev_frames, cat_map)

    # osc_b block starts at global index 12 (osc_a block is 12-wide).
    # osc_b params order: waveform, warp_amt, warp_type, ... -> warp_type one-hot
    # occupies block-local indices 2..5, i.e. global 14..18.
    osc_b_warp_onehot = prev_params[14:18]
    assert (osc_b_warp_onehot == [0.0, 1.0, 0.0, 0.0]).all()


def test_frame_dataset_row_filtering_is_deterministic_across_instances():
    """
    build_frame_dataloaders, FrameDataset'i iki kez (training=True/False) olusturup
    aralarinda ortak rastgele indeks bolme uyguluyor. Bu testin gecmesi icin
    _is_frame_active row filtrelemesinin iki cagri arasinda deterministik ve
    ayni sirada olmasi gerekir.
    """
    filter_cfg = FRAMES[3]
    rows_pass_1 = [r for r in ROWS if _is_frame_active(r, filter_cfg)]
    rows_pass_2 = [r for r in ROWS if _is_frame_active(r, filter_cfg)]
    assert rows_pass_1 == rows_pass_2
