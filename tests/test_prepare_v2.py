import numpy as np

from prepare_v2 import encode_factors_v2, get_feature_names_v2, FEATURE_DIM_V2


def test_feature_dim_matches_names():
    names = get_feature_names_v2()
    assert len(names) == FEATURE_DIM_V2


def test_encode_shape(sample_bean):
    features = encode_factors_v2(sample_bean)
    assert features.shape == (FEATURE_DIM_V2,)


def test_encode_returns_finite(sample_bean):
    features = encode_factors_v2(sample_bean)
    assert np.all(np.isfinite(features))


def test_encode_altitude_normalized(sample_bean):
    """Altitude should be normalized to 0-1 range."""
    features = encode_factors_v2(sample_bean)
    names = get_feature_names_v2()
    alt_idx = names.index("altitude_m")
    assert 0 <= features[alt_idx] <= 1


def test_encode_variety_onehot(sample_bean):
    """Variety one-hot should have exactly one 1."""
    features = encode_factors_v2(sample_bean)
    names = get_feature_names_v2()
    variety_indices = [i for i, n in enumerate(names) if n.startswith("variety_")]
    variety_values = features[variety_indices]
    assert sum(variety_values) == 1.0
