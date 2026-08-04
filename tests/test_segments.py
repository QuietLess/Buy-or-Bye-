import pytest

from src.segments import build_segments, segment_probability


def test_segments_are_contiguous_and_cover_probability_range():
    segments = build_segments(0.26)
    assert segments[0].lower == 0.0
    assert segments[-1].upper > 1.0
    assert all(left.upper == right.lower for left, right in zip(segments, segments[1:]))


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.00, "LOW"), (0.05, "LOW"), (0.13, "MEDIUM"),
     (0.26, "HIGH"), (0.50, "HIGH"), (1.00, "HIGH")],
)
def test_probability_boundaries(probability, expected):
    assert segment_probability(probability, 0.26)["code"] == expected


def test_probability_validation():
    with pytest.raises(ValueError, match="0 ile 1"):
        segment_probability(1.01, 0.26)
