import pandas as pd
import pytest

from src.data import EXPECTED_COLUMNS, deduplicate_sessions, get_feature_groups, load_data, validate_data


def valid_frame():
    row = {column: 0 for column in EXPECTED_COLUMNS}
    row.update({"VisitorType": "Returning_Visitor", "Month": "Nov", "Weekend": "False", "Revenue": "True"})
    return pd.DataFrame([row])


def test_validate_data_normalises_booleans():
    result = validate_data(valid_frame())
    assert bool(result.loc[0, "Revenue"]) is True
    assert bool(result.loc[0, "Weekend"]) is False
    assert len(get_feature_groups(result)["numeric"]) == 10


def test_validate_data_rejects_missing_column():
    with pytest.raises(ValueError, match="eksik sütun"):
        validate_data(valid_frame().drop(columns="Revenue"))


def test_deduplicate_sessions_removes_exact_rows_before_split():
    frame = pd.concat([valid_frame(), valid_frame()], ignore_index=True)
    result = deduplicate_sessions(validate_data(frame))
    assert len(result) == 1
    assert result.index.tolist() == [0]


def test_load_data_deduplicates_raw_csv(tmp_path):
    frame = pd.concat([valid_frame(), valid_frame()], ignore_index=True)
    path = tmp_path / "duplicated.csv"
    frame.to_csv(path, index=False)
    assert len(load_data(path)) == 1
