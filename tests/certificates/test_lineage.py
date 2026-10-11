from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from provable.certificates import (ColumnDeclaration, ColumnFreshnessError, Lineage, Slice,
                      assert_column_freshness, check_columns, clock_overlaps,
                      hereditary_union, identity_fresh, identity_overlap,
                      slice_from_frame, source_fingerprint)

UTC = timezone.utc


def hourly_frame(tag, day, n=24, seed=0):
    rng = np.random.default_rng(seed)
    start = pd.date_range(datetime(2026, 9, day, tzinfo=UTC), periods=n, freq="h")
    return pd.DataFrame({"start_utc": start, "end_utc": start + pd.Timedelta(hours=1), "x": rng.normal(size=n), "tag": tag})


def test_source_fingerprint_content_sensitivity():
    a = hourly_frame("dataset_1", 1)
    b = a.copy()
    assert source_fingerprint(a) == source_fingerprint(b)
    b.loc[3, "x"] = b.loc[3, "x"] + 1.0
    assert source_fingerprint(a) != source_fingerprint(b)


def test_identity_overlap_and_all_sentinel():
    f = hourly_frame("dataset_1", 1)
    s_a = slice_from_frame(f, "dataset_1", rows=range(0, 10))
    s_b = slice_from_frame(f, "dataset_1", rows=range(8, 12))
    s_c = slice_from_frame(f, "dataset_1", rows=range(12, 20))
    s_all = slice_from_frame(f, "dataset_1", rows=None)
    ov = identity_overlap(Lineage.of(s_a), Lineage.of(s_b))
    assert len(ov) == 1 and ov[0]["rows"] == [8, 9]
    assert identity_fresh(Lineage.of(s_a), Lineage.of(s_c))
    ov_all = identity_overlap(Lineage.of(s_all), Lineage.of(s_c))
    assert len(ov_all) == 1 and ov_all[0]["whole_source"]
    g = hourly_frame("dataset_1", 2, seed=1)
    s_g = slice_from_frame(g, "dataset_1", rows=range(0, 10))
    assert identity_fresh(Lineage.of(s_a), Lineage.of(s_g))


def test_union_monotone_property():
    rng = np.random.default_rng(5)
    f = hourly_frame("dataset_1", 1)
    g = hourly_frame("dataset_2", 1, seed=2)
    for _ in range(200):
        ra = frozenset(rng.choice(24, rng.integers(1, 10), replace=False).tolist())
        rb = frozenset(rng.choice(24, rng.integers(1, 10), replace=False).tolist())
        rs = frozenset(rng.choice(24, rng.integers(1, 10), replace=False).tolist())
        a = Lineage.of(slice_from_frame(f, "dataset_1", ra))
        b = Lineage.of(slice_from_frame(g, "dataset_2", rb))
        S = Lineage.of(slice_from_frame(f, "dataset_1", rs))
        fresh_union = identity_fresh(a.union(b), S)
        assert fresh_union == (identity_fresh(a, S) and identity_fresh(b, S))
        assert identity_fresh(a, S) == identity_fresh(S, a)


def test_hereditary_union_contains_all():
    f = hourly_frame("dataset_1", 1)
    a = Lineage.of(slice_from_frame(f, "dataset_1", range(0, 5)))
    b = Lineage.of(slice_from_frame(f, "dataset_1", range(5, 9)))
    c = Lineage.of(slice_from_frame(f, "dataset_1", range(9, 12)))
    h = hereditary_union(a, [b, c])
    for lin in (a, b, c):
        for sl in lin.slices:
            assert sl in h


def test_clock_overlap_cross_tag_only():
    f = hourly_frame("dataset_1", 1)
    g = hourly_frame("dataset_2", 1, seed=3)
    h = hourly_frame("dataset_2", 2, seed=4)
    same_hours = clock_overlaps(Lineage.of(slice_from_frame(f, "dataset_1")), Lineage.of(slice_from_frame(g, "dataset_2")))
    assert len(same_hours) == 1
    lo, hi = same_hours[0]["window"]
    assert lo < hi and same_hours[0]["tags"] == ("dataset_1", "dataset_2")
    assert clock_overlaps(Lineage.of(slice_from_frame(f, "dataset_1")), Lineage.of(slice_from_frame(h, "dataset_2"))) == []
    f2 = hourly_frame("dataset_1", 1, seed=9)
    assert clock_overlaps(Lineage.of(slice_from_frame(f, "dataset_1")), Lineage.of(slice_from_frame(f2, "dataset_1"))) == []


def test_column_lag_rule():
    f = hourly_frame("dataset_1", 3)
    targets = [Slice(source_fingerprint(f), "dataset_1", frozenset([h]), datetime(2026, 9, 3, h, tzinfo=UTC), datetime(2026, 9, 3, h + 1, tzinfo=UTC)) for h in range(4)]
    gate = datetime(2026, 9, 2, 10, 0, tzinfo=UTC)
    lag0 = ColumnDeclaration(column="realized_cost", dataset_tag="dataset_1", lag=timedelta(0), outcome_derived=True)
    lag2 = ColumnDeclaration(column="realized_cost_lag2d", dataset_tag="dataset_1", lag=timedelta(days=2), outcome_derived=True)
    fcst = ColumnDeclaration(column="da_forecast", dataset_tag="dataset_1", lag=timedelta(0), outcome_derived=False)
    v = check_columns([lag0], targets, gate)
    assert v and all(x["reason"] == "own outcome rows in column lineage" for x in v)
    assert check_columns([lag2], targets, gate) == []
    assert check_columns([fcst], targets, gate) == []
    with pytest.raises(ColumnFreshnessError):
        assert_column_freshness([lag0, lag2], targets, gate)
    assert_column_freshness([lag2, fcst], targets, gate)


def test_column_post_gate_rule():
    f = hourly_frame("dataset_1", 5)
    targets = [Slice(source_fingerprint(f), "dataset_1", frozenset([12]), datetime(2026, 9, 5, 12, tzinfo=UTC), datetime(2026, 9, 5, 13, tzinfo=UTC))]
    gate = datetime(2026, 9, 4, 10, 0, tzinfo=UTC)
    lag_future = ColumnDeclaration(column="tomorrow_outturn", dataset_tag="FR", lag=timedelta(hours=-6), outcome_derived=True)
    v = check_columns([lag_future], targets, gate)
    assert v and v[0]["reason"] == "outcome row delivers after gate time"
    explicit = ColumnDeclaration(column="late_obs", dataset_tag="dataset_1", rows=Lineage.of(Slice(source_fingerprint(f), "dataset_1", frozenset([20]), datetime(2026, 9, 5, 20, tzinfo=UTC), datetime(2026, 9, 5, 21, tzinfo=UTC))), outcome_derived=True)
    v2 = check_columns([explicit], targets, gate)
    assert any(x["reason"] == "outcome row delivers after gate time" for x in v2)


def test_column_explicit_rows_own_outcome():
    f = hourly_frame("dataset_1", 6)
    sid = source_fingerprint(f)
    target = Slice(sid, "dataset_1", frozenset([7]), datetime(2026, 9, 6, 7, tzinfo=UTC), datetime(2026, 9, 6, 8, tzinfo=UTC))
    gate = datetime(2026, 9, 7, 0, 0, tzinfo=UTC)
    col = ColumnDeclaration(column="same_row_outcome", dataset_tag="dataset_1", rows=Lineage.of(Slice(sid, "dataset_1", frozenset([7]), target.t0, target.t1)), outcome_derived=True)
    v = check_columns([col], [target], gate)
    assert any(x["reason"] == "own outcome rows in column lineage" for x in v)
