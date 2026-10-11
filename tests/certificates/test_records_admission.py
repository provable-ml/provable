from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from provable.certificates import (Certificate, CertificateKind, FreshnessError, Guarantee, GuaranteeKind, Ledger, Lineage, MISCALIBRATED, Verdict, admit, detect_screen_then_gate, issue, refines, slice_from_frame)

UTC = timezone.utc


def frame(tag, day, n=24, seed=0):
    rng = np.random.default_rng(seed)
    start = pd.date_range(datetime(2026, 9, day, tzinfo=UTC), periods=n, freq="h")
    return pd.DataFrame({"start_utc": start, "end_utc": start + pd.Timedelta(hours=1), "x": rng.normal(size=n)})


def g(q=0.1, kind=GuaranteeKind.FDR, m=10):
    return Guarantee(kind=kind, level=q, family={"rule": "BH", "m": m, "q": q})


def mk(claim, ell_test, ell_prop, consumed=(), kind=CertificateKind.CARRIER, support=("a1",), guarantee=None):
    return issue(claim=claim, guarantee=guarantee or g(), ell_test=ell_test, proposer_rows=ell_prop, artifact={"pvalues": [0.001]}, support=support, gram_context={"gram": "identity"}, kind=kind, consumed=consumed)


def test_hereditary_issue():
    f = frame("dataset_1", 1)
    S = Lineage.of(slice_from_frame(f, "dataset_1", range(0, 8)))
    P = Lineage.of(slice_from_frame(f, "dataset_1", range(8, 16)))
    c1 = mk("c1", S, P)
    assert c1.full_lineage() == S.union(P)
    Q = Lineage.of(slice_from_frame(f, "dataset_1", range(16, 20)))
    T = Lineage.of(slice_from_frame(f, "dataset_1", range(20, 24)))
    c2 = mk("c2", T, Q, consumed=(c1,))
    for sl in S.union(P, Q).slices:
        assert sl in c2.ell_prop
    for sl in T.slices:
        assert sl in c2.ell_test


def test_guarantee_level_domain():
    with pytest.raises(ValueError):
        Guarantee(kind=GuaranteeKind.FDR, level=0.0, family={})
    with pytest.raises(ValueError):
        Guarantee(kind=GuaranteeKind.EXACT, level=0.1, family={})
    Guarantee(kind=GuaranteeKind.EXACT, level=0.0, family={})


def test_refines_predicate(tmp_path):
    f = frame("dataset_1", 2, seed=1)
    L = Lineage.of(slice_from_frame(f, "dataset_1", range(0, 6)))
    M = Lineage.of(slice_from_frame(f, "dataset_1", range(6, 12)))
    ex = mk("exists", L, Lineage(), kind=CertificateKind.EXISTENCE, support=("a1", "a2", "a3"))
    good = mk("carrier", M, Lineage(), support=("a1",), guarantee=Guarantee(GuaranteeKind.FDR, 0.1, {"rule": "BH", "multiplicity": 3, "q": 0.1}))
    assert refines(good, ex)
    bad_support = mk("carrier2", M, Lineage(), support=("zz",), guarantee=Guarantee(GuaranteeKind.FDR, 0.1, {"multiplicity": 3}))
    assert not refines(bad_support, ex)
    bad_mult = mk("carrier3", M, Lineage(), support=("a1",), guarantee=Guarantee(GuaranteeKind.FDR, 0.1, {"multiplicity": 7}))
    assert not refines(bad_mult, ex)
    led = Ledger(str(tmp_path / "book"))
    led.register(mk("x", L, Lineage(), support=tuple(f"s{i}" for i in range(7))))
    assert refines(bad_mult, ex, ledger=led)
    assert not refines(good, good)


def test_ledger_state(tmp_path):
    f = frame("dataset_1", 3, seed=2)
    L = Lineage.of(slice_from_frame(f, "dataset_1", range(0, 4)))
    led = Ledger(str(tmp_path / "book"), budget=0.5)
    i1 = led.register(mk("a", L, Lineage(), support=("a1", "a2")))
    led.register(mk("b", L, Lineage(), support=("a2", "a3")))
    assert led.certified_support_union() == frozenset({"a1", "a2", "a3"})
    assert led.statement().charged == pytest.approx(0.2)
    led.void(i1, cause="staged leak")
    assert led.certified_support_union() == frozenset({"a2", "a3"})
    assert led.statement().charged == pytest.approx(0.1)


def test_admit_contract():
    f = frame("dataset_1", 4, seed=3)
    S = Lineage.of(slice_from_frame(f, "dataset_1", range(0, 8)))
    Sp = Lineage.of(slice_from_frame(f, "dataset_1", range(8, 16)))
    c = mk("c", S, Lineage())
    ok = admit([c], Sp, level=0.05, mode="object")
    assert ok.admitted and ok.items[0].verdict == Verdict.FRESH
    assert ok.inflation == 1.0 and ok.level == pytest.approx(0.05)
    with pytest.raises(FreshnessError) as ei:
        admit([c], S, level=0.05, mode="guidance")
    assert "rows" in str(ei.value)
    dem = admit([c], S, level=0.05, on_stale="demote")
    assert not dem.admitted and dem.verdict == Verdict.NOT_FRESH
    assert dem.demotion[0].status == MISCALIBRATED
    assert dem.demotion[0].overlap
    assert dem.demotion[0].original_level == pytest.approx(0.1)
    with pytest.raises(ValueError):
        admit([c], Sp, level=0.05, mode="license")


def test_admit_dependence_flag():
    gb = frame("dataset_1", 5, seed=4)
    de = frame("dataset_2", 5, seed=5)
    c = mk("gb-cert", Lineage.of(slice_from_frame(gb, "dataset_1")), Lineage())
    S = Lineage.of(slice_from_frame(de, "dataset_2"))
    res = admit([c], S, level=0.1)
    assert res.admitted
    assert res.items[0].verdict == Verdict.FRESH_DEP_FLAG
    fl = res.flagged()[0].dependence_flags[0]
    assert set(fl["tags"]) == {"dataset_1", "dataset_2"} and fl["window"][0] < fl["window"][1]


def test_screen_then_gate_detector():
    f = frame("dataset_1", 6, seed=6)
    rows = Lineage.of(slice_from_frame(f, "dataset_1", range(0, 12)))
    other = Lineage.of(slice_from_frame(f, "dataset_1", range(12, 24)))
    rep = detect_screen_then_gate(2.0, 0.1, rows, rows)
    assert rep is not None
    assert rep.realized_level == pytest.approx(min(1.0, 0.1 / rep.null_pass_rate))
    assert rep.repaired_level == pytest.approx(0.1 * rep.null_pass_rate)
    assert rep.inflation == pytest.approx(1.0 / rep.null_pass_rate)
    assert detect_screen_then_gate(2.0, 0.1, rows, other) is None
