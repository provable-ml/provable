"""Twin v0.1 numbered lock suite. Every lock states its target before
reading the observation; all-pass is the acceptance bar. Run as pytest,
or as a script for the scoreboard table."""
import inspect
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from tabulate import tabulate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import provable as pv
from provable import certificates as ct
from provable.certificates import kit
from provable.certificates.records import to_dict, from_dict, content_id, verify_chain, from_evidence, unknown_lineage
from provable.facade import ManifestMismatch, PROFILES

UTC = timezone.utc
LOCKS = []
_STATE = {}


def lock(lid, target):
    def deco(fn):
        LOCKS.append((lid, target, fn))
        return fn
    return deco


def frame(tag, day, n=24, seed=0):
    rng = np.random.default_rng(seed)
    start = pd.date_range(datetime(2026, 9, day, tzinfo=UTC), periods=n, freq="h")
    return pd.DataFrame({"start_utc": start, "end_utc": start + pd.Timedelta(hours=1), "x": rng.normal(size=n)})


def g(q=0.1):
    return ct.Guarantee(kind=ct.GuaranteeKind.FDR, level=q, family={"rule": "BH", "q": q, "m": 10})


def mk(claim, S, P, consumed=(), support=("a1",), kind=ct.CertificateKind.CARRIER, guarantee=None):
    return ct.issue(claim=claim, guarantee=guarantee or g(), ell_test=S, proposer_rows=P, artifact={"note": claim}, support=support, gram_context={}, kind=kind, consumed=consumed)


def _panel(n=800, seed=0, signal=True):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"x{i}" for i in range(8)])
    y = rng.normal(size=n)
    if signal:
        y = y + 1.2 * (X["x0"] > 0.3).astype(float) + 0.9 * (X["x1"] > 0.0).astype(float) * (X["x2"] > 0.0).astype(float)
    return X, y


def _certify_pair():
    if "rep" not in _STATE:
        X, y = _panel(seed=1)
        led_dir = tempfile.mkdtemp(prefix="book_")
        rep = pv.certify(X, y, q=0.10, ledger=led_dir, profile="fast", seed=3, dataset_tag="synthetic")
        bare = pv.CertifiedDiscovery(fdr_q=0.10, seed=3, verbose=False, **PROFILES["fast"])
        bare.run(X, y)
        _STATE.update(rep=rep, bare=bare, X=X, y=y, led_dir=led_dir)
    return _STATE["rep"], _STATE["bare"]


def _xgb_models():
    if "clf" not in _STATE:
        import xgboost as xgb
        rng = np.random.default_rng(7)
        Xt = rng.normal(size=(400, 6))
        yt = (Xt[:, 0] + 0.8 * (Xt[:, 1] > 0) * (Xt[:, 2] > 0) + 0.3 * rng.normal(size=400) > 0).astype(int)
        clf = xgb.XGBClassifier(n_estimators=25, max_depth=2, learning_rate=0.3, base_score=0.62, tree_method="hist", random_state=0)
        clf.fit(Xt, yt)
        reg = xgb.XGBRegressor(n_estimators=25, max_depth=2, learning_rate=0.3, tree_method="hist", random_state=0)
        reg.fit(Xt, 0.4 * Xt[:, 0] + 1.5 * (Xt[:, 1] > 0) * (Xt[:, 2] > 0) + 0.3 * (Xt[:, 4] > 0.5))
        td = tempfile.mkdtemp(prefix="mdl_")
        cp, rp = os.path.join(td, "clf.json"), os.path.join(td, "reg.json")
        clf.get_booster().save_model(cp)
        reg.get_booster().save_model(rp)
        _STATE.update(clf_path=cp, reg_path=rp, Xt=Xt)
    return _STATE["clf_path"], _STATE["reg_path"], _STATE["Xt"]


# ------------------------------------------------------------------ TA records
@lock("TA1", "certificate to-dict/from-dict round trip preserves canonical bytes")
def ta1():
    f = frame("dataset_1", 1)
    c = mk("c", ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(0, 8))), ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(8, 16))))
    d = to_dict(c)
    c2 = from_dict(d)
    assert ct.canonical_bytes(c) == ct.canonical_bytes(c2)
    return "round trip byte-equal"


@lock("TA2", "content id stable under identity, moved by any field change")
def ta2():
    f = frame("dataset_1", 1)
    S = ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(0, 8)))
    ts = datetime(2026, 9, 13, tzinfo=UTC)
    c1 = ct.issue("c", g(), S, ct.Lineage(), {"a": 1}, ("a1",), {}, ct.CertificateKind.CARRIER, timestamp=ts)
    c2 = ct.issue("c", g(), S, ct.Lineage(), {"a": 1}, ("a1",), {}, ct.CertificateKind.CARRIER, timestamp=ts)
    c3 = ct.issue("c", g(0.05), S, ct.Lineage(), {"a": 1}, ("a1",), {}, ct.CertificateKind.CARRIER, timestamp=ts)
    assert c1.id == c2.id and c1.id != c3.id and c1.id.startswith("CD-")
    return c1.id


@lock("TA3", "consumed hashes inside content: tamper breaks verify_chain")
def ta3():
    f = frame("dataset_1", 2)
    S = ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(0, 8)))
    T = ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(8, 16)))
    base = mk("base", S, ct.Lineage())
    child = mk("child", T, ct.Lineage(), consumed=(base,))
    assert verify_chain(child)
    tampered = ct.Certificate(**{**child.__dict__, "consumed": (mk("base", S, ct.Lineage(), support=("EVIL",)),)})
    assert to_dict(tampered)["consumed_ids"] != to_dict(child)["consumed_ids"]
    d = to_dict(child)
    d2 = dict(d)
    d2["claim"] = "edited"
    assert content_id(d2) != content_id(d)
    return "chain tamper-evident"


@lock("TA4", "additive schema: unknown fields preserved through load/save")
def ta4():
    f = frame("dataset_1", 3)
    c = mk("c", ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(0, 4))), ct.Lineage())
    d = to_dict(c)
    d["future_field"] = {"x": 1}
    c2 = from_dict(d)
    assert to_dict(c2)["future_field"] == {"x": 1}
    return "future_field preserved"


@lock("TA5", "row-set compression round trips exactly")
def ta5():
    from provable.certificates.records import _rows_pack, _rows_unpack
    rows = frozenset(int(x) for x in np.random.default_rng(0).choice(100000, size=5000, replace=False))
    assert _rows_unpack(_rows_pack(rows)) == rows
    assert _rows_unpack(None) is None
    return "5000-row set exact"


@lock("TA6", "unification: audit bundle -> artifact certificate, PV id kept, verify_id intact")
def ta6():
    cp, rp, Xt = _xgb_models()
    pol = pv.Policy([pv.between("f1", "f2")])
    b = pv.audit(cp, pol, Xt)
    assert pv.verify_id(b)
    c = from_evidence(b)
    assert c.kind == ct.CertificateKind.ARTIFACT
    assert c.guarantee.kind.value == "EXACT" and c.guarantee.level == 0.0
    assert c.id == b["certificate_id"] and c.id.startswith("PV-")
    assert c.full_lineage().is_empty()
    assert pv.verify_id(c.artifact)
    d = to_dict(c)
    assert content_id(d) == b["certificate_id"]
    return c.id


@lock("TA7", "EXACT level domain: level 0 required, positive refused")
def ta7():
    with pytest.raises(ValueError):
        ct.Guarantee(kind=ct.GuaranteeKind.EXACT, level=0.05, family={})
    ct.Guarantee(kind=ct.GuaranteeKind.EXACT, level=0.0, family={})
    return "EXACT pinned at level 0"


# ------------------------------------------------------------------ TB ledger
@lock("TB1", "ledger append-only jsonl reloads to the same book")
def tb1():
    d = tempfile.mkdtemp(prefix="led_")
    led = ct.Ledger(d, budget=0.3)
    f = frame("dataset_1", 4)
    led.register(mk("a", ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(0, 4))), ct.Lineage()))
    led.register(mk("b", ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(4, 8))), ct.Lineage()))
    led2 = ct.Ledger(d)
    assert len(led2.active()) == 2 and led2.budget == 0.3
    with open(os.path.join(d, "ledger.jsonl")) as fh:
        assert sum(1 for _ in fh) == 2
    return "2 ops, reload equal"


@lock("TB2", "void excludes from the account with cause recorded")
def tb2():
    d = tempfile.mkdtemp(prefix="led_")
    led = ct.Ledger(d, budget=0.3)
    f = frame("dataset_1", 5)
    cid = led.register(mk("a", ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(0, 4))), ct.Lineage()))
    led.void(cid, "staged leak")
    st = led.statement()
    assert st.charged == 0.0 and st.voids[0]["cause"] == "staged leak"
    return "voided, charged 0"


@lock("TB3", "admission against the book: fresh rows pass at the account level")
def tb3():
    d = tempfile.mkdtemp(prefix="led_")
    led = ct.Ledger(d, budget=0.10)
    f = frame("dataset_1", 6)
    led.register(mk("a", ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(0, 8))), ct.Lineage()))
    adm = led.admit(rows=ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(8, 16))))
    assert adm.admitted and adm.verdict == ct.Verdict.FRESH and adm.level == pytest.approx(0.10)
    return "FRESH at book level 0.10"


@lock("TB4", "stale reuse refuses by default with rows named; demote path records MISCALIBRATED")
def tb4():
    f = frame("dataset_1", 7)
    S = ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(0, 8)))
    c = mk("c", S, ct.Lineage())
    with pytest.raises(ct.FreshnessError) as ei:
        ct.admit([c], S, level=0.05)
    assert "rows" in str(ei.value)
    dem = ct.admit([c], S, level=0.05, on_stale="demote")
    assert dem.demotion[0].status == ct.MISCALIBRATED and dem.demotion[0].overlap
    return "refuse + demote both priced"


@lock("TB5", "verdict enum identifiers distinct from printed labels")
def tb5():
    assert ct.Verdict.NOT_FRESH.name == "NOT_FRESH" and ct.Verdict.NOT_FRESH.label == "NOT FRESH"
    assert ct.Verdict.UNKNOWN_LINEAGE.label == "UNKNOWN LINEAGE"
    assert ct.Verdict.FRESH_DEP_FLAG.label == "FRESH WITH DEPENDENCE FLAG"
    return "labels carry spaces, identifiers do not"


@lock("TB6", "UNKNOWN LINEAGE: un-instrumented item is returned, never FRESH, never budget-bearing")
def tb6():
    d = tempfile.mkdtemp(prefix="led_")
    led = ct.Ledger(d, budget=0.10)
    hist = ct.issue("historical run", g(0.1), unknown_lineage(), ct.Lineage(), {"note": "pre-instrumentation"}, ("a1",), {}, ct.CertificateKind.CARRIER)
    led.register(hist)
    adm = led.admit(rows=ct.Lineage())
    assert adm.verdict == ct.Verdict.UNKNOWN_LINEAGE and adm.admitted is False
    st = led.statement()
    assert st.charged == 0.0 and len(st.unknowns) == 1
    assert "UNKNOWN" in st.render()
    return "UNKNOWN listed, charged 0"


@lock("TB7", "additive budget: charged = sum of active levels; EXACT excluded")
def tb7():
    d = tempfile.mkdtemp(prefix="led_")
    led = ct.Ledger(d, budget=0.5)
    f = frame("dataset_1", 8)
    L = ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(0, 4)))
    led.register(mk("a", L, ct.Lineage(), guarantee=g(0.10)))
    led.register(mk("b", L, ct.Lineage(), guarantee=g(0.05)))
    cp, rp, Xt = _xgb_models()
    led.register(from_evidence(pv.audit(cp, pv.Policy([pv.involving("f0")]), Xt)))
    st = led.statement()
    assert st.charged == pytest.approx(0.15)
    assert kit.additive_budget([0.10, 0.05]) == pytest.approx(0.15)
    return "0.15 of 0.5; artifact kind free"


@lock("TB8", "reserve: plain hold debits before outcomes and over-budget refuses; the vocabulary form registers the Prop 6.3 charge")
def tb8():
    d = tempfile.mkdtemp(prefix="led_")
    led = ct.Ledger(d, budget=0.10)
    rid = led.reserve(0.04, note="next quarter test")
    st = led.statement()
    assert st.reserved == pytest.approx(0.04) and st.remaining == pytest.approx(0.06)
    with pytest.raises(ValueError):
        led.reserve(0.07)
    f = frame("dataset_1", 9)
    led.register(mk("a", ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(0, 4))), ct.Lineage(), guarantee=g(0.04)), reservation=rid)
    st2 = led.statement()
    assert st2.reserved == 0.0 and st2.charged == pytest.approx(0.04)
    rv = led.reserve(0.05, note="registered vocabulary", vocabulary=[f"m{i}@rows1" for i in range(25)])
    with pytest.raises(ValueError):
        led.claim_registered(rid, "m0@rows1", 0.001)
    chk = led.claim_registered(rv, "m3@rows1", 0.05 / 25 * 0.9)
    assert chk["valid"] and chk["bar"] == pytest.approx(0.002) and chk["source"] == "M2 Prop 6.3"
    return "hold consumed; vocabulary charge at bar 0.002"


@lock("TB9", "batch step-up active per the transcribed Algorithm 1: levels match a hand derivation, estimator stays under target, deviation detected")
def tb9():
    from provable.certificates.batch import BatchSchedule, bh_rejections, default_gamma, r_plus
    q = 0.10
    sched = BatchSchedule(q=q)
    g1, g2, g3 = default_gamma(1), default_gamma(2), default_gamma(3)
    a1 = sched.next_level(4)
    assert a1 == pytest.approx(g1 * q)
    p1 = [0.0004 * (i + 1) for i in range(4)]
    b1 = sched.record(p1, a1)
    assert b1.R == bh_rejections(p1, a1) and b1.R_plus == r_plus(p1, a1) and b1.R_plus >= b1.R
    beta2 = a1 * b1.R_plus / (b1.R_plus + 0)
    a2_hand = ((g1 + g2) * q - beta2) * (6 + b1.R) / 6
    a2 = sched.next_level(6)
    assert a2 == pytest.approx(max(0.0, a2_hand))
    p2 = [0.9, 0.8, 0.7, 0.6, 0.5, 0.001 * a2]
    b2 = sched.record(p2, a2)
    beta3 = a1 * b1.R_plus / (b1.R_plus + b2.R) + a2 * b2.R_plus / (b2.R_plus + b1.R)
    a3_hand = ((g1 + g2 + g3) * q - beta3) * (5 + b1.R + b2.R) / 5
    assert sched.next_level(5) == pytest.approx(max(0.0, a3_hand))
    assert sched.fdp_hat() <= q + 1e-12 and sched.conforms()
    bad = BatchSchedule(q=q)
    bad.record(p1, 0.05)
    assert not bad.conforms()
    return f"a1={a1:.4f}, a2={a2:.4f} both hand-matched; FDPhat {sched.fdp_hat():.4f} <= q"


@lock("TB10", "the batch schedule controls the union FDR in simulation and reduces to the online estimator at batch size one")
def tb10():
    from provable.certificates.batch import BatchSchedule
    from provable.certificates import kit as _kit
    rng = np.random.default_rng(20260913)
    q, reps, T, n = 0.10, 300, 6, 25
    fdps = []
    for _ in range(reps):
        sched = BatchSchedule(q=q)
        V = R = 0
        for t in range(T):
            lvl = sched.next_level(n)
            nul = rng.random(n) > 0.2
            z = np.where(nul, 0.0, 3.0) + rng.normal(size=n)
            p = 2 * norm.sf(np.abs(z))
            rec = sched.record(p, lvl)
            rej = _kit.rejected(p, lvl) if lvl > 0 else np.array([], int)
            V += int(nul[rej].sum())
            R += len(rej)
        fdps.append(V / max(R, 1))
    fdr = float(np.mean(fdps))
    se = float(np.std(fdps) / np.sqrt(reps))
    assert fdr <= q + 3 * se
    ones = BatchSchedule(q=q)
    lv = []
    for t in range(5):
        lv.append(ones.next_level(1))
        ones.record([0.001 if t % 2 == 0 else 0.9], lv[-1])
    Rs = [b.R for b in ones.batches]
    hand = sum(l * 1.0 / (1.0 + (sum(Rs) - r)) for l, r in zip(lv, Rs))
    assert ones.fdp_hat() == pytest.approx(hand)
    return f"union FDR {fdr:.3f} vs q={q} (se {se:.3f}); size-1 estimator matches"


@lock("TB11", "the registered-vocabulary charge survives arbitrary reuse: valid on spent rows at alpha/|V|, refused outside V, familywise level held")
def tb11():
    d = tempfile.mkdtemp(prefix="led_")
    led = ct.Ledger(d, budget=0.25)
    f = frame("dataset_1", 15)
    S = ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(0, 12)))
    led.register(mk("spender", S, ct.Lineage()))
    alpha, V = 0.05, 40
    rid = led.reserve(alpha, note="preregistered pairs", vocabulary=[f"m{i}@S" for i in range(V)])
    with pytest.raises(ct.FreshnessError):
        led.admit(rows=S)
    chk = led.claim_registered(rid, "m7@S", alpha / V * 0.5)
    assert chk["valid"] and chk["lineage_free"]
    assert not led.claim_registered(rid, "m7@S", alpha / V * 1.5)["valid"]
    assert not led.claim_registered(rid, "outside@S", 1e-9)["valid"]
    assert 1 - (1 - alpha / V) ** V <= alpha + 1e-12
    rng = np.random.default_rng(3)
    hits = (rng.random((4000, V)).min(axis=1) <= alpha / V).mean()
    assert hits <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / 4000)
    return f"bar {alpha / V:.4g}; any-false rate {hits:.4f} <= {alpha}"


@lock("TB12", "ledger notes: first-class op and raw appended note both inert to the account and rendered")
def tb12():
    d = tempfile.mkdtemp(prefix="led_")
    led = ct.Ledger(d, budget=0.10)
    f = frame("dataset_1", 16)
    led.register(mk("a", ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(0, 4))), ct.Lineage()))
    led.note("baseline pre-dates the bars dedupe")
    with open(os.path.join(d, "ledger.jsonl"), "a") as fh:
        fh.write(json.dumps({"note": "manually appended provenance line", "op": "note"}, sort_keys=True, separators=(",", ":")).replace("manually", "manually") + "\n")
    led2 = ct.Ledger(d)
    st = led2.statement()
    assert st.charged == pytest.approx(0.1) and len(led2.active()) == 1
    assert any("bars dedupe" in n for n in st.notes)
    assert "-- notes --" in st.render()
    led2.next_level(10)
    return f"{len(st.notes)} notes carried, account untouched"


def _in_units(f, unit):
    """The same table with its clock columns stored in another unit, as
    another pandas version (or another loader) would hold it."""
    g = f.copy()
    for c in ("start_utc", "end_utc"):
        g[c] = g[c].astype(f"datetime64[{unit}, UTC]")
    return g


@lock("TB13", "one id whatever the storage unit: the same table with its clock columns in s, ms, us or ns has one fingerprint, equal to the earlier releases' id for microsecond storage; a table without clock columns keeps its id; the aliases hold the earlier releases' id under every storage")
def tb13():
    from provable.certificates.lineage import legacy_fingerprint
    f = frame("dataset_1", 3)
    variants = {u: _in_units(f, u) for u in ("s", "ms", "us", "ns")}
    ids = {u: ct.source_fingerprint(v) for u, v in variants.items()}
    assert len(set(ids.values())) == 1 and ids["us"] == legacy_fingerprint(variants["us"])
    legacy = {u: legacy_fingerprint(v) for u, v in variants.items()}
    assert len(set(legacy.values())) == 4
    for v in variants.values():
        assert set(legacy.values()) <= ct.source_aliases(v)
    plain = f[["x"]]
    assert ct.source_fingerprint(plain) == legacy_fingerprint(plain) and ct.source_aliases(plain) == {legacy_fingerprint(plain)}
    sub = pd.DataFrame({"t": pd.to_datetime([1_700_000_000_000_000_001, 1_700_000_000_000_000_002]), "x": [1.0, 2.0]})
    assert ct.source_fingerprint(sub) == legacy_fingerprint(sub)
    return f"one id {ids['us']} across s/ms/us/ns; {len(ct.source_aliases(variants['ns']))} aliases cover the 4 earlier ids"


@lock("TB14", "a book written by an earlier release still recognises its rows: a holdout sealed under the earlier id of nanosecond storage refuses a model whose lineage, built now from the same table in microseconds, meets the sealed rows, naming both ids; the converse holds; disjoint rows pass")
def tb14():
    from provable.certificates.lineage import legacy_fingerprint
    rng = np.random.default_rng(14)
    n = 240
    panel = pd.DataFrame({"x": rng.normal(size=n)})
    panel["y"] = 0.5 * panel["x"] + rng.normal(size=n)
    panel["start_utc"] = pd.date_range("2026-01-05", periods=n, freq="h", tz="UTC")
    panel["end_utc"] = panel["start_utc"] + pd.Timedelta(hours=1)
    out = []
    for sealed_unit, asked_unit in (("ns", "us"), ("us", "ns")):
        old = _in_units(panel, sealed_unit)
        live = _in_units(panel, asked_unit)
        hold = slice(180, n)
        sl = ct.slice_from_frame(old, "p", rows=range(180, n))
        legacy = ct.Slice(source_id=legacy_fingerprint(old), dataset_tag="p", rows=sl.rows, t0=sl.t0, t1=sl.t1)
        ho = pv.seal(panel.iloc[hold][["x"]], panel.iloc[hold]["y"], tempfile.mkdtemp(prefix="leg_"), lineage=ct.Lineage.of(legacy), baseline=0.0, baseline_lineage=ct.Lineage(), exam_share=0.25, seed=1)
        f = lambda Z: 0.5 * Z["x"]
        with pytest.raises(ValueError, match="meets the sealed rows") as e:
            ho.ask(f, lineage=ct.Lineage.of(ct.slice_from_frame(live, "p")))
        assert legacy.source_id in str(e.value) and ct.source_fingerprint(live) in str(e.value)
        ok = ho.ask(f, lineage=ct.Lineage.of(ct.slice_from_frame(live, "p", rows=range(0, 180))))
        out.append(f"{sealed_unit} book refuses a {asked_unit} lineage; disjoint rows {'accepted' if ok.accepted else 'rejected'}")
    return "; ".join(out)


# ------------------------------------------------------------------ TC lint
@lock("TC1", "R1 pass on disjoint lineage, violation names rows, source tag exact")
def tc1():
    f = frame("dataset_1", 10)
    S = ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(0, 8)))
    T = ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(8, 16)))
    rr = ct.RunRecord(name="clean", test_rows=T, proposal_rows=S)
    fnd = ct.run_checks(rr)
    r1 = [r for r in fnd.rows if r.check == "R1"][0]
    assert r1.verdict == "PASS" and r1.source == "M2 Thm 3.1; U1 Lem 2.4"
    rr2 = ct.RunRecord(name="leaky", test_rows=T, proposal_rows=T)
    bad = [r for r in ct.run_checks(rr2).rows if r.check == "R1"][0]
    assert bad.verdict == "VIOLATION" and "rows" in bad.detail
    return "PASS/VIOLATION with named rows"


@lock("TC2", "R1 over un-instrumented history returns UNKNOWN, not PASS")
def tc2():
    fnd = pv.lint({"name": "old console run", "test_rows": "unknown", "proposal_rows": "unknown"})
    r1 = [r for r in fnd.rows if r.check == "R1"][0]
    assert r1.verdict == "UNKNOWN" and len(fnd.unknowns) >= 1
    return "UNKNOWN verdict emitted"


@lock("TC3", "R2 prices fresh confirmations at the product level")
def tc3():
    f = frame("dataset_1", 11)
    T = ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(0, 8)))
    A = ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(8, 16)))
    B = ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(16, 24)))
    rr = ct.RunRecord(name="conf", test_rows=T, proposal_rows=ct.Lineage(), retests=[ct.Retest(rows=A, level=0.05), ct.Retest(rows=B, level=0.04)])
    fnd = ct.run_checks(rr)
    r2 = [r for r in fnd.rows if r.check == "R2"][0]
    assert r2.repair["joint_level"] == pytest.approx(0.002)
    return "joint level 0.002 = 0.05*0.04"


@lock("TC4", "R3 vacuous re-test flagged with the leave-one-in repair; kit relations hold on 2000 seeded cases")
def tc4():
    f = frame("dataset_1", 12)
    T = ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(0, 12)))
    rng = np.random.default_rng(0)
    p = np.clip(rng.uniform(size=6), 1e-6, 1)
    p[:2] = [1e-5, 3e-4]
    kh = kit.khat(p, 0.1)
    cert = [int(i) for i in kit.rejected(p, 0.1)]
    rr = ct.RunRecord(name="retest", test_rows=T, proposal_rows=ct.Lineage(), q=0.1, m=6, khat=kh, pvalues=list(p), certified_idx=cert, retests=[ct.Retest(rows=T, level=0.05, on_certified=True)])
    fnd = ct.run_checks(rr)
    r3 = [r for r in fnd.rows if r.check == "R3"][0]
    assert r3.verdict == "VIOLATION" and set(fnd.repair("R3")["confirmation_levels"]) == set(cert)
    for lv in fnd.repair("R3")["confirmation_levels"].values():
        assert 0 < lv <= 0.05
    rng = np.random.default_rng(20260913)
    for _ in range(2000):
        m = rng.integers(3, 25)
        pp = np.clip(rng.uniform(size=m) ** rng.uniform(0.3, 3), 1e-12, 1)
        q = float(rng.uniform(0.02, 0.3))
        kh = kit.khat(pp, q)
        rej = kit.rejected(pp, q)
        assert len(rej) == kh
        for i in rej:
            assert pp[i] <= kit.tau(pp, q, int(i)) + 1e-15
    return "repair emitted; kit brute-consistent 2000/2000"


@lock("TC5", "R4 screen-then-gate: inflation 1/s = 3.15x at c=1, repair q*s, repaired p divided by s")
def tc5():
    f = frame("dataset_1", 13)
    T = ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(0, 12)))
    p = [0.001, 0.02, 0.2]
    rr = ct.RunRecord(name="screened", test_rows=T, proposal_rows=ct.Lineage(), q=0.10, pvalues=p, screen=ct.ScreenDecl(threshold=1.0, rows=T))
    fnd = ct.run_checks(rr)
    r4 = [r for r in fnd.rows if r.check == "R4"][0]
    s = 2 * norm.sf(1.0)
    assert r4.verdict == "VIOLATION"
    assert abs(1.0 / s - 3.1515) < 5e-3
    assert "3.15" in r4.price
    rep = fnd.repair("R4")
    assert rep["repaired_level"] == pytest.approx(0.10 * s)
    assert rep["repaired_pvalues"][0] == pytest.approx(min(1.0, p[0] / s))
    return f"inflation {1.0 / s:.4f}x, repair {0.10 * s:.4f}"


@lock("TC6", "R5 union at a common level flagged; additive level is the valid statement")
def tc6():
    rr = ct.RunRecord(name="union", test_rows=ct.Lineage(), proposal_rows=ct.Lineage(), instrumented=True, union_claims=[ct.UnionClaim(cert_levels=[0.1, 0.1, 0.05], claimed_level=0.1)])
    fnd = ct.run_checks(rr)
    r5 = [r for r in fnd.rows if r.check == "R5"][0]
    assert r5.verdict == "VIOLATION" and fnd.repair("R5")["additive_level"] == pytest.approx(0.25)
    assert r5.source == "M2 Thm 6.5; Prop 6.3"
    return "additive 0.25 quoted"


@lock("TC7", "R6 refinement truth table through the ledger multiplicity")
def tc7():
    f = frame("dataset_1", 14)
    L = ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(0, 6)))
    M = ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(6, 12)))
    ex = mk("exists", L, ct.Lineage(), kind=ct.CertificateKind.EXISTENCE, support=("a1", "a2", "a3"))
    good = mk("carrier", M, ct.Lineage(), support=("a1",), guarantee=ct.Guarantee(ct.GuaranteeKind.FDR, 0.1, {"multiplicity": 3}))
    bad = mk("carrier", M, ct.Lineage(), support=("a1",), guarantee=ct.Guarantee(ct.GuaranteeKind.FDR, 0.1, {"multiplicity": 7}))
    rr = ct.RunRecord(name="refine", test_rows=M, proposal_rows=ct.Lineage(), refinements=[ct.RefinementDecl(child=good, parent=ex), ct.RefinementDecl(child=bad, parent=ex)])
    verdicts = [r.verdict for r in ct.run_checks(rr).rows if r.check == "R6"]
    assert verdicts == ["PASS", "VIOLATION"]
    return "PASS then VIOLATION"


# ------------------------------------------------------------------ TD rates
@lock("TD1", "row_equivalent matches the currency law")
def td1():
    n, lam, tau = 19596, 0.658, 0.4
    want = n * lam ** 2 / (tau * (1 - lam ** 2))
    assert ct.row_equivalent(n, lam, tau) == pytest.approx(want)
    assert "columns -> rows" in ct.REGISTER["row_equivalent"].line()
    return f"{want:,.0f} effective rows"


@lock("TD2", "rows_for_level is the squared-bar identity, monotone the right way")
def td2():
    base = ct.rows_for_level(1000, 0.10, 0.10, k=3, m=100)
    tighter = ct.rows_for_level(1000, 0.10, 0.01, k=3, m=100)
    looser = ct.rows_for_level(1000, 0.10, 0.30, k=3, m=100)
    z = lambda q: norm.isf(q * 3 / (2 * 100))
    assert base == pytest.approx(1000)
    assert tighter == pytest.approx(1000 * (z(0.01) / z(0.10)) ** 2) and tighter > 1000 > looser
    return f"q 0.10->0.01 needs {tighter:,.0f} rows"


@lock("TD3", "anchor_to_promise is DECLARED: refuses without a bridge, applies one when given")
def td3():
    with pytest.raises(ValueError) as ei:
        ct.anchor_to_promise(100.0)
    assert "DECLARED" in str(ei.value)
    assert ct.anchor_to_promise(100.0, bridge=lambda a: 0.003 * a) == pytest.approx(0.3)
    assert "DECLARED" in ct.REGISTER["anchor_to_promise"].grade
    return "declared bridge only"


# ------------------------------------------------------------------ TE facade
@lock("TE1", "certify is byte-neutral over the engine: same seed, identical certification")
def te1():
    rep, bare = _certify_pair()
    a, b = rep.engine["certification"], bare.report_["certification"]
    assert a == b
    assert rep.claims["atom"].tolist()[: len(bare.claims_)] == bare.claims_["atom"].tolist()
    assert np.allclose(rep.claims["t"].values, bare.claims_["t"].values)
    return f"khat={a['khat']}, claims identical"


@lock("TE2", "manifest gate: expected-manifest mismatch refuses before any ledger write")
def te2():
    rep, _ = _certify_pair()
    X, y = _STATE["X"], _STATE["y"]
    d = tempfile.mkdtemp(prefix="book_")
    with pytest.raises(ManifestMismatch):
        pv.certify(X, y, q=0.10, ledger=d, profile="fast", seed=3, expect_manifest={"x_sha": "deadbeef"})
    assert not os.path.exists(os.path.join(d, "ledger.jsonl"))
    ok = pv.certify(X, y, q=0.10, profile="fast", seed=3, expect_manifest={"x_sha": rep.manifest["x_sha"]})
    assert ok.manifest["x_sha"] == rep.manifest["x_sha"]
    return "mismatch refused, match passes"


@lock("TE3", "selective intervals at alpha_sel = khat*q/m, widths match statsmodels")
def te3():
    rep, bare = _certify_pair()
    m = len(bare.core_)
    want = bare.khat_ * 0.10 / m
    assert rep.alpha_sel == pytest.approx(want)
    ci = np.asarray(bare.core_res_.conf_int(alpha=want))[1:]
    pos = np.where(bare.cert_sel_)[0][0]
    j = bare.core_[pos]
    sd = float(bare._sd[j])
    row = rep.claims.loc[rep.claims["atom_id"] == j].iloc[0]
    assert row["ci_lo_sel"] == pytest.approx(ci[pos, 0] / sd) and row["ci_hi_sel"] == pytest.approx(ci[pos, 1] / sd)
    assert row["ci_hi_sel"] - row["ci_lo_sel"] > row["ci_hi"] - row["ci_lo"]
    return f"alpha_sel={want:.4g}, wider than marginal"


@lock("TE4", "FCR of the selective intervals holds (rep-clustered se) and beats the marginal 95 on selected signals")
def te4():
    rng = np.random.default_rng(20260913)
    m, q, reps = 60, 0.10, 400
    theta = np.zeros(m)
    theta[:12] = 3.2
    fcps, miss = [], 0
    sig_miss_sel = sig_miss_marg = sig_tot = 0
    h95 = norm.isf(0.025)
    for _ in range(reps):
        z = theta + rng.normal(size=m)
        p = 2 * norm.sf(np.abs(z))
        kh = kit.khat(p, q)
        if kh == 0:
            continue
        sel = kit.rejected(p, q)
        a = kh * q / m
        h = norm.isf(a / 2)
        bad = (theta < z - h) | (theta > z + h)
        fcps.append(float(bad[sel].mean()))
        miss += int(bad[sel].sum())
        sig = sel[sel < 12]
        sig_tot += len(sig)
        sig_miss_sel += int(bad[sig].sum())
        sig_miss_marg += int(((theta < z - h95) | (theta > z + h95))[sig].sum())
    fcr = float(np.mean(fcps))
    se = float(np.std(fcps) / np.sqrt(len(fcps)))
    assert fcr <= q + 3 * se
    assert sig_miss_marg > sig_miss_sel
    return f"FCR {fcr:.3f} vs q={q} (se {se:.3f}); signal noncoverage marginal {sig_miss_marg / sig_tot:.3f} vs selective {sig_miss_sel / sig_tot:.3f}"


@lock("TE5", "penumbra correction: fixed-dictionary null world corrected to a small residue; identity holds on the signal run")
def te5():
    Xn, yn = _panel(n=700, seed=11, signal=False)
    repn = pv.certify(Xn, yn, q=0.10, profile="fast", seed=5, cart_cuts=False, n_harvest=0, interactions=False)
    p = repn.penumbra
    assert p["floor"] is not None and p["raw"] > 0
    assert p["m0"] >= 0.6 * repn.state["P"]
    assert p["corrected"] <= 0.35 * p["raw"]
    repa = pv.certify(Xn, yn, q=0.10, profile="fast", seed=5)
    pa = repa.penumbra
    assert pa["corrected"] == pytest.approx(max(0.0, pa["raw"] - pa["floor"])) and "adaptive" in pa["note"]
    rep, _ = _certify_pair()
    ps = rep.penumbra
    assert ps["corrected"] == pytest.approx(max(0.0, ps["raw"] - ps["floor"]))
    return f"fixed-dict null: raw {p['raw']:.1f} -> corrected {p['corrected']:.1f} (m0 {p['m0']:.0f}/{repn.state['P']}); adaptive residue {pa['corrected']:.0f} documented"


@lock("TE6", "ledger delta: exactly one carrier certificate per certified run, family and Gram descriptor aboard")
def te6():
    rep, bare = _certify_pair()
    led = ct.Ledger(_STATE["led_dir"])
    certs = led.certificates(kind="carrier")
    assert len(certs) == 1 and rep.certificate is not None
    c = certs[0]
    fam = c.guarantee.family
    assert fam["khat"] == bare.khat_ and len(fam["p"]) == fam["m"] == len(bare.core_)
    assert c.gram_context["n_support"] == len(bare.certified_) and c.gram_context["descriptor"]
    assert len(c.support) == len(bare.certified_)
    return f"1 carrier, |V|={len(c.support)}, gram {c.gram_context['descriptor'][:8]}"


@lock("TE7", "instrumented run lints clean: R1 PASS from the run record itself")
def te7():
    rep, _ = _certify_pair()
    fnd = pv.lint(rep)
    r1 = [r for r in fnd.rows if r.check == "R1"]
    assert r1 and all(r.verdict == "PASS" for r in r1)
    return "R1 PASS on instrumented lineage"


@lock("TE8", "report renders and serializes: manifest, corrected penumbra, book line, json schema")
def te8():
    rep, _ = _certify_pair()
    txt = rep.render()
    assert "manifest" in txt and "penumbra" in txt and "carrier certificate appended" in txt
    assert "engine vs certified core" in txt and rep.engine_vs_core()["lead"] is not None
    assert json.loads(rep.to_json())["engine_vs_core"]["lead"] == pytest.approx(rep.engine_vs_core()["lead"])
    js = json.loads(rep.to_json())
    assert js["schema"] == "provable/report/v1" and js["certificate_id"] == rep.certificate.id
    assert js["engine"]["certification"] == rep.engine["certification"]
    return "render + json stable"


@lock("TE9", "second run on the same panel is refused by the book: spent rows meet the new test rows")
def te9():
    rep, _ = _certify_pair()
    X, y = _STATE["X"], _STATE["y"]
    with pytest.raises(ct.FreshnessError):
        pv.certify(X, y, q=0.10, ledger=_STATE["led_dir"], profile="fast", seed=3)
    led = ct.Ledger(_STATE["led_dir"])
    assert len(led.certificates(kind="carrier")) == 1
    return "reuse of spent panel refused, book unchanged"


@lock("TE10", "atom card: est marginals on S2 and test only, header present, hand-checked, net line under declared unit cost")
def te10():
    rep, bare = _certify_pair()
    pc = rep.atom_card
    assert pc is not None and len(pc) == bare.khat_
    txt = rep.render()
    assert "atom card (est; marginal per-atom effects on S2 and held-out test, never the proposal rows; net across cells via rep.core_fit())" in txt
    row = pc.iloc[0]
    j = [bare.core_[p] for p in np.where(bare.cert_sel_)[0]][0]
    raw = bare.Z[:, j] * bare._sd[j] + bare._mu[j]
    defined = 0
    for tag, idx in (("s2", bare.idx_s2), ("test", bare.idx_test)):
        idx = np.asarray(idx, int)
        f = raw[idx] > 0.5
        yy = np.asarray(bare.y, float)[idx]
        assert row[f"fire_{tag}"] == pytest.approx(f.mean())
        if f.all() or not f.any():
            # a marginal against an empty complement is undefined; the card must say nan, not a number (A24)
            assert np.isnan(row[f"eff_{tag}"]) and np.isnan(row[f"se_{tag}"]) and np.isnan(row[f"agree_{tag}"])
            continue
        defined += 1
        assert row[f"eff_{tag}"] == pytest.approx(float(yy[f].mean() - yy[~f].mean()))
        assert 0.0 <= row[f"agree_{tag}"] <= 1.0
    assert defined >= 1
    X, y = _STATE["X"], _STATE["y"]
    repc = pv.certify(X, y, q=0.10, profile="fast", seed=3, unit_cost=0.25)
    r0 = repc.atom_card.iloc[0]
    if np.isnan(r0["eff_test"]):
        assert np.isnan(r0["net_test"])
    else:
        assert r0["net_test"] == pytest.approx(r0["direction"] * r0["eff_test"] - 0.5)
    assert "declared unit cost" in repc.render()
    return f"{len(pc)} atoms carded; {defined}/2 slices with a defined marginal hand-checked; net line checks"


@lock("TE11", "core_fit is the certified tier's fitted value, exactly, on own rows and on fresh rows")
def te11():
    rep, bare = _certify_pair()
    base = bare.cert_pred_all_[bare.refit_scope] if isinstance(bare.cert_pred_all_, dict) else bare.cert_pred_all_
    assert np.array_equal(rep.core_fit(), np.asarray(base, float))
    Xf = _STATE["X"].iloc[:100]
    assert np.allclose(rep.core_fit(Xf), bare.predict(Xf, model="certified"))
    return "core_fit == certified-tier predictor"


# ------------------------------------------------------------------ TF anatomy
@lock("TF1", "adapter.describe: objective, size, link-mapped intercept")
def tf1():
    cp, rp, Xt = _xgb_models()
    ad, _ = pv.load(cp)
    d = ad.describe()
    assert d["objective"] == "binary:logistic" and d["trees"] == 25
    assert 0 < d["base_score_stored"] < 1
    assert abs(d["base_score_margin"] - np.log(d["base_score_stored"] / (1 - d["base_score_stored"]))) < 1e-9
    return f"stored {d['base_score_stored']:.3f} -> margin {d['base_score_margin']:.3f}"


@lock("TF2", "the classifier spectrum is exact after the link fix: evaluate equals predict_margin")
def tf2():
    cp, rp, Xt = _xgb_models()
    ad, spec = pv.load(cp)
    diff = float(np.max(np.abs(spec.evaluate(Xt) - ad.predict_margin(Xt))))
    assert diff < 1e-4
    ad2, spec2 = pv.load(rp)
    diff2 = float(np.max(np.abs(spec2.evaluate(Xt) - ad2.predict_margin(Xt))))
    assert diff2 < 1e-4
    return f"clf max diff {diff:.2e}, reg {diff2:.2e}"


@lock("TF3", "unmapped objective refuses with code R-09")
def tf3():
    cp, rp, Xt = _xgb_models()
    ad, _ = pv.load(rp)
    ad.objective = "survival:cox"
    with pytest.raises(ValueError) as ei:
        ad._stored_to_margin(0.5)
    assert "R-09" in str(ei.value)
    from provable.evidence import REFUSAL_CODES
    assert "R-09" in REFUSAL_CODES
    return "R-09 registered and raised"


@lock("TF4", "families is one call: labeled quantities, ranked, no user loops")
def tf4():
    cp, rp, Xt = _xgb_models()
    ad, spec = pv.load(rp)
    df = spec.families(Xt, by="removal_cost", top=10)
    assert {"support", "magnitude", "removal_cost", "share", "joint_value"} <= set(df.columns)
    assert df["removal_cost"].is_monotonic_decreasing
    dff = spec.families(None, top=5)
    assert dff["removal_cost"].isna().all() and dff["magnitude"].is_monotonic_decreasing
    return f"{len(df)} families priced"


@lock("TF5", "the family describe prints all four quantities labeled (the F1 deliverable)")
def tf5():
    cp, rp, Xt = _xgb_models()
    ad, spec = pv.load(rp)
    fam = spec.family(feats=[1, 2])
    txt = spec.describe_family(fam, Xt)
    for label in ("magnitude", "removal cost", "share", "joint value", "ceiling", "not an importance"):
        assert label in txt
    m = spec.family_metrics(fam, Xt)
    assert m["magnitude"] == pytest.approx(sum(abs(c) for A, c in fam.items() if A))
    assert m["joint_value"] == pytest.approx(sum(c for A, c in fam.items() if A))
    return "four quantities, labeled"


@lock("TF6", "removal cost equals the engine's exact price on the same rows")
def tf6():
    cp, rp, Xt = _xgb_models()
    ad, spec = pv.load(rp)
    fam = {A: c for A, c in spec.family(feats=[1, 2]).items() if A}
    mass, _ = spec.data_mass(fam, Xt)
    m = spec.family_metrics(fam, Xt)
    assert m["removal_cost"] == pytest.approx(mass)
    var = float(np.var(ad.predict_margin(Xt)))
    assert m["share"] == pytest.approx(mass / var)
    return f"removal_cost {mass:.5f} == data_mass"


@lock("TF7", "as_column is the exact bridge: the built column equals the family component")
def tf7():
    cp, rp, Xt = _xgb_models()
    ad, spec = pv.load(rp)
    fam = spec.family(feats=[1, 2])
    col = spec.as_column(fam)
    want = spec.evaluate(Xt, atoms={A: c for A, c in fam.items() if A})
    assert np.array_equal(col(Xt), want)
    return f"exact on {len(Xt)} rows, {col.n_atoms} atoms"


@lock("TF8", "examine in one call: model + spectrum + families + transport share on a declared available set")
def tf8():
    cp, rp, Xt = _xgb_models()
    a = pv.examine(rp, Xt, available=["f0", "f3", "f4", "f5"], top=12)
    assert a.transport is not None and a.transport["unavailable_share"] > 0.25
    txt = a.render()
    assert "transport" in txt and "removal cost" in txt
    return f"outside share {a.transport['unavailable_share']:.0%}"


@lock("TF9", "missing data: exact mode refuses (R-07 semantics), declared mode prices per-family complete rows and says so")
def tf9():
    cp, rp, Xt = _xgb_models()
    ad, spec = pv.load(rp)
    Xn = Xt.copy()
    Xn[:60, 4] = np.nan
    fam = spec.family(feats=[1, 2])
    with pytest.raises(ValueError) as ei:
        spec.family_metrics(fam, Xn, data="exact")
    assert "R-07" in str(ei.value)
    m = spec.family_metrics(fam, Xn, data="complete-per-family")
    assert m["rows_used"] == len(Xt)
    fam2 = spec.family(involving=4, min_order=1)
    m2 = spec.family_metrics(fam2, Xn, data="complete-per-family")
    assert m2["rows_used"] == len(Xt) - 60
    a = pv.examine(rp, Xn, data="complete-per-family", top=5)
    assert "DECLARED" in a.render()
    b = pv.audit(rp, pv.Policy([pv.between("f1", "f2")]), Xn)
    assert b["verdict"]["state"] == "DECLINED" and b["refusals"][0]["code"] == "R-07"
    return "refusal intact, DECLARED mode row-accounted"


@lock("TF10", "groups hook: driver-style rollup sums match the family table")
def tf10():
    cp, rp, Xt = _xgb_models()
    groups = {"f0": "level", "f1": "pair", "f2": "pair", "f3": "other", "f4": "other", "f5": "other"}
    a = pv.examine(rp, Xt, groups=groups, top=50)
    assert a.groups is not None and len(a.groups)
    assert a.groups["removal_cost"].sum() == pytest.approx(a.families["removal_cost"].sum())
    return f"{len(a.groups)} group rows, mass conserved"


# ------------------------------------------------------------------ TG probes
@lock("TG1", "chunked probe equals the one-shot probe exactly")
def tg1():
    cp, rp, Xt = _xgb_models()
    ad, spec = pv.load(rp)
    th = ad.thresholds()
    ctx = pv.Spectrum.grid_contexts(th, ad.num_feature, 40, seed=0)
    full = pv.Spectrum.grid_probe(ad.predict_margin, th.get(1, []), th.get(2, []), 1, 2, ctx)
    chunked = pv.Spectrum.grid_probe(ad.predict_margin, th.get(1, []), th.get(2, []), 1, 2, ctx, tol_context=3)
    assert full == chunked
    return f"probe {full:.3e} both paths"


@lock("TG2", "enforce accepts a memory budget additively, prints its sizing, gates still pass")
def tg2(capsys=None):
    cp, rp, Xt = _xgb_models()
    out = os.path.join(tempfile.mkdtemp(prefix="enf_"), "reg_cut.json")
    b = pv.enforce(rp, pv.Policy([pv.between("f1", "f2")]), Xt[:250], Xt[250:], out, probe_contexts=30, probe_memory_budget_mb=1)
    assert b["verdict"]["state"] == "PASS"
    assert b["action"]["stock_diff_off"] == 0.0
    return "PASS under 1 MB probe budget"


@lock("TG3", "enforce on a classifier artifact: gates pass and the edited file round-trips its stored base_score space")
def tg3():
    cp, rp, Xt = _xgb_models()
    out = os.path.join(tempfile.mkdtemp(prefix="enf_"), "clf_cut.json")
    b = pv.enforce(cp, pv.Policy([pv.between("f1", "f2")]), Xt[:250], Xt[250:], out, probe_contexts=30)
    assert b["verdict"]["state"] == "PASS" and b["action"]["stock_diff_off"] == 0.0
    ad2, _ = pv.load(out)
    assert 0.0 < ad2.base_score_stored < 1.0
    return "classifier cut certified, stored space preserved"


# ------------------------------------------------------------------ TH conformance
FROZEN_SIGS = {
    "surface.load": ["path", "booster", "max_atoms"],
    "surface.audit": ["model_path", "policy", "X", "out_dir"],
    "surface.enforce": ["model_path", "policy", "X_train", "X_holdout", "out_path", "y_holdout", "out_dir", "probe_contexts", "seed"],
    "policy.between": ["a", "b"],
    "policy.involving": ["f", "min_order"],
    "Spectrum.evaluate": ["self", "X", "atoms"],
    "Spectrum.family": ["self", "feats", "involving", "min_order"],
    "Spectrum.data_mass": ["self", "fam", "X"],
    "Spectrum.profile": ["self", "fam", "feat"],
    "Spectrum.grid_probe": ["predict", "thr_i", "thr_j", "i", "j", "contexts", "tol_context"],
    "CertifiedDiscovery.run": ["self", "X", "y"],
    "CertifiedDiscovery.predict": ["self", "X", "model"],
    "evidence.verify_id": ["bundle"],
}


@lock("TH1", "released signatures unchanged; parameters added since 0.9.0 are appended and keyword-only")
def th1():
    import provable.surface as sf
    import provable.policy as po
    import provable.evidence as ev
    objs = {"surface.load": sf.load, "surface.audit": sf.audit, "surface.enforce": sf.enforce, "policy.between": po.between, "policy.involving": po.involving, "Spectrum.evaluate": pv.Spectrum.evaluate, "Spectrum.family": pv.Spectrum.family, "Spectrum.data_mass": pv.Spectrum.data_mass, "Spectrum.profile": pv.Spectrum.profile, "Spectrum.grid_probe": pv.Spectrum.grid_probe, "CertifiedDiscovery.run": pv.CertifiedDiscovery.run, "CertifiedDiscovery.predict": pv.CertifiedDiscovery.predict, "evidence.verify_id": ev.verify_id}
    added = 0
    for name, frozen in FROZEN_SIGS.items():
        params = list(inspect.signature(objs[name]).parameters.values())
        got = [p.name for p in params]
        assert got[: len(frozen)] == frozen, (name, got)
        late = [p for p in params[len(frozen):] if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
        assert all(p.kind == inspect.Parameter.KEYWORD_ONLY for p in late), (name, [(p.name, str(p.kind)) for p in late])
        added += len(late)
    return f"{len(FROZEN_SIGS)} signatures conform; {added} parameter(s) added since 0.9.0, all keyword-only"


@lock("TH2", "the released engine test suite still passes byte-for-byte")
def th2():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.abspath("src")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_contracts.py", "tests/test_firewall.py", "tests/test_protocol.py", "tests/test_verifier_mutations.py", "-q"], capture_output=True, text=True, env=env, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert r.returncode == 0, r.stdout[-800:]
    tail = [l for l in r.stdout.strip().splitlines() if "passed" in l][-1]
    return tail.strip()


@lock("TH3", "the verifier logic holds on the bundle inside an artifact certificate")
def th3():
    cp, rp, Xt = _xgb_models()
    b = pv.audit(rp, pv.Policy([pv.involving("f1")]), Xt)
    c = from_evidence(b)
    assert pv.verify_id(c.artifact) and c.id == b["certificate_id"]
    d = to_dict(c)
    c2 = from_dict(d)
    assert pv.verify_id(c2.artifact) and c2.id == c.id
    return "verify_id survives the round trip"


@lock("TH4", "render labels: removal cost added additively, bundle hashing untouched")
def th4():
    cp, rp, Xt = _xgb_models()
    b = pv.audit(rp, pv.Policy([pv.between("f1", "f2")]), Xt)
    txt = pv.render_text(b)
    assert "removal cost" in txt and "data_mass" in txt
    from provable.evidence import bundle_hash
    body = {k: v for k, v in b.items() if k != "certificate_id"}
    assert b["certificate_id"] == "PV-" + bundle_hash(body)[:16].upper()
    return "label additive, id law intact"


@lock("TH5", "the artifact scripts' live repro surface holds on this release")
def th5():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_repro_surface import ARCHIVE_ONLY, LIVE_ATTRS
    _, bare = _certify_pair()
    missing = [a for a in LIVE_ATTRS if not hasattr(bare, a)]
    shipped = [a for a in ARCHIVE_ONLY if hasattr(bare, a)]
    assert not missing and not shipped
    sig = list(inspect.signature(pv.CertifiedDiscovery.__init__).parameters)
    assert "seed" in sig and "verbose" in sig
    return f"{len(LIVE_ATTRS)} live attrs present; {len(ARCHIVE_ONLY)} archive-only stay archived"


@lock("TH6", "the intercept is written in the host's form: an intercept atom folded by append_rules is stored plain on a plain host (what xgboost before 3.1 writes and reads) and bracketed on a bracketed one (xgboost 3.1 on), as a float32 with 9 significant digits; a stock reload by the running xgboost moves the margin by the folded constant, for squared error and logistic")
def th6():
    import xgboost as xgb
    from provable.adapters_xgb import XGBAdapter
    rng = np.random.default_rng(6)
    X = rng.normal(size=(300, 4))
    modern = tuple(int(v) for v in xgb.__version__.split(".")[:2]) >= (3, 1)
    out = []
    for obj, y in (("reg:squarederror", X[:, 0] + rng.normal(size=300)), ("binary:logistic", (X[:, 0] + rng.normal(size=300) > 0.4).astype(float))):
        bst = xgb.train({"objective": obj, "max_depth": 2, "eta": 0.3, "seed": 0}, xgb.DMatrix(X, label=y), num_boost_round=4)
        assert str(XGBAdapter(booster=bst).raw["learner"]["learner_model_param"]["base_score"]).startswith("[") == modern
        for form in ("plain", "bracketed") if modern else ("plain",):
            ad = XGBAdapter(booster=bst)
            lp = ad.raw["learner"]["learner_model_param"]
            stored = str(lp["base_score"]).strip("[]")
            lp["base_score"] = f"[{stored}]" if form == "bracketed" else stored
            raw = ad.append_rules({frozenset(): 0.37})
            written = raw["learner"]["learner_model_param"]["base_score"]
            assert written.startswith("[") == (form == "bracketed")
            digits = written.strip("[]").lower().split("e")[0].replace("-", "").replace(".", "").lstrip("0")
            assert len(digits) <= 9
            p = os.path.join(tempfile.mkdtemp(prefix="bs_"), "m.json")
            XGBAdapter.save_raw_json(raw, p)
            d = bst.predict(xgb.DMatrix(X[:50]), output_margin=True) - xgb.Booster(model_file=p).predict(xgb.DMatrix(X[:50]), output_margin=True)
            assert np.abs(d - 0.37).max() <= 1e-5
            out.append(f"{obj.split(':')[1]} {form} {written}")
    return f"xgboost {xgb.__version__}: " + "; ".join(out)


# ------------------------------------------------------------------ TJ grade
def _grade_world(seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(3200, 6)), columns=[f"g{i}" for i in range(6)])
    y = 1.0 * (X["g0"] >= 0.5) - 0.9 * ((X["g1"] >= 0) & (X["g2"] >= 0)) + rng.normal(size=3200)
    return X, y


def _graded_book():
    if "gcert" not in _STATE:
        X, y = _grade_world(1)
        d = tempfile.mkdtemp(prefix="gbook_")
        led = ct.Ledger(d, budget=0.40)
        rep = pv.certify(X, y, q=0.10, ledger=led, seed=2, dataset_tag="gdemo")
        assert rep.certificate is not None
        assert any(a.startswith("g0>=") for a in rep.certificate.guarantee.family["support_ordered"])
        _STATE.update(gX=X, gy=y, gled_dir=d, gcert=rep.certificate)
    return _STATE["gcert"], _STATE["gled_dir"]


@lock("TJ1", "certificate atom specs evaluate exactly: spec indicator equals the de-standardized dictionary column")
def tj1():
    cert, _ = _graded_book()
    X = _STATE["gX"]
    from provable.certificates.records import eval_atom_spec
    fam = cert.guarantee.family
    for nm, sp in zip(fam["support_ordered"], fam["atoms_spec"]):
        v = eval_atom_spec(sp, X)
        assert set(np.unique(v)) <= {0.0, 1.0} and 0 < v.mean() < 1
    return f"{len(fam['atoms_spec'])} specs evaluable, 0/1, interior"


def _planted_g0(gr):
    """The certified g0 cut that carries the planted step: the largest-support
    `g0>=` cell. How many g0 cuts the certifier harvests beside it is an
    engine-version artifact (A24), so the locks assert the planted cell's
    direction and verdict, never a cell count."""
    g0 = gr.cells[gr.cells["atom"].str.startswith("g0>=")]
    assert len(g0) >= 1
    return g0.sort_values("n_fired", ascending=False).iloc[0], len(g0)


@lock("TJ2", "grade on fresh rows: the planted g0 cell (largest-support g0 cut, committed direction +1) CONFIRMED, book charged, grades section rendered")
def tj2():
    cert, d = _graded_book()
    Xn, yn = _grade_world(77)
    before = ct.Ledger(d).statement().charged
    gr = pv.grade(cert, Xn, yn, ledger=d, level=0.05, dataset_tag="gdemo2")
    assert gr.admission_verdict == "FRESH" and gr.bar == pytest.approx(gr.level / gr.k)
    strong, n_g0 = _planted_g0(gr)
    assert strong["direction"] == 1 and strong["verdict"] == "CONFIRMED"
    assert gr.summary()["confirmed"] >= 1
    st = ct.Ledger(d).statement()
    assert st.charged == pytest.approx(before + gr.level)
    assert "grades (budget-bearing" in st.render()
    assert "REVERSED is significant evidence against the original direction" in gr.render()
    return f"{gr.summary()} at bar {gr.bar:.4g}; planted cell {strong['atom']} CONFIRMED ({n_g0} g0 cuts certified); charged +{gr.level:g}"


@lock("TJ3", "reversal detected: sign-flipped world grades the planted g0 cell REVERSED, not merely unconfirmed")
def tj3():
    cert, d = _graded_book()
    Xn, yn = _grade_world(78)
    gr = pv.grade(cert, Xn, -yn, ledger=d, level=0.05, dataset_tag="gdemo3")
    strong, _ = _planted_g0(gr)
    assert strong["direction"] == 1
    assert strong["verdict"] == "REVERSED" and strong["p_reverse"] <= gr.bar
    return f"planted cell {strong['atom']} REVERSED (p_reverse {strong['p_reverse']:.2e})"


@lock("TJ4", "grade on spent rows refuses; the book is unchanged")
def tj4():
    cert, d = _graded_book()
    n_ops_before = sum(1 for _ in open(os.path.join(d, "ledger.jsonl")))
    with pytest.raises(ct.FreshnessError):
        pv.grade(cert, _STATE["gX"], _STATE["gy"], ledger=d, dataset_tag="gdemo")
    assert sum(1 for _ in open(os.path.join(d, "ledger.jsonl"))) == n_ops_before
    return "spent-panel grade refused pre-write"


@lock("TJ5", "reservation grading: bar = alpha/|V|, hold converts to charge, outside-vocabulary refused")
def tj5():
    cert, d = _graded_book()
    led = ct.Ledger(d)
    fam = cert.guarantee.family
    vocab = list(fam["support_ordered"]) + [f"decoy{i}" for i in range(7)]
    rid = led.reserve(0.06, note="pre-registered grading", vocabulary=vocab)
    st0 = led.statement()
    Xn, yn = _grade_world(79)
    gr = pv.grade(cert, Xn, yn, ledger=d, reservation=rid, dataset_tag="gdemo4")
    assert gr.bar == pytest.approx(0.06 / len(vocab))
    st1 = ct.Ledger(d).statement()
    assert st1.reserved == pytest.approx(st0.reserved - 0.06)
    assert st1.charged == pytest.approx(st0.charged + 0.06)
    assert st1.remaining == pytest.approx(st0.remaining)
    rid2 = ct.Ledger(d).reserve(0.05, note="wrong vocab", vocabulary=["nope@1"])
    with pytest.raises(ValueError):
        pv.grade(cert, Xn, yn, ledger=d, reservation=rid2, dataset_tag="gdemo5")
    return f"bar {gr.bar:.4g} = 0.06/{len(vocab)}; hold converted"


# ------------------------------------------------------------------ runner
def run_all():
    rows, misses = [], 0
    for lid, target, fn in LOCKS:
        try:
            obs = fn()
            rows.append([lid, target[:76], str(obs)[:52], "PASS"])
        except Exception as e:
            misses += 1
            rows.append([lid, target[:76], f"{type(e).__name__}: {e}"[:52], "MISS"])
    print(tabulate(rows, headers=["lock", "target", "observed", "verdict"], tablefmt="psql"))
    print(f"Locks: {len(LOCKS)} graded, {len(LOCKS) - misses} PASS, {misses} MISS")
    return misses




# ------------------------------------------------------------------ TK refine
def _existence_pair():
    led = ct.Ledger(tempfile.mkdtemp(prefix="tkref"), budget=0.10)
    S = ct.Lineage.of(ct.slice_from_frame(frame("dataset_1", 1), "dataset_1", range(0, 16)))
    P = ct.Lineage.of(ct.slice_from_frame(frame("dataset_1", 1), "dataset_1", range(16, 24)))
    g = ct.Guarantee(kind=ct.GuaranteeKind.FDR, level=0.05, family={"m": 10})
    par = ct.issue(claim="existence: something in v0..v9", guarantee=g, ell_test=S, proposer_rows=P, artifact={}, support=tuple(f"v{i}" for i in range(10)), gram_context={}, kind=ct.CertificateKind.EXISTENCE)
    led.register(par)
    return led, par


@lock("TK1", "refine licensed: carrier issued at bar level/|V|, predicate holds, account uncharged, statement section renders")
def tk1():
    led, par = _existence_pair()
    cid = led.refine(par, "v3", p=0.004, direction=1)
    kids = led.certificates(kind="carrier")
    assert len(kids) == 1 and kids[0].support == frozenset({"v3"})
    assert ct.refines(kids[0], par, ledger=led)
    st = led.statement()
    assert st.charged == pytest.approx(0.05)
    assert len(st.refinements) == 1 and st.refinements[0]["parent"] == content_id(to_dict(par)) and st.refinements[0]["id"] == cid
    assert st.refinements[0]["bar"] == pytest.approx(0.005) and st.refinements[0]["multiplicity"] == 10
    assert st.n_active == 2
    txt = st.render()
    assert "-- refinements (licensed carriers under an existence certificate" in txt
    return f"carrier {cid[:14]} at bar 0.005; charged stays 0.0500"


@lock("TK2", "refine refuses above the licensed bar, naming bar, level and multiplicity")
def tk2():
    led, par = _existence_pair()
    with pytest.raises(ValueError, match="above the licensed bar"):
        led.refine(par, "v3", p=0.006)
    assert led.certificates(kind="carrier") == []
    return "p 0.006 vs bar 0.005 refused; nothing written"


@lock("TK3", "refine refuses members outside V and non-existence parents")
def tk3():
    led, par = _existence_pair()
    with pytest.raises(ValueError, match="outside the licensed vocabulary"):
        led.refine(par, "w9", p=0.001)
    cid = led.refine(par, "v0", p=0.001)
    with pytest.raises(ValueError, match="must be an existence certificate"):
        led.refine(cid, "v0", p=0.0001)
    return "member and kind guards hold"


@lock("TK4", "a refine-issued pair lints R6 PASS with the shipped price text")
def tk4():
    led, par = _existence_pair()
    cid = led.refine(par, "v7", p=0.0049)
    child = led.certificates(kind="carrier")[0]
    rr = ct.RunRecord(name="tk4 declared", instrumented=False, q=0.05, m=10, khat=1, test_rows=par.ell_test, refinements=[ct.RefinementDecl(child=child, parent=par, label="tk4 pair")])
    f = ct.run_checks(rr, ledger=led)
    r6 = [r for r in f.rows if r.check == "R6"]
    assert len(r6) == 1 and r6[0].verdict == "PASS" and "licensed at the charged multiplicity" in r6[0].price
    return "R6 PASS on the op's own output"


# ------------------------------------------------------------------ TL purchase
@lock("TL1", "purchase rows leg: weakest requirement equals the engine N3_hat; per-atom formula hand-checked")
def tl1():
    rep, bare = _certify_pair()
    po = pv.purchase_order(rep)
    assert po.kind == "rows" and np.isclose(po.requirement, float(bare.diag_["N3_hat"]), rtol=1e-9)
    assert po.weakest["t"] == pytest.approx(float(bare.diag_["min_cert_t"]))
    r0 = po.rows.iloc[-1]
    assert r0["rows_for_level"] == pytest.approx(float(bare.diag_["zbar_core"]) ** 2 * len(bare.idx_s2) / float(r0["t"]) ** 2)
    assert po.n_s2 == len(bare.idx_s2) and po.n_train == len(bare.idx_train)
    return f"requirement {po.requirement:.4g} == N3_hat; {len(po.rows)} atoms priced"


@lock("TL2", "purchase order arithmetic and render: additional = max(0, requirement - N_train); columns-leg pointer when no promise")
def tl2():
    rep, _ = _certify_pair()
    po = pv.purchase_order(rep)
    assert po.additional_rows == int(round(max(0.0, po.requirement - po.n_train)))
    assert (po.rows["additional"] == np.maximum(0.0, po.rows["rows_for_level"] - po.n_train)).all()
    assert list(po.rows["rows_for_level"]) == sorted(po.rows["rows_for_level"], reverse=True)
    txt = po.render()
    assert "-- purchase order (rows leg" in txt and "additional rows" in txt and "pass promise=" in txt
    return "order arithmetic and render hold"


@lock("TL3", "purchase without a promise: rows leg only, no columns leg, state.nu unavailable (post-fold behavior)")
def tl3():
    rep, _ = _certify_pair()
    po = pv.purchase_order(rep)
    assert po.columns is None and po.kind == "rows"
    with pytest.raises(AttributeError, match="pass promise="):
        _ = po.state.nu
    return "rows-leg-only order carries no columns leg"


@lock("TL4", "purchase refuses on khat=0: nothing certified, no level to hold")
def tl4():
    r = np.random.default_rng(0)
    X = pd.DataFrame(r.normal(size=(1200, 6)), columns=[f"v{i}" for i in range(6)])
    y = pd.Series(r.normal(size=1200))
    rep = pv.certify(X, y, q=0.10, profile="fast", seed=0)
    assert rep.claims is None or len(rep.claims) == 0
    with pytest.raises(ValueError, match="nothing certified"):
        pv.purchase_order(rep)
    return "khat=0 refusal holds"


# ------------------------------------------------------------------ TM columns leg (fold)
def _binary_certpair(seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(6000, 8)), columns=[f"x{i}" for i in range(8)])
    base = (X["x0"] >= 0.3).astype(float) * 0.85 + ((X["x1"] >= 0) & (X["x2"] >= 0)).astype(float) * 0.10
    y = (rng.random(6000) < np.clip(base + 0.02, 0, 1)).astype(float)
    rep = pv.certify(X, y, q=0.10, ledger=None, seed=0, profile="fast")
    return rep, X, y


@lock("TM1", "columns leg attaches on a promise; feasible-dark present; po.state.nu == SieveState nu_deb")
def tm1():
    from provable.purchase._engine.state import SieveState
    rep, X, y = _binary_certpair()
    po = pv.purchase_order(rep, promise=pv.PurchasePromise(M0=0.02, rho=0.10, delta=0.10))
    assert po.kind == "rows+columns" and isinstance(po.columns, pv.purchase.ColumnsLeg)
    assert po.columns.feasible_dark["J"] >= 1
    W = pd.get_dummies(rep.cd.X_, dtype=float).to_numpy()
    st = SieveState(q=0.10).fit(np.asarray(y, int), W, run_provable=False)
    assert po.state.nu == pytest.approx(st.nu_deb)
    return f"columns leg attached; nu_deb={po.state.nu:.4g}"


@lock("TM2", "terminal-remainder guard fires iff M0 exceeds pihat(1-pihat)nu_deb; excess quoted; pricing suppressed")
def tm2():
    rep, X, y = _binary_certpair()
    po = pv.purchase_order(rep, promise=pv.PurchasePromise(M0=0.02, rho=0.10))
    ceil = po.columns.reachability_ceiling
    assert not po.columns.terminal_remainder["fires"] and po.columns.guaranteed_reliability is not None
    po2 = pv.purchase_order(rep, promise=pv.PurchasePromise(M0=ceil * 10 + 1.0, rho=0.10))
    tr = po2.columns.terminal_remainder
    assert tr["fires"] and tr["remainder"] == pytest.approx(po2.columns.promise.M0 - ceil, rel=1e-9)
    assert po2.columns.guaranteed_reliability is None and po2.columns.r_min is None
    return f"guard clears at M0<={ceil:.4g}, fires above"


@lock("TM3", "PG pricing hand-check: lam_A, reliability-law on the mass-scale ceiling, rows-route match the engine; law and guard agree at the boundary")
def tm3():
    from provable.purchase._engine.state import SieveState
    from provable.purchase._engine.acquire import Acquisition
    from provable.purchase._engine.envelope import Envelope
    rep, X, y = _binary_certpair()
    pr = pv.PurchasePromise(M0=0.02, rho=0.10, delta=0.10)
    po = pv.purchase_order(rep, promise=pr)
    W = pd.get_dummies(rep.cd.X_, dtype=float).to_numpy()
    st = SieveState(q=0.10).fit(np.asarray(y, int), W, run_provable=False)
    acq = Acquisition(st, Envelope(st.hbar, st.one_minus_d2))
    assert po.columns.guaranteed_reliability["lam2"] == pytest.approx(acq.lambda_A(pr.M0, pr.rho, pr.delta)["lam2"])
    ceil = st.pihat * (1 - st.pihat) * max(st.nu_deb, 0.0)
    assert po.columns.reliability_law_lam2 == pytest.approx(acq.reliability_law(pr.M0, ceil))
    assert po.columns.rows_route == acq.n_req_rows(st.nu_deb, pr.rho, pr.delta)
    assert acq.reliability_law(0.999 * ceil, ceil) < 1e-4 and acq.reliability_law(ceil, ceil) == 0.0
    po_edge = pv.purchase_order(rep, promise=pv.PurchasePromise(M0=ceil * 1.001, rho=pr.rho))
    assert po_edge.columns.terminal_remainder["fires"] and po_edge.columns.reliability_law_lam2 is None
    return "reliability, reliability-law (mass-scale ceiling) and rows-route match; law vanishes exactly where the guard fires"


@lock("TM4", "no promise: no columns leg, rows leg unchanged, po.state.nu raises")
def tm4():
    rep, X, y = _binary_certpair()
    po = pv.purchase_order(rep)
    assert po.columns is None and po.kind == "rows" and po.requirement > 0
    with pytest.raises(AttributeError):
        _ = po.state.nu
    return "rows-leg-only order carries no columns leg"


@lock("TM5", "r_min priced only when kappa and B are declared; matches the engine r_min")
def tm5():
    from provable.purchase._engine.state import SieveState
    from provable.purchase._engine.acquire import Acquisition
    from provable.purchase._engine.envelope import Envelope
    rep, X, y = _binary_certpair()
    po_no = pv.purchase_order(rep, promise=pv.PurchasePromise(M0=0.02, rho=0.10))
    assert po_no.columns.r_min is None
    pr = pv.PurchasePromise(M0=0.02, rho=0.10, kappa=3, B=1.0)
    po = pv.purchase_order(rep, promise=pr)
    W = pd.get_dummies(rep.cd.X_, dtype=float).to_numpy()
    st = SieveState(q=0.10).fit(np.asarray(y, int), W, run_provable=False)
    acq = Acquisition(st, Envelope(st.hbar, st.one_minus_d2))
    exp = acq.r_min(support_eigs=st.evals, s=len(st.evals), kappa=3, B=1.0, M0=pr.M0)
    assert po.columns.r_min["r_min"] == exp["r_min"]
    return f"r_min={po.columns.r_min['r_min']} matches engine"


@lock("TM6", "purchase_refit is ledger-free and manifest-locked: good manifest reproduces the baseline support, wrong manifest refuses, no book touched")
def tm6():
    import os
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(6000, 8)), columns=[f"x{i}" for i in range(8)])
    base = (X["x0"] >= 0.3).astype(float) * 0.85 + ((X["x1"] >= 0) & (X["x2"] >= 0)).astype(float) * 0.10
    y = (rng.random(6000) < np.clip(base + 0.02, 0, 1)).astype(float)
    bk = tempfile.mkdtemp(prefix="tmbase")
    baseline = pv.certify(X, y, q=0.10, ledger=os.path.join(bk, "book"), seed=0, profile="fast")
    cert, man = baseline.certificate, baseline.manifest
    d = tempfile.mkdtemp(prefix="tmrefit")
    cwd = os.getcwd()
    os.chdir(d)
    try:
        r2 = pv.purchase_refit(cert, X, y, expect_manifest={"x_sha": man["x_sha"]}, q=0.10, seed=0, profile="fast")
        assert r2.certificate is None and set(r2.claims["atom"]) == set(cert.guarantee.family["support_ordered"])
        with pytest.raises(pv.ManifestMismatch):
            pv.purchase_refit(cert, X, y, expect_manifest={"x_sha": "0" * 16}, q=0.10, seed=0, profile="fast")
        assert not os.path.exists("panel") and not os.path.exists("ledger.jsonl") and not os.listdir(".")
    finally:
        os.chdir(cwd)
    return "ledger-free manifest-locked refit reproduces the baseline support, no book written"


@lock("TM7", "columns leg refuses a non-binary target by name (binarization is a registration decision)")
def tm7():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(6000, 8)), columns=[f"x{i}" for i in range(8)])
    yv = 0.8 * (X["x0"] >= 0.3) + rng.normal(size=6000)
    rep = pv.certify(X, yv, q=0.10, ledger=None, seed=0, profile="fast")
    if rep.claims is None or not len(rep.claims):
        return "no certified atom on the continuous panel; refusal path not reachable this seed"
    with pytest.raises(ValueError, match="binary target"):
        pv.purchase_order(rep, promise=pv.PurchasePromise(M0=0.02))
    return "non-binary target refused before pricing"


@lock("TM_BOARD", "folded AC-series acquisition suite: synthetic worlds W-A..W-R, scientific locks all PASS")
def tm_board():
    import importlib.util
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "acquisition", "worlds.py")
    spec = importlib.util.spec_from_file_location("acq_worlds", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = mod.Grader().grade_all()
    assert r["miss"] == 0 and r["pass"] >= 34, f"acquisition board: {r['pass']} pass, {r['miss']} miss"
    return f"acquisition board {r['pass']}/{r['pass']} PASS"


# ======================================================================
# TN / TO / TP: the inquiry layer (shadow, characterize, screen)
# ======================================================================
from provable.inquiry import (Shadow, characterize, cluster_ids, deff_cluster, m_req as _inq_mreq,
                              screen as _inq_screen, shadow as _inq_shadow, xi as _xi, xi_inv as _xi_inv,
                              _lambda_hat_stat, _orth_slice)
from scipy.stats import chi2 as _chi2, kstest as _kstest


def _inq_world():
    if "inq" not in _STATE:
        rng = np.random.default_rng(0)
        n = 9000
        X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"x{i}" for i in range(8)])
        base = (X["x0"] >= 0.3).astype(float) * 0.80 + ((X["x1"] >= 0) & (X["x2"] >= 0)).astype(float) * 0.12
        y = (rng.random(n) < np.clip(0.05 + base, 0, 1)).astype(float)
        clk = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
        led = tempfile.mkdtemp(prefix="inqbook_")
        fit = np.arange(0, 6000)
        rep = pv.certify(X.iloc[fit], y[fit], q=0.10, ledger=led, seed=0, profile="fast",
                         dataset_tag="dataset_1", clock=(clk[fit], clk[fit] + pd.Timedelta(hours=1)))
        _STATE["inq"] = (rep, X, y, clk)
    return _STATE["inq"]


def _inq_slice(rep, X, y, clk, lo=6002, hi=9000):
    return _inq_shadow(rep, X.iloc[lo:hi].reset_index(drop=True), y[lo:hi], clk[lo:hi])


def _mk_sh(e, clock, y=None, p=None, cluster="day", tag="dataset_2"):
    rows = {"clock": pd.Series(clock)}
    if y is not None:
        rows["y"] = y
    if p is not None:
        rows["p_hat"] = p
    rows["e"] = e
    df = pd.DataFrame(rows)
    cid = cluster_ids(df["clock"], cluster)
    d = deff_cluster(np.asarray(e, float), cid)
    diffs = df["clock"].diff().dropna()
    step = diffs[diffs > pd.Timedelta(0)].median()
    return Shadow(rows=df, manifest={"x_sha": "aa", "y_sha": "yy", "dictionary_sha": "dd", "config_sha": "cc", "seed": 0},
                  dataset_tag=tag, estimand="E2-realized", era=(df["clock"].iloc[0], df["clock"].iloc[-1]),
                  khat=1, base_rate=None, var=float(np.mean(np.asarray(e) ** 2)), mean_shift=float(np.mean(e)),
                  deff=d, m=len(df), m_eff=len(df) / max(d, 1.0), cluster=cluster, clock_step=step, tag=tag)


def _host(rng, m, rho, J):
    w = rng.normal(size=(m, J))
    Z = rho * w[:, 0] + np.sqrt(1 - rho * rho) * rng.normal(size=m)
    return Z, w


@lock("TN1", "shadow refuses overlap by row identity: same-fingerprint slice refused; disjoint bytes inside the fitted clock envelope under the same tag refused; a clean later slice builds")
def tn1():
    rep, X, y, clk = _inq_world()
    with pytest.raises(ValueError, match="overlap by row identity"):
        _inq_shadow(rep, X.iloc[:6000], y[:6000], clk[:6000])
    with pytest.raises(ValueError, match="clock window"):
        _inq_shadow(rep, X.iloc[100:200].reset_index(drop=True), y[100:200], clk[100:200])
    sh = _inq_slice(rep, X, y, clk)
    assert sh.m == 2998 and sh.khat == rep.cd.khat_ and sh.khat > 0
    return f"both refusals fire by name; clean slice builds with m={sh.m}"


@lock("TN2", "khat = 0 falls back to the training base rate and the Shadow says so")
def tn2():
    rep = None
    for seed in range(6):
        rng = np.random.default_rng(400 + seed)
        Xn = pd.DataFrame(rng.normal(size=(4000, 6)), columns=[f"x{i}" for i in range(6)])
        yn = (rng.random(4000) < 0.42).astype(float)
        r = pv.certify(Xn, yn, q=0.10, ledger=None, seed=seed, profile="fast", dataset_tag="dataset_2")
        if int(r.cd.khat_) == 0:
            rep = r
            break
    assert rep is not None, "no khat=0 seed found"
    rng = np.random.default_rng(999)
    Xs = pd.DataFrame(rng.normal(size=(800, 6)), columns=[f"x{i}" for i in range(6)])
    ys = (rng.random(800) < 0.42).astype(float)
    sh = _inq_shadow(rep, Xs, ys, pd.date_range("2027-01-01", periods=800, freq="h", tz="UTC"))
    assert sh.khat == 0 and sh.base_rate == pytest.approx(float(np.mean(rep.cd.y)))
    assert np.allclose(sh.rows["p_hat"], sh.base_rate) and "training base rate" in sh.render()
    return f"base-rate fallback at {sh.base_rate:.4f}, stated in the read"


@lock("TN3", "join refuses mixed clocks by name; nesting clocks aggregate to the coarsest and align on common rows")
def tn3():
    clk_h = pd.date_range("2026-03-01", periods=480, freq="h", tz="UTC")
    clk_90 = pd.date_range("2026-03-01", periods=320, freq="90min", tz="UTC")
    clk_d = pd.date_range("2026-03-01", periods=20, freq="D", tz="UTC")
    rng = np.random.default_rng(2)
    a = _mk_sh(rng.normal(size=480) * 0.3, clk_h)
    b = _mk_sh(rng.normal(size=320) * 0.3, clk_90)
    c = _mk_sh(rng.normal(size=20) * 0.3, clk_d)
    with pytest.raises(ValueError, match="mixed clocks"):
        Shadow.join([a, b])
    fr = Shadow.join([a, c])
    assert fr.is_frame and fr.m == 20 and fr.clock_step == pd.Timedelta("1D")
    return "90min-into-1h refused; 1h-into-1D nests to 20 common daily rows"


@lock("TN4", "mean shift equals the mean residual and the second moment its mean square, to 1e-12")
def tn4():
    rep, X, y, clk = _inq_world()
    sh = _inq_slice(rep, X, y, clk)
    e = sh.rows["e"].to_numpy()
    assert abs(sh.mean_shift - float(np.mean(e))) <= 1e-12
    assert abs(sh.var - float(np.mean(e * e))) <= 1e-12
    return f"mean shift {sh.mean_shift:+.6g}, ||e||^2_n {sh.var:.6g}, both exact"


@lock("TO1", "exact ledger identity: per family, sum beta_hat^2 equals the fitted drop to 1e-12 (Thm 7.3(i))")
def to1():
    rep, X, y, clk = _inq_world()
    sh = _inq_slice(rep, X, y, clk)
    rng = np.random.default_rng(3)
    arch = {"nb_state": {"values": 0.3 * sh.rows["e"].to_numpy() + rng.normal(size=sh.m), "lag_class": "post-decision"}}
    ch = characterize(sh, atoms={"archive": arch})
    for r in ch.ledger.itertuples():
        s2 = float((ch.profiles.query("family == @r.family")["beta"] ** 2).sum())
        assert abs(s2 - r.explained) <= 1e-12 * max(1.0, sh.var)
    return f"{len(ch.ledger)} families, every explained mass equals its profile sum exactly"


@lock("TO2", "pure-noise shadow: no deterministic-family atom exceeds its exact hypothesis-free band at level 0.05 in >= 95% of 100 seeds (Prop 2.8)")
def to2():
    clean = 0
    for s in range(100):
        rng = np.random.default_rng(1000 + s)
        n = 1200
        clk = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
        yv = (rng.random(n) < 0.5).astype(float)
        ch = characterize(_mk_sh(yv - 0.5, clk, y=yv, p=np.full(n, 0.5)), level=0.05)
        det = ch.profiles[ch.profiles["grade"] == "exact"]
        clean += int(not det["exceeds"].any())
    assert clean >= 95, f"{clean}/100 clean"
    return f"{clean}/100 seeds fully inside the hypothesis-free bands (Azuma is conservative)"


@lock("TO3", "a planted Fourier line is recovered at its period as the strongest temporal atom, outside its band")
def to3():
    rng = np.random.default_rng(4)
    n = 2000
    t = np.arange(n)
    e = 0.15 * np.sin(2 * np.pi * t / 168.0) + 0.2 * rng.normal(size=n)
    ch = characterize(_mk_sh(e, pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")), level=0.05)
    st = ch.profiles[(ch.profiles["family"] == "temporal (Fourier)") & ch.profiles["strongest"]].iloc[0]
    assert st["atom"].startswith("week") and bool(st["exceeds"])
    return f"strongest line {st['atom']}, beta {st['beta']:+.4f} outside the exact band {st['band']:.4f}"


@lock("TO4", "planted AR(1) shadow: the persistence ledger covers the truth at all four lags in >= 92% of 120 seeds under the uniform long-run bands, the half-life verdict reads fast-decaying, and a phi=0.97 plant reads persistent (Thm 8.5, 8.6(iv))")
def to4():
    cov = fast = 0
    for s in range(120):
        rng = np.random.default_rng(2000 + s)
        n, phi = 1600, 0.55
        z = np.zeros(n)
        eps = rng.normal(size=n) * 0.2 * np.sqrt(1 - phi * phi)
        for t in range(1, n):
            z[t] = phi * z[t - 1] + eps[t]
        ch = characterize(_mk_sh(z, pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")), level=0.05)
        cov += int(all(r.lo <= phi ** r.lag <= r.hi for r in ch.persistence.itertuples()))
        fast += int(ch.verdict_persistence == "fast-decaying")
    assert cov >= 110 and fast >= 114, f"coverage {cov}/120, fast {fast}/120"
    rng = np.random.default_rng(7)
    n, phi = 3000, 0.97
    z = np.zeros(n)
    eps = rng.normal(size=n) * 0.2 * np.sqrt(1 - phi * phi)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + eps[t]
    chp = characterize(_mk_sh(z, pd.date_range("2018-01-01", periods=n, freq="D", tz="UTC")), level=0.05)
    assert chp.verdict_persistence == "persistent"
    return f"coverage {cov}/120, fast verdict {fast}/120; the persistent plant reads persistent"


@lock("TO5", "planted two-member block on a four-member frame: the residual-Gram top eigenvalue clears the independence reference and the planted pair's partial survives in >= 97/100 seeds; every null pair sits below its uniform band in >= 90/100 (Cor 7.7)")
def to5():
    top_ok = surv = nulls = 0
    for s in range(100):
        rng = np.random.default_rng(3000 + s)
        n = 1500
        clk = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
        f = rng.normal(size=n)
        E = [0.6 * f + rng.normal(size=n), 0.6 * f + rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)]
        fr = Shadow.join([_mk_sh(0.3 * E[i] / np.std(E[i]), clk) for i in range(4)])
        ch = characterize(fr, level=0.05, lags=(1, 2))
        g = ch.gram
        pl = next(p for p in g["pairs"] if p["pair"] == "0-1")
        top_ok += int(g["top_eig"] > g["reference"])
        surv += int(pl["survives"])
        nulls += int(all(not p["survives"] for p in g["pairs"] if p["pair"] != "0-1"))
    assert top_ok >= 97 and surv >= 97 and nulls >= 90, f"top {top_ok}, planted {surv}, nulls {nulls} (/100)"
    return f"top eigenvalue {top_ok}/100 above reference, planted partial {surv}/100, null pairs clean {nulls}/100"


@lock("TO6", "leak guard refuses the target's own outcome and a declared lineage descendant, both by name")
def to6():
    rep, X, y, clk = _inq_world()
    sh = _inq_slice(rep, X, y, clk)
    with pytest.raises(ValueError, match="affinely identical"):
        characterize(sh, atoms={"archive": {"own_y": {"values": 2.0 * sh.rows["y"].to_numpy() + 3.0, "lag_class": "post-decision"}}})
    with pytest.raises(ValueError, match="descends from the target"):
        characterize(sh, atoms={"archive": {"child": {"values": np.zeros(sh.m), "lag_class": "post-decision", "lineage": ["target"]}}})
    return "own outcome and lineage descendant refused by name"


@lock("TO7", "the filed charter has twelve rows; rows 8-12 are walled statuses carrying no value, with the row-8 and row-11 pointers; row 3 is not applicable on a single-unit shadow and answered on a frame; a frame files one charter per member")
def to7():
    rep, X, y, clk = _inq_world()
    ch = characterize(_inq_slice(rep, X, y, clk))
    assert len(ch.charter) == 12
    for r in ch.charter[7:12]:
        assert r["value"] is None and "walled" in r["status"]
    assert "remainder" in ch.charter[7]["status"] and "Screening" in ch.charter[10]["status"]
    assert ch.charter[2]["status"].startswith("not applicable")
    rng = np.random.default_rng(9)
    clkh = pd.date_range("2026-01-01", periods=900, freq="h", tz="UTC")
    fr = Shadow.join([_mk_sh(0.3 * rng.normal(size=900), clkh) for _ in range(2)])
    chf = characterize(fr, lags=(1, 2))
    assert chf.member_charters is not None and len(chf.member_charters) == 2
    assert all(len(c) == 12 for c in chf.member_charters)
    assert all(c[2]["status"].startswith("answerable (multi-unit frame)") for c in chf.member_charters)
    return "12 rows; five walled rows value-free with their pointers; frame files one charter per member"


@lock("TP1", "E[lam_hat^2] = rho^2 within Monte Carlo tolerance on the Gaussian host at three rho values and two J values (Thm 4.6(ii))")
def tp1():
    m, reps = 600, 1500
    for J in (1, 2):
        for rho in (0.1, 0.3, 0.6):
            rng = np.random.default_rng(11)
            v = []
            for _ in range(reps):
                Z, w = _host(rng, m, rho, J)
                l2, _, _, _ = _lambda_hat_stat(Z, _orth_slice(w), m, J)
                v.append(l2)
            v = np.asarray(v)
            tol = 4 * v.std() / np.sqrt(reps) + 2.0 / m
            assert abs(v.mean() - rho * rho) <= tol, f"J={J} rho={rho}: {v.mean():.5f} vs {rho*rho:.5f}"
    return "six cells unbiased within tolerance"


@lock("TP2", "R ~ chi2_J at rho = 0, exactly and independently of the scale (Cor 4.7): KS on the host")
def tp2():
    for J in (1, 2):
        rng = np.random.default_rng(5)
        Rs = []
        for _ in range(400):
            Z, w = _host(rng, 4000, 0.0, J)
            _, _, R, _ = _lambda_hat_stat(Z, _orth_slice(w), 4000, J)
            Rs.append(R)
        p = _kstest(Rs, _chi2(df=J).cdf).pvalue
        assert p > 0.005, f"J={J} KS p={p:.4f}"
    return "null statistic indistinguishable from chi2_J at both J"


@lock("TP3", "at m = m_req the threshold test at t* meets both budgets at the boundary alternatives within 0.03 of 1 - eps (Def 4.4, Thm 4.9(a); floor 0.30, dlam 0.10, eps 0.10)")
def tp3():
    for J in (1, 2):
        mr, _ = _inq_mreq(0.30, 0.10, J, 0.10, 0.10)
        m = int(np.ceil(mr))
        tstar = 0.30 ** 2 + 0.10 ** 2
        rng = np.random.default_rng(21)
        hit = [0, 0]
        for _ in range(1200):
            for i, rho in enumerate((0.40, 0.20)):
                Z, w = _host(rng, m, rho, J)
                l2, _, _, _ = _lambda_hat_stat(Z, _orth_slice(w), m, J)
                hit[i] += (l2 >= tstar) if i == 0 else (l2 < tstar)
        assert hit[0] / 1200 >= 0.87 and hit[1] / 1200 >= 0.87, f"J={J}: {hit[0]/1200:.3f}/{hit[1]/1200:.3f}"
    return "miss and clear sides land within budget at m_req for both J"


@lock("TP4", "the band verdict at rho = floor passes with probability at most level + 0.03 (one-sided validity; Lem 4.8, Thm 4.11(i))")
def tp4():
    for J in (1, 2):
        mr, _ = _inq_mreq(0.30, 0.10, J, 0.10, 0.10)
        m = int(np.ceil(mr))
        zl = norm.ppf(0.95)
        rng = np.random.default_rng(31)
        c = 0
        for _ in range(1200):
            Z, w = _host(rng, m, 0.30, J)
            _, lam, _, _ = _lambda_hat_stat(Z, _orth_slice(w), m, J)
            c += float(_xi_inv(_xi(lam) - zl / np.sqrt(m))) >= 0.30
        assert c / 1200 <= 0.08, f"J={J}: {c/1200:.3f}"
    return "false PASS at the floor held at the declared level for both J"


@lock("TP5", "a sign flip between the ranking slice and the confirming slice yields REJECT at J = 1")
def tp5():
    rng = np.random.default_rng(6)
    n = 1440
    clk = pd.date_range("2026-05-01", periods=n, freq="h", tz="UTC")
    e = 0.3 * rng.normal(size=n)
    sh = _mk_sh(e, clk, y=(e > 0).astype(float), p=np.full(n, 0.5))
    f = pd.Series(clk).dt.floor("D")
    par = (((f - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1D")) % 2).to_numpy()
    Z = np.where(par == 0, e, -e) + 0.1 * rng.normal(size=n)
    sc = _inq_screen(sh, {"flip": {"values": Z, "lag_class": "decision-time"}}, floor=(0.10, "lock"))
    assert sc.table.iloc[0]["verdict"] == "REJECT (sign flip)"
    return "flipped-orientation candidate rejected by the sign gate"


@lock("TP6", "m_req reproduces both limits of Theorem 4.9(b): the signal asymptote within 1%, and the corner row z sqrt(2J) / (2 lam dlam) exactly on the clear side, the side clause (e) hands to exact null testing")
def tp6():
    from provable.inquiry import n_z as _nz
    z = norm.ppf(0.90)
    for J in (1, 2):
        lam, dl = 0.30, 1e-5
        mr, _ = _inq_mreq(lam, dl, J, 0.10, 0.10)
        sig = z * z * (2 - lam * lam) / (2 * dl * dl)
        assert abs(mr / sig - 1) < 0.01
        lam = dl = 1e-4
        Delta = 2 * lam * dl
        cor = z * np.sqrt(2 * J) / (2 * lam * dl)
        assert abs(_nz(0.0, z, Delta, J) / cor - 1) < 1e-9
        mr, corner = _inq_mreq(lam, dl, J, 0.10, 0.10)
        assert corner and mr >= cor
    return "signal asymptote to 1%; corner row exact on the clear side, corner clause flagged, both J"


@lock("TP7", "a candidate without a decision-time attestation is refused by name (Ass 4.5)")
def tp7():
    rep, X, y, clk = _inq_world()
    sh = _inq_slice(rep, X, y, clk)
    with pytest.raises(ValueError, match="attestation"):
        _inq_screen(sh, {"bare": {"values": np.zeros(sh.m)}}, floor=(0.10, "lock"))
    return "unattested candidate refused before any statistic"


@lock("TP8", "the floor is mandatory and single: no floor and no order refuses by name; a declared floor next to a priced columns leg refuses by name")
def tp8():
    rep, X, y, clk = _inq_world()
    sh = _inq_slice(rep, X, y, clk)
    good = {"z": {"values": np.zeros(sh.m), "lag_class": "decision-time"}}
    with pytest.raises(ValueError, match="mandatory"):
        _inq_screen(sh, good)
    po = pv.purchase_order(rep, promise=pv.PurchasePromise(M0=0.01, rho=0.10, delta=0.10))
    with pytest.raises(ValueError, match="one floor rules"):
        _inq_screen(sh, good, floor=(0.10, "lock"), po=po)
    return "missing floor and doubled floor both refuse by name"



# ======================================================================
# TQ / TR: the loop layer (alternate, pursue)
# ======================================================================
from provable.loop import alternate as _alt, pursue as _pur, _angles as _loop_angles, _basis as _loop_basis


def _loop_world(seed=11, n=3000):
    rng = np.random.default_rng(seed)
    clk = pd.date_range("2026-03-01", periods=n, freq="h", tz="UTC")
    t = np.arange(n)
    h = t % 24
    A = np.column_stack([np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * h / 24), rng.normal(size=n)])
    B = np.column_stack([np.cos(2 * np.pi * t / 168), np.sin(2 * np.pi * t / 168), 0.6 * np.cos(2 * np.pi * h / 24) + 0.8 * rng.normal(size=n)])
    z1 = rng.normal(size=n)
    z2 = 0.5 * z1 + np.sqrt(0.75) * rng.normal(size=n)
    noise = rng.normal(size=n)
    e = 0.3 * A[:, 0] + 0.2 * B[:, 0] + 0.25 * z1 + 0.2 * z2 + 0.6 * rng.normal(size=n)
    return _mk_sh(e, clk), A, B, z1, z2, noise


def _cls(v, **kw):
    d = {"values": v, "lag_class": "decision-time"}
    d.update(kw)
    return d


@lock("TQ1", "alternation converges to the joint projection: fitted sum equals the least-squares fit on the union span to 1e-8 relative, excess to below tol (von Neumann)")
def tq1():
    sh, A, B, *_ = _loop_world()
    alt = _alt(sh, {"diurnal": _cls(A), "weekly": _cls(B)}, tol=1e-10)
    e = sh.rows["e"].to_numpy() - sh.rows["e"].mean()
    Wj = _loop_basis(np.column_stack([A - A.mean(axis=0), B - B.mean(axis=0)]))
    joint = float(np.mean((Wj @ (Wj.T @ e)) ** 2)) / sh.var
    assert alt.converged and alt.ledger["excess_out"].iloc[-1] <= 1e-10 * alt.excess0
    assert abs(alt.shares["joint"] - joint) <= 1e-8 * max(joint, 1e-12)
    assert alt.dims == (3, 3, 6) and alt.d0 == 0
    return f"joint share {alt.shares['joint']:.4f} = direct {joint:.4f}; {alt.rounds} rounds"


@lock("TQ2", "Friedrichs angle exact on two lines at a planted angle and the half-step ratio equals cos^2 exactly from the second half-step (Kayalar and Weinert 1988, sharp)")
def tq2():
    n = 400
    theta = np.deg2rad(50.0)
    clk = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(5)
    u = np.zeros(n); u[0] = 1.0
    v = np.zeros(n); v[0] = np.cos(theta); v[1] = np.sin(theta)
    e = 0.7 * u + 0.4 * v + 0.05 * rng.normal(size=n)
    e = e - e.mean()
    sh = _mk_sh(e, clk)
    # the intercept removal touches u and v; plant the lines orthogonal to the constant vector instead
    u = np.zeros(n); u[0], u[1] = 1.0, -1.0
    v = np.zeros(n); v[0], v[1], v[2], v[3] = 1.0, -1.0, 0.0, 0.0
    w = np.zeros(n); w[2], w[3] = 1.0, -1.0
    v = np.cos(theta) * u / np.linalg.norm(u) + np.sin(theta) * w / np.linalg.norm(w)
    alt = _alt(sh, {"L1": _cls(u), "L2": _cls(v)}, tol=1e-12, max_rounds=60)
    assert abs(alt.c - np.cos(theta)) <= 1e-9, alt.c
    rat = alt.ledger["ratio"].to_numpy()[1:]
    ok = rat[np.isfinite(rat)]
    assert np.all(np.abs(ok[:6] - np.cos(theta) ** 2) <= 1e-6), ok[:6]
    return f"c = {alt.c:.6f} = cos 50deg; ratios {ok[0]:.6f}, {ok[1]:.6f} = c^2 {np.cos(theta) ** 2:.6f}"


@lock("TQ3", "a shared span is reported, excluded from the angle, does not close the tunnel, and leaves only the sum of the two fits identified (concurvity)")
def tq3():
    sh, A, B, *_ = _loop_world()
    shared = np.column_stack([A, B[:, 0]])
    other = np.column_stack([B, A[:, 2]])
    alt = _alt(sh, {"P": _cls(shared), "Q": _cls(other)}, tol=1e-9)
    alt2 = _alt(sh, {"Q": _cls(other), "P": _cls(shared)}, tol=1e-9)
    assert alt.d0 == 2 and alt.converged and alt.c < 1.0
    assert abs(alt.shares["joint"] - alt2.shares["joint"]) <= 1e-9
    assert "concurvity" in alt.render()
    return f"shared span dim {alt.d0}; c = {alt.c:.4f}; joint share order-invariant {alt.shares['joint']:.4f}"


@lock("TQ4", "contraction bound holds at every half-step after the first on random classes and the ledger's excess drop equals the explained increment exactly")
def tq4():
    rng = np.random.default_rng(3)
    n = 1200
    clk = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    for s in range(4):
        A = rng.normal(size=(n, 4)); B = rng.normal(size=(n, 3)) + 0.5 * A[:, :3]
        e = A @ rng.normal(size=4) + B @ rng.normal(size=3) + rng.normal(size=n)
        sh = _mk_sh(e - e.mean(), clk)
        alt = _alt(sh, {"A": _cls(A), "B": _cls(B)}, tol=1e-9)
        led = alt.ledger
        r = led["ratio"].to_numpy()[1:]
        assert np.all(r <= alt.c ** 2 + 1e-9)
        d_excess = led["excess_in"].to_numpy() - led["excess_out"].to_numpy()
        d_expl = np.diff(np.concatenate([[0.0], led["explained_share"].to_numpy() * sh.var]))
        assert np.allclose(d_excess, d_expl, atol=1e-10)
    return "bound and ledger identity hold on four random pairs"


@lock("TQ5", "alternate refuses by name: not two classes, a frame, a class without attestation, a class column equal to the outcome (leak guard)")
def tq5():
    sh, A, B, *_ = _loop_world()
    with pytest.raises(ValueError, match="two declared classes"):
        _alt(sh, {"A": _cls(A)})
    with pytest.raises(ValueError, match="attestation"):
        _alt(sh, {"A": _cls(A), "B": {"values": B}})
    y = (sh.rows["e"].to_numpy() > 0).astype(float)
    shy = _mk_sh(sh.rows["e"].to_numpy(), sh.rows["clock"], y=y)
    with pytest.raises(ValueError, match="leak guard"):
        _alt(shy, {"A": _cls(A), "B": _cls(np.column_stack([B, 2.0 * y + 1.0]))})
    fr = Shadow.join([sh, _mk_sh(sh.rows["e"].to_numpy() * 0.5, sh.rows["clock"])])
    with pytest.raises(ValueError, match="one member at a time"):
        _alt(fr, {"A": _cls(A[:fr.m]), "B": _cls(B[:fr.m])})
    return "four refusals by name"


@lock("TQ6", "EXIT chart output: one point per half-step, the bound line and the diagonal present, the HTML page self-contained")
def tq6():
    sh, A, B, *_ = _loop_world()
    alt = _alt(sh, {"diurnal": _cls(A), "weekly": _cls(B)})
    svg = alt.exit_svg()
    pts = alt.exit_points()
    assert len(pts) == len(alt.ledger) and svg.count("<circle") == int((pts["cls"] == "diurnal").sum()) + 1
    assert "stroke-dasharray" in svg and "diagonal" in svg and svg.startswith("<svg")
    d = tempfile.mkdtemp(prefix="exit_")
    path = alt.exit_html(os.path.join(d, "exit.html"))
    txt = open(path, encoding="utf-8").read()
    assert "<svg" in txt and "http" not in txt.replace("http://www.w3.org/2000/svg", "")
    return f"{len(pts)} half-steps charted; page {len(txt)} chars, no external references"


@lock("TR1", "increments telescope and the full-slice total is order-invariant: two candidates in both orders, increments differ, totals equal the joint projection drop to 1e-10")
def tr1():
    sh, A, B, z1, z2, noise = _loop_world()
    c1 = {"z1": _cls(z1, price=0.0), "z2": _cls(z2, price=0.0)}
    c2 = {"z2": _cls(z2, price=0.0), "z1": _cls(z1, price=0.0)}
    p1 = _pur(sh, c1, requirement=(0.9, "lock"))
    p2 = _pur(sh, c2, requirement=(0.9, "lock"))
    e = sh.rows["e"].to_numpy() - sh.rows["e"].mean()
    Wj = _loop_basis(np.column_stack([z1 - z1.mean(), z2 - z2.mean()]))
    joint = float(np.mean((Wj @ (Wj.T @ e)) ** 2)) / sh.var
    f1 = p1.staircase["delta_full"].dropna().to_numpy(); f2 = p2.staircase["delta_full"].dropna().to_numpy()
    assert len(f1) == 2 and len(f2) == 2
    assert abs(p1.total_drop_full - joint) <= 1e-10 and abs(p2.total_drop_full - joint) <= 1e-10
    assert abs(f1[0] - f2[1]) > 1e-3 or abs(f1[1] - f2[0]) > 1e-3
    return f"totals {p1.total_drop_full:.6f} = {p2.total_drop_full:.6f} = joint {joint:.6f}; increments {f1.round(4)} vs {f2.round(4)}"


@lock("TR2", "trichotomy stops by name: build when findings cover the requirement, impossible above the residual mass, stall when no candidate clears its price, met when covered")
def tr2():
    sh, A, B, z1, z2, noise = _loop_world()

    class F:
        lam_arch = {"value": 0.5}
    p_build = _pur(sh, {"z1": _cls(z1, price=0.0)}, requirement=(0.2, "lock"), findings=F())
    assert p_build.staircase["verdict"].iloc[0] == "build" and "build" in p_build.outcome
    p_imp = _pur(sh, {"z1": _cls(z1, price=0.0)}, requirement=(1.5, "lock"))
    assert p_imp.staircase["verdict"].iloc[0] == "impossible"
    p_stall = _pur(sh, {"z1": _cls(z1, price=0.9), "z2": _cls(z2, price=0.9)}, requirement=(0.2, "lock"))
    assert p_stall.staircase["verdict"].iloc[0] == "stall" and p_stall.total_price == 0.0
    p_met = _pur(sh, {"z1": _cls(z1, price=0.01), "z2": _cls(z2, price=0.01)}, requirement=(0.1, "lock"))
    assert p_met.staircase["verdict"].iloc[-1] == "met" and p_met.total_drop >= 0.1
    return f"build / impossible / stall / met all reached; met after {len(p_met.staircase) - 1} acquisition(s)"


@lock("TR3", "honesty: a pure-noise free candidate is not acquired, the planted candidates are, and the confirmed drop of the first acquisition lies within its band of the full-slice drop")
def tr3():
    sh, A, B, z1, z2, noise = _loop_world()
    p = _pur(sh, {"z1": _cls(z1, price=0.02), "z2": _cls(z2, price=0.01), "noise": _cls(noise, price=0.0)}, requirement=(0.9, "lock"))
    acq = p.staircase["candidate"].tolist()
    assert "noise" not in acq and "z1" in acq and "z2" in acq
    r0 = p.staircase.iloc[0]
    assert r0["delta_lo"] <= r0["delta_full"] + 0.03
    assert p.staircase["verdict"].iloc[-1] in ("stall", "exhausted")
    return f"acquired {acq[:2]}; noise refused at its lower band end; first drop {r0['delta']:.4f} (full {r0['delta_full']:.4f})"


@lock("TR4", "the extrinsic discipline: a copy of an acquired column has no content beyond the archive and is skipped by name; refusals: missing price, missing attestation, frame, non-positive requirement")
def tr4():
    sh, A, B, z1, z2, noise = _loop_world()
    p = _pur(sh, {"z1": _cls(z1, price=0.0), "copy": _cls(3.0 * z1 - 1.0, price=0.0)}, requirement=(0.9, "lock"))
    st = p.candidates
    assert (st[(st["candidate"] == "copy") & (st["round"] == 1)]["status"].str.contains("no content beyond")).all()
    with pytest.raises(ValueError, match="no declared price"):
        _pur(sh, {"z1": _cls(z1)}, requirement=(0.1, "lock"))
    with pytest.raises(ValueError, match="attestation"):
        _pur(sh, {"z1": {"values": z1, "price": 0.0}}, requirement=(0.1, "lock"))
    with pytest.raises(ValueError, match="positive share"):
        _pur(sh, {"z1": _cls(z1, price=0.0)}, requirement=(0.0, "lock"))
    fr = Shadow.join([sh, _mk_sh(sh.rows["e"].to_numpy() * 0.5, sh.rows["clock"])])
    with pytest.raises(ValueError, match="one member at a time"):
        _pur(fr, {"z1": _cls(z1[:fr.m], price=0.0)}, requirement=(0.1, "lock"))
    return "copy skipped after its original; four refusals by name"



# ======================================================================
# TS: the loop on owned columns (select)
# ======================================================================
from provable.loop import select as _sel
from sklearn.linear_model import LogisticRegression as _LR
from sklearn.metrics import log_loss as _ll


def _sel_world(seed=3, n=5000, N=12, kind="A"):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, N)), columns=[f"x{i}" for i in range(N)])
    if kind == "A":
        eta = 1.2 * (X["x0"] ** 2 - 1) + 1.0 * X["x1"] - 0.8 * X["x2"] + 1.2 * (X["x3"] > 1.0)
    else:
        eta = 0.0 * X["x0"]
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    return X, y, pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")


def _stepwise_bic(X, y):
    n = len(y); pen = np.log(n); cols, rem = [], list(X.columns)
    def dev(c):
        if not c:
            return 2 * n * _ll(y, np.full(n, y.mean()))
        return 2 * n * _ll(y, _LR(C=1e3, max_iter=5000).fit(X[c], y).predict_proba(X[c])[:, 1])
    cur = dev(cols)
    while rem:
        b = min(rem, key=lambda c: dev(cols + [c])); d = dev(cols + [b])
        if cur - d > pen:
            cols.append(b); rem.remove(b); cur = d
        else:
            break
    return cols, (_LR(C=1e3, max_iter=5000).fit(X[cols], y) if cols else None)


@lock("TS1", "the loop admits the quadratic column by its model-free value with a zero class gain, absorbs it by enrichment, admits no noise column, and beats forward stepwise BIC on held-out log-loss")
def ts1():
    X, y, clk = _sel_world()
    fit, test = slice(0, 3500), slice(3500, 5000)
    sel = _sel(X.iloc[fit], y[fit], clk[fit], price="bic")
    r0 = sel.rounds.iloc[0]
    assert r0["verdict"] == "acquire" and r0["admitted"] == "x0" and r0["value_free_lo"] > 0 and abs(r0["gain_class"]) < 0.01 and sel.level_enrich >= 1
    assert set(sel.admitted) <= {"x0", "x1", "x2", "x3"} and {"x0", "x1", "x2"} <= set(sel.admitted)
    cols, m = _stepwise_bic(X.iloc[fit], y[fit])
    p_step = m.predict_proba(X.iloc[test][cols])[:, 1] if cols else np.full(1500, y[fit].mean())
    ll_loop, ll_step = _ll(y[test], sel.predict_proba(X.iloc[test])), _ll(y[test], p_step)
    assert ll_loop < ll_step - 0.05 and "x0" not in cols
    return f"loop {sel.admitted} enrich {sel.level_enrich}, test log-loss {ll_loop:.4f} vs stepwise-BIC {cols} {ll_step:.4f}"


@lock("TS2", "loop accounting: at every acquire round, class reduction = model-free value minus the change in the class shadow, to 1e-9; the final fit reproduces the confirm log-loss")
def ts2():
    X, y, clk = _sel_world(seed=5, n=3000)
    sel = _sel(X, y, clk, price="none", max_rounds=3)
    acq = sel.rounds[sel.rounds["verdict"].str.startswith("acquire")]
    assert len(acq) >= 2
    lhs = acq["loss_before"] - acq["loss_after"]
    rhs = acq["value_free"] - (acq["shadow_after"] - acq["shadow_before"])
    assert np.allclose(lhs, rhs, atol=1e-9)
    day = ((pd.to_datetime(pd.Series(clk)).dt.floor("D") - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1D")).to_numpy()
    cm = day % 2 == 1
    assert abs(_ll(y[cm], sel.predict_proba(X[cm])) - sel.loss_final) <= 1e-9
    return f"{len(acq)} acquire rounds, identity to 1e-9; final confirm log-loss {sel.loss_final:.5f} reproduced by predict_proba"


@lock("TS3", "select refuses by name: non-binary target, non-numeric column, leak guard, undeclared price rule, undeclared split")
def ts3():
    X, y, clk = _sel_world(seed=1, n=800)
    with pytest.raises(ValueError, match="binary"):
        _sel(X, y + 0.5, clk)
    Xs = X.copy(); Xs["s"] = "a"
    with pytest.raises(ValueError, match="non-numeric"):
        _sel(Xs, y, clk)
    Xl = X.copy(); Xl["leak"] = 2.0 * y + 1.0
    with pytest.raises(ValueError, match="leak guard"):
        _sel(Xl, y, clk)
    with pytest.raises(ValueError, match="price rule"):
        _sel(X, y, clk, price="cv")
    with pytest.raises(ValueError, match="split"):
        _sel(X, y, clk, split="random")
    return "five refusals by name"


@lock("TS4", "pure noise: nothing admitted, stall at round 0, every candidate reads 'no exhibited value' under the union-bound band")
def ts4():
    X, y, clk = _sel_world(seed=7, n=3000, kind="noise")
    sel = _sel(X, y, clk, price="none")
    assert sel.admitted == [] and sel.rounds.iloc[0]["verdict"] == "stall"
    assert (sel.candidates["status"] == "no exhibited value beyond the admitted set").all()
    return f"{len(sel.candidates)} noise candidates, none real, stall at round 0"


@lock("TS5", "prices: aic and bic charge k/n and k log n/(2n) per parameter at the current enrichment level; a declared price above a column's value keeps it out")
def ts5():
    X, y, clk = _sel_world(seed=2, n=3000, N=6)
    n = 3000
    sa = _sel(X, y, clk, price="aic", max_rounds=1)
    sb = _sel(X, y, clk, price="bic", max_rounds=1)
    ca, cb = sa.candidates[sa.candidates["round"] == 0], sb.candidates[sb.candidates["round"] == 0]
    assert np.allclose(ca["price"], 1.0 / n) and np.allclose(cb["price"], np.log(n) / (2 * n))
    sd = _sel(X, y, clk, price={"x1": 10.0, "x2": 10.0, "x0": 10.0, "x3": 10.0, "x4": 0.0, "x5": 0.0}, max_rounds=2)
    assert not ({"x0", "x1", "x2", "x3"} & set(sd.admitted))
    return f"aic price {1.0 / n:.5f}, bic price {np.log(n) / (2 * n):.5f} per parameter; overpriced true columns kept out ({sd.outcome.split(':')[0]})"


@lock("TS6", "the enriched class is held to the events-per-parameter budget: on a wide admitted set the absorb step stops at the last level the budget carries, names the stop and the shadow left as unabsorbed by budget; with the budget relaxed the same world enriches further")
def ts6():
    rng = np.random.default_rng(5)
    n, p = 2400, 12
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"w{i}" for i in range(p)])
    eta = 2.5 * X["w0"] * X["w1"] + 2.5 * X["w2"] * X["w3"] + 0.5 * X["w4"]   # pairwise products: squares cannot absorb it, level 2 can
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    clk = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    start = list(X.columns[:p])
    sel = _sel(X, y, clk, candidates=[], start=start, price="none", max_rounds=0)
    b = sel.budget
    assert b["params_allowed"] == pytest.approx(b["events"] / 10.0) and 24 <= b["params_allowed"] < 90
    assert sel.level_enrich == 1 and b["stopped_at_level"] == 1 and b["shadow_lower_end"] > 0 and b["params_used"] == 24
    assert any("stopped at level 1 by the parameter budget" in c and "unabsorbed by budget" in c for c in sel.caveats)
    assert "enrichment stopped by the budget at level 1" in sel.render()
    sh1 = float(sel.rounds["shadow_before"].iloc[0])
    sel2 = _sel(X, y, clk, candidates=[], start=start, price="none", max_rounds=0, epv=1.0)
    sh2 = float(sel2.rounds["shadow_before"].iloc[0])
    assert sel2.level_enrich == 2 and sel2.budget["stopped_at_level"] is None and sel2.budget["params_used"] == 90 and sh2 < 0 < sh1
    with pytest.raises(ValueError, match="epv"):
        _sel(X, y, clk, candidates=[], start=start, max_rounds=0, epv=0.0)
    return f"budget {b['params_allowed']:.0f} params ({b['events']} events at 10 per parameter): level 2 needs 90, stopped at level 1, shadow {sh1:+.4f} (lower end {b['shadow_lower_end']:+.4f}); at epv 1 the class reaches level 2 and the shadow closes ({sh2:+.4f})"


@lock("TS7", "union-bound fallback: a decoy that carries signal only on the screen days ranks first and fails its own band on the confirm slice; the weaker real candidate is admitted at the union-bound level instead of a stall, and the round after it stalls")
def ts7():
    rng = np.random.default_rng(3); n = 6000
    clk = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    day = ((clk.floor("D") - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1D")).to_numpy()
    even = day % 2 == 0
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["x0", "x1", "x2", "x3"])
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.0 * X["x0"] + 1.4 * z)))).astype(float)
    X["a"] = np.where(even, z, rng.normal(size=n))
    X["b"] = 0.6 * z + 0.8 * rng.normal(size=n)
    sel = _sel(X, y, clk, candidates=["a", "b"], start=["x0", "x1", "x2", "x3"], price="none", max_rounds=2)
    c0 = sel.candidates[sel.candidates["round"] == 0].set_index("candidate")
    assert c0.loc["a", "rank_stat"] > c0.loc["b", "rank_stat"] and c0.loc["a", "value_free_lo"] < 0 < c0.loc["b", "value_free_lo"]
    r0 = sel.rounds.iloc[0]
    assert r0["verdict"].startswith("acquire (union-bound fallback") and r0["admitted"] == "b"
    assert sel.admitted[-1] == "b" and "a" not in sel.admitted and sel.rounds.iloc[1]["verdict"] == "stall"
    assert "no candidate is real at the union-bound level" in sel.outcome
    return f"decoy a ranked {c0.loc['a', 'rank_stat']:+.3f} vs b {c0.loc['b', 'rank_stat']:+.3f}; a confirm lower end {c0.loc['a', 'value_free_lo']:+.3f}, b {c0.loc['b', 'value_free_lo']:+.3f}; b admitted by fallback, then stall"


# ======================================================================
# TU: completeness, every verb that reads a column
# ======================================================================
@lock("TU1", "missing values refused by name, with the count, before any statistic in every verb that reads a column: screen, characterize, alternate, pursue, select")
def tu1():
    sh, A, B, z1, z2, noise = _loop_world()
    zn = z1.copy(); zn[::97] = np.nan
    An = A.copy(); An[5, 1] = np.inf
    with pytest.raises(ValueError, match="screen refused: candidate 'zn' carries 31 missing"):
        _inq_screen(sh, {"zn": _cls(zn)}, floor=(0.10, "lock"))
    with pytest.raises(ValueError, match="characterize refused: atom 'zn' carries 31 missing"):
        characterize(sh, atoms={"archive": {"zn": {"values": zn, "lag_class": "post-decision"}}})
    with pytest.raises(ValueError, match="alternate refused: 'A' carries 1 missing"):
        _alt(sh, {"A": _cls(An), "B": _cls(B)})
    with pytest.raises(ValueError, match="pursue refused: 'zn' carries 31 missing"):
        _pur(sh, {"zn": _cls(zn, price=0.0)}, requirement=(0.1, "lock"))
    X, y, clk = _sel_world(seed=1, n=800)
    Xn = X.copy(); Xn.iloc[3, 0] = np.nan
    with pytest.raises(ValueError, match="select refused: column '.*' carries 1 missing"):
        _sel(Xn, y, clk)
    Xl = X.copy(); Xl["leak"] = 2.0 * y + 1.0; Xl.iloc[3, Xl.columns.get_loc("leak")] = np.nan
    with pytest.raises(ValueError, match="select refused: column 'leak' carries 1 missing"):
        _sel(Xl, y, clk)
    return "five verbs refuse a gapped column by name with its count; a gapped leak column is refused for the gap, not skipped by the guard"


# ======================================================================
# TV: the transcript kit
# ======================================================================
from provable.certificates import transcript as tr


def _stopped_mass(code, stop, depth):
    total, stack = 0.0, [((), 0.0)]
    while stack:
        h, L = stack.pop()
        if stop(h):
            total += 2.0 ** (-L)
            continue
        if len(h) >= depth:
            continue
        t, r = len(h), sum(h)
        for a in (True, False):
            stack.append((h + (a,), L + tr.answer_bits(code, t, r, a)))
    return total


@lock("TV1", "KT codelength: the closed form equals the sequential sum in every order (all 8191 sequences to k = 12), each complete level sums to one, and the table to k = 1000 is reproduced")
def tv1():
    import itertools
    worst, n_seq = 0.0, 0
    for k in range(13):
        total = 0.0
        for seq in itertools.product((True, False), repeat=k):
            L = tr.sequence_bits(seq)
            worst = max(worst, abs(L - tr.kt_bits(k, sum(seq))))
            total += 2.0 ** (-L)
            n_seq += 1
        assert abs(total - 1.0) < 1e-12
    assert worst < 1e-9
    table = {(5, 0): 2.0, (20, 1): 8.3, (20, 2): 11.9, (20, 3): 14.7, (100, 3): 23.1, (100, 7): 40.3, (100, 10): 50.6, (1000, 10): 86.1}
    for (k, r), L in table.items():
        assert round(tr.kt_bits(k, r), 1) == L
    assert tr.kt_bits(0, 0) == 0.0 and tr.fixed_answer_bits(2) == 1.0 and tr.sequence_bits([True, False, True], code=tr.FIXED) == 3.0
    with pytest.raises(ValueError):
        tr.kt_bits(3, 4)
    with pytest.raises(ValueError):
        tr.answer_bits("huffman", 0, 0, True)
    return f"sequential = closed form to {worst:.1e} over {n_seq} sequences; levels sum to one; 8 table cells"


@lock("TV2", "Kraft over strategy trees: under both codes and every stopping rule tried, the leaves of a finite tree carry 2^-L summing to exactly one and the stopped leaves of a truncated tree at most one")
def tv2():
    import hashlib as _h
    def rule(p, salt):
        return lambda h: len(h) > 0 and int(_h.sha256((salt + repr(h)).encode()).hexdigest(), 16) % 1000 < p * 1000
    out = []
    for code in (tr.KT, tr.FIXED):
        for salt, p in (("a", 0.15), ("b", 0.5), ("c", 0.85)):
            full = tr.kraft_leaves(code, rule(p, salt), depth=12)
            stopped = _stopped_mass(code, rule(p, salt), 12)
            assert abs(full - 1.0) < 1e-12 and stopped <= 1.0 + 1e-12
            out.append(stopped)
    assert abs(tr.kraft_leaves(tr.KT, lambda h: False, 10) - 1.0) < 1e-12
    return f"finite trees sum to 1; stopped mass in [{min(out):.3f}, {max(out):.3f}]"


@lock("TV3", "bars and rows: z_0 = 1.6449 and z_1 = 1.9600 at delta 0.05; z_L in log space matches the direct inverse where representable and stays finite and increasing to 5000 bits; s(L) inverts; s = 0.05 sits at 38.94 bits; the exam slice wins iff its share exceeds s(L)")
def tv3():
    import math
    assert abs(tr.z_bar(0.05, 0) - 1.6448536) < 1e-6 and abs(tr.z_bar(0.05, 1) - 1.9599640) < 1e-6
    prev = 0.0
    for L in (0, 0.5, 1, 5, 10, 40, 100, 500, 1000, 1100, 2000, 5000):
        z = tr.z_bar(0.05, L)
        p = 0.05 * 2.0 ** (-L)
        if p > 1e-300:
            assert abs(z - norm.isf(p)) < 1e-9 * max(1.0, z)
        assert np.isfinite(z) and z > prev
        prev = z
    for s in (0.9, 0.5, 0.2, 0.05, 0.01):
        assert abs(tr.clean_share(0.05, tr.bits_for_share(0.05, s)) - s) < 1e-9
    Lr = tr.bits_for_share(0.05, 0.05)
    assert round(Lr, 2) == 38.94
    for L in (1.0, 8.3, 23.1, 40.3):
        sL = tr.clean_share(0.05, L)
        assert tr.exam_wins(0.05, L, sL + 1e-6) and not tr.exam_wins(0.05, L, sL - 1e-6)
    assert abs(tr.hoeffding_width(0.05, 0, 100, 1.0) - math.sqrt(math.log(20) / 200)) < 1e-12
    assert abs(tr.inflation(3.0) - 8.0) < 1e-12 and abs(tr.level(0.05, 2.0) - 0.0125) < 1e-15
    return f"z(5000 bits) = {prev:.2f}; retirement at {Lr:.2f} bits; s(L) inverse exact"


def _gnc_tree(beta, n, tau, depth):
    per, stack = {}, [(1, ())]
    while stack:
        i, gam = stack.pop()
        b = tr.gnc_level(beta, i, gam)
        per[i] = per.get(i, 0.0) + b
        if i == depth:
            continue
        stack.append((i + 1, gam))
        g = tau - tr.hoeffding_tol(b, n)
        assert g > 0
        stack.extend([(i + 1, gam + (g,))] * tr.gnc_grid_size(g))
    return per


@lock("TV4", "Guess and Check weights: c_j sums to one; beta_i = beta 2^-L_i; over an enumerated transcript tree the per-depth sum of beta_i equals beta c_(i-1) sum_(f<i) c_f, so the union over every reachable query stays under beta")
def tv4():
    import math
    assert abs(sum(tr.gnc_weight(j) for j in range(200000)) - 1.0) < 1e-5
    per = _gnc_tree(0.05, 2000, 0.2, 4)
    worst = 0.0
    for i, s_i in per.items():
        expect = 0.05 * tr.gnc_weight(i - 1) * sum(tr.gnc_weight(f) for f in range(i))
        worst = max(worst, abs(s_i / expect - 1.0))
    assert worst < 1e-9 and sum(per.values()) < 0.05
    for i, gam in ((1, ()), (3, (0.1,)), (6, (0.2, 0.05, 0.3))):
        assert abs(tr.gnc_level(0.05, i, gam) - 0.05 * 2.0 ** (-tr.gnc_bits(i, gam))) < 1e-15
        assert tr.gnc_transcripts(i, gam) == math.comb(i - 1, len(gam)) * int(np.prod([tr.gnc_grid_size(g) for g in gam]))
    assert tr.gnc_grid_size(0.25) == 5 and tr.gnc_round(0.62, 0.25) == 0.5 and tr.gnc_round(1.0, 0.3) == pytest.approx(0.9)
    return f"per-depth union = beta c_(i-1) sum c_f to {worst:.1e}; total {sum(per.values()):.4f} < 0.05"


@lock("TV5", "extreme transcripts: inflation past 1024 bits is infinite without overflow in the kit, in admission and in R7; Guess and Check counts are exact integers with bits in log space and a finite tolerance where the level underflows; the log-space Newton fallback for z_L matches the direct inverse to 5000 bits")
def tv5():
    import math
    assert tr.inflation(2000) == math.inf and tr.inflation(10) == 1024.0
    c = ct.issue(claim="far", guarantee=g(), ell_test=ct.Lineage(), proposer_rows=ct.Lineage(), artifact={}, support=("a",), gram_context={}, kind=ct.CertificateKind.CARRIER, peeks=[{"holdout": "HO-FAR", "bits": 2000.0, "rows": ct.Lineage.of(ct.slice_from_frame(frame("dataset_1", 3), "dataset_1", range(0, 8)))}])
    rows = ct.Lineage.of(ct.slice_from_frame(frame("dataset_1", 3), "dataset_1", range(0, 4)))
    adm = pv.reuse([c], rows, level=0.1)
    assert adm.verdict == ct.Verdict.PEEKED and adm.inflation == math.inf and adm.level == 0.0
    f = pv.lint(ct.RunRecord(name="far", test_rows=rows, q=0.1, consumed=[c]))
    assert f.repair("R7")["inflation"] == math.inf and f.repair("R7")["repaired_level"] == 0.0
    big = [0.001] * 300
    assert isinstance(tr.gnc_transcripts(5000, big), int) and tr.gnc_level(0.05, 5000, big) == 0.0
    b = tr.gnc_bits(5000, big)
    assert 4000 < b < 6000 and math.isfinite(tr.hoeffding_tol_bits(0.05, b, 1000))
    worst = max(abs(tr._isf_log_newton(math.log(0.05) - L * math.log(2)) / tr.z_bar(0.05, L) - 1) for L in (0, 3, 50, 500, 1077, 1100, 2000, 5000))
    assert worst < 1e-9
    return f"inflation inf past 1024 bits; Guess and Check bits {b:.0f} with a finite tolerance; fallback z_L within {worst:.1e}"


# ======================================================================
# TW: the mediated holdout
# ======================================================================
HFEATS = [f"h{i}" for i in range(5)]


def _ho_world(n=1200, seed=0, signal=True):
    rng = np.random.default_rng(seed)
    panel = pd.DataFrame(rng.normal(size=(n, 5)), columns=HFEATS)
    panel["y"] = (0.6 * panel["h0"] - 0.4 * panel["h1"] if signal else 0.0) + rng.normal(size=n)
    panel["start_utc"] = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    panel["end_utc"] = panel["start_utc"] + pd.Timedelta(hours=1)
    return panel


def _ho_split(panel, n_s):
    S, H = panel.index[:n_s], panel.index[n_s:]
    lin = lambda idx: ct.Lineage.of(ct.slice_from_frame(panel, "hdemo", rows=idx))
    return S, H, lin(S), lin(H)


def _ols(X, y, cols):
    A = np.column_stack([np.ones(len(X))] + [X[c].to_numpy() for c in cols])
    b = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)[0]
    return lambda Z: b[0] + (np.column_stack([Z[c].to_numpy() for c in cols]) @ b[1:] if cols else 0.0)


def _ho_basic(seed=1, exam_share=0.2, **kw):
    panel = _ho_world(seed=seed)
    S, H, S_lin, H_lin = _ho_split(panel, 800)
    d = tempfile.mkdtemp(prefix="ho_")
    ys = panel.loc[S, "y"]
    ho = pv.seal(panel.loc[H, HFEATS], panel.loc[H, "y"], d, lineage=H_lin, baseline=float(ys.mean()), baseline_lineage=S_lin, exam_share=exam_share, seed=4, **kw)
    return panel, S, H, S_lin, H_lin, d, ho


@lock("TW1", "seal commits before any answer: salted content hashes of rows, target and baseline recomputed from custody files that hold no pickle, row identities and split hashed, the id is the hash of the seal record, the protocol fixes route and retirement bits, a second handle reads the same holdout")
def tw1():
    from provable.holdout import _x_sha, _encode_frame, _salted, _npy_bytes, _sha, _canon
    panel, S, H, S_lin, H_lin, d, ho = _ho_basic()
    op = ho.seal_op
    assert op["seq"] == 0 and op["prev"] is None and op["action"] == "seal"
    cm, pr = ho.commit, ho.protocol
    X, y, tune, ex = ho._data()
    cu = json.load(open(os.path.join(d, "holdouts", ho.id, "custody.json")))
    files, schema = _encode_frame(panel.loc[H, HFEATS])
    assert cm["x_sha"] == _x_sha(cu["salt"], files, schema) and cm["y_sha"] == _salted(cu["salt"], _npy_bytes(np.asarray(panel.loc[H, "y"], float)))
    assert cm["baseline_sha"] == _salted(cu["salt"], _npy_bytes(np.full(400, float(panel.loc[S, "y"].mean()))))
    assert cm["y_sha"] != _sha(_npy_bytes(np.asarray(panel.loc[H, "y"], float)))
    kinds = {os.path.splitext(f)[1] for _, _, fs in os.walk(os.path.join(d, "holdouts", ho.id)) for f in fs}
    assert kinds <= {".npy", ".json"}
    for root, _, fs in os.walk(os.path.join(d, "holdouts", ho.id)):
        for f in fs:
            if f.endswith(".npy"):
                np.load(os.path.join(root, f), allow_pickle=False)
    assert X.equals(panel.loc[H, HFEATS]) and np.array_equal(y, panel.loc[H, "y"].to_numpy())
    assert cm["n"] == 400 and cm["n_tune"] == 320 and cm["n_exam"] == 80 and set(tune) | set(ex) == set(range(400)) and not (set(tune) & set(ex))
    assert pr["route"] == "exam" and pr["code"] == "kt" and pr["eta"] == 1.0 and pr["delta"] == 0.05 and abs(pr["retire_bits"] - 38.941) < 1e-3 and pr["rowwise"] is False
    assert ho.id == "HO-" + _sha(_canon({k: v for k, v in op.items() if k != "holdout"}))[:16].upper()
    ho2 = pv.Holdout(d, ho.id)
    assert ho2.commit == cm and ho2.tuning_lineage() == ho.tuning_lineage() and ho2.exam_lineage() == ho.exam_lineage()
    assert set().union(*[s_.rows for s_ in ho.tuning_lineage().union(ho.exam_lineage()).slices]) == set(H)
    d2 = tempfile.mkdtemp(prefix="ho_")
    ho3 = pv.seal(panel.loc[H, HFEATS], panel.loc[H, "y"], d2, lineage=H_lin, baseline=0.0, baseline_lineage=ct.Lineage(), seed=4)
    d3 = tempfile.mkdtemp(prefix="ho_")
    ho4 = pv.seal(panel.loc[H, HFEATS], panel.loc[H, "y"], d3, lineage=H_lin, baseline=0.0, baseline_lineage=ct.Lineage(), seed=5, exam_share=0.0)
    assert ho3.commit["split_sha"] == cm["split_sha"] and ho3.commit["x_sha"] != cm["x_sha"] and ho4.protocol["route"] == "kraft" and ho4.commit["n_exam"] == 0
    return f"{ho.id}: 320 tuning / 80 exam, route exam; salted commitments recomputed; custody holds only .npy and .json"


@lock("TW2", "refusals by name, none depending on sealed values: lineage overlap with rows named, undeclared and UNKNOWN lineage; an outcome-derived lag column checked against the sealed clock at the seal (its own outcome, no clock) and, after the seal, refused unless sealed with one message whatever its lag; a rows column meeting sealed rows or UNKNOWN; gapped column and target at seal, rows already spent or already sealed, and the wrong verb for the mode")
def tw2():
    from datetime import timedelta
    panel, S, H, S_lin, H_lin, d, ho = _ho_basic()
    X, y = panel.loc[S, HFEATS], panel.loc[S, "y"]
    f = _ols(X, y, ["h0"])
    clash = ct.Lineage.of(ct.slice_from_frame(panel, "hdemo", rows=list(S[:5]) + list(H[:3])))
    with pytest.raises(ValueError, match=r"ask refused: the model's lineage meets the sealed rows of HO-.*rows \[800, 801, 802\]"):
        ho.ask(f, lineage=clash)
    with pytest.raises(ValueError, match="ask refused: declare the rows the model was fit on"):
        ho.ask(f)
    with pytest.raises(ValueError, match="ask refused: the model's lineage is UNKNOWN"):
        ho.ask(f, lineage=unknown_lineage())
    d2 = tempfile.mkdtemp(prefix="ho_")
    clk = (panel.loc[H, "start_utc"], panel.loc[H, "end_utc"])
    hoc = pv.seal(panel.loc[H, HFEATS], panel.loc[H, "y"], d2, lineage=H_lin, baseline=0.0, baseline_lineage=ct.Lineage(), clock=clk)
    own = ct.ColumnDeclaration(column="y_recent", dataset_tag="hdemo", lag=timedelta(minutes=30), outcome_derived=True)
    lag1 = ct.ColumnDeclaration(column="y_lag1", dataset_tag="hdemo", lag=timedelta(hours=1), outcome_derived=True)
    far = ct.ColumnDeclaration(column="y_lag500", dataset_tag="hdemo", lag=timedelta(hours=500), outcome_derived=True)
    msgs = []
    for c in (own, lag1, far):
        with pytest.raises(ValueError, match=r"ask refused: \[" + c.column + r"\] an outcome-derived lag column is checked against the sealed clock row by row, so it is part of the protocol: declare it when sealing") as e:
            hoc.ask(f, lineage=S_lin, columns=[c])
        msgs.append(str(e.value).replace(c.column, "COL"))
    assert len(set(msgs)) == 1 and hoc.state["asks"] == 0
    with pytest.raises(ValueError, match=r"seal refused: declared outcome columns reach sealed outcomes: \[y_recent\] own outcome rows in column lineage: lag 0:30:00"):
        pv.seal(panel.loc[H, HFEATS], panel.loc[H, "y"], tempfile.mkdtemp(), lineage=H_lin, baseline=0.0, baseline_lineage=ct.Lineage(), clock=clk, columns=[own])
    with pytest.raises(ValueError, match="a lag declaration needs the holdout's clock"):
        pv.seal(panel.loc[H, HFEATS], panel.loc[H, "y"], tempfile.mkdtemp(), lineage=H_lin, baseline=0.0, baseline_lineage=ct.Lineage(), columns=[far])
    hfar = pv.seal(panel.loc[H, HFEATS], panel.loc[H, "y"], tempfile.mkdtemp(prefix="ho_"), lineage=H_lin, baseline=0.0, baseline_lineage=ct.Lineage(), clock=clk, columns=[far])
    assert hfar.protocol["rowwise"] is False
    hfar.ask(f, lineage=S_lin, columns=[far])
    enc = ct.ColumnDeclaration(column="target_enc", dataset_tag="hdemo", rows=H_lin, outcome_derived=True)
    with pytest.raises(ValueError, match=r"ask refused: \[target_enc\] sealed outcome rows in column lineage"):
        hoc.ask(f, lineage=S_lin, columns=[enc])
    unk = ct.ColumnDeclaration(column="archive_enc", dataset_tag="hdemo", rows=unknown_lineage(), outcome_derived=True)
    with pytest.raises(ValueError, match=r"\[archive_enc\] column lineage is UNKNOWN"):
        hoc.ask(f, lineage=S_lin, columns=[unk])
    Xg = panel.loc[H, HFEATS].copy(); Xg.iloc[7, 2] = np.nan
    with pytest.raises(ValueError, match="seal refused: column 'h2' carries 1 missing or non-finite"):
        pv.seal(Xg, panel.loc[H, "y"], tempfile.mkdtemp(), lineage=H_lin, baseline=0.0, baseline_lineage=ct.Lineage())
    yg = panel.loc[H, "y"].copy(); yg.iloc[3] = np.inf
    with pytest.raises(ValueError, match="seal refused: the target carries 1 missing or non-finite"):
        pv.seal(panel.loc[H, HFEATS], yg, tempfile.mkdtemp(), lineage=H_lin, baseline=0.0, baseline_lineage=ct.Lineage())
    with pytest.raises(ValueError, match=r"seal refused: rows already under seal in HO-"):
        pv.seal(panel.loc[H, HFEATS], panel.loc[H, "y"], d, lineage=H_lin, baseline=0.0, baseline_lineage=ct.Lineage())
    led = ct.Ledger(tempfile.mkdtemp(prefix="ho_"), budget=0.2)
    led.register(mk("spent", ct.Lineage.of(ct.slice_from_frame(panel, "hdemo", rows=H[:10])), ct.Lineage()))
    with pytest.raises(ValueError, match="seal refused: rows already spent in the book"):
        pv.seal(panel.loc[H, HFEATS], panel.loc[H, "y"], led, lineage=H_lin, baseline=0.0, baseline_lineage=ct.Lineage())
    with pytest.raises(ValueError, match="seal refused: the holdout's row identity is undeclared"):
        pv.seal(panel.loc[H, HFEATS], panel.loc[H, "y"], tempfile.mkdtemp(), baseline=0.0, baseline_lineage=ct.Lineage())
    with pytest.raises(ValueError, match="seal refused: comparative mode compares each model with an incumbent"):
        pv.seal(panel.loc[H, HFEATS], panel.loc[H, "y"], tempfile.mkdtemp(), lineage=H_lin)
    with pytest.raises(ValueError, match="seal refused: declare the rows the baseline was fit on"):
        pv.seal(panel.loc[H, HFEATS], panel.loc[H, "y"], tempfile.mkdtemp(), lineage=H_lin, baseline=0.0)
    with pytest.raises(ValueError, match="seal refused: numeric mode needs a declared loss range"):
        pv.seal(panel.loc[H, HFEATS], panel.loc[H, "y"], tempfile.mkdtemp(), lineage=H_lin, mode="numeric")
    with pytest.raises(ValueError, match="seal refused: y's index does not match X's"):
        pv.seal(panel.loc[H, HFEATS], panel.loc[H, "y"].reset_index(drop=True), tempfile.mkdtemp(), lineage=H_lin, baseline=0.0, baseline_lineage=ct.Lineage())
    with pytest.raises(ValueError, match="check refused: .* is sealed in comparative mode"):
        ho.check(f, 0.5, 0.1, lineage=S_lin)
    return "20 refusals by name; an unsealed lag column gets one message whatever its lag; a sealed lag reaching only rows before the holdout clears the gate"


@lock("TW3", "ask: accept iff the paired gain over the incumbent exceeds eta paired standard errors, recomputed independently with the incumbent following the accepts; bits per answer and their running sum match the kit; the answer carries no score")
def tw3():
    panel, S, H, S_lin, H_lin, d, ho = _ho_basic(seed=2)
    X, y = panel.loc[S, HFEATS], panel.loc[S, "y"]
    _, yh, tune, _ = ho._data()
    Xh = panel.loc[H, HFEATS]
    inc = np.full(len(H), float(y.mean()))
    answers = []
    for cols in (["h0"], ["h0", "h1"], ["h0", "h1", "h2"], ["h1"], ["h0", "h1", "h3"], ["h0", "h1", "h2", "h3", "h4"]):
        f = _ols(X, y, cols)
        a = ho.ask(f, lineage=S_lin)
        p = f(Xh)
        dd = (yh[tune] - inc[tune]) ** 2 - (yh[tune] - p[tune]) ** 2
        expect = bool(dd.mean() > 1.0 * dd.std(ddof=1) / np.sqrt(len(dd)))
        assert a.accepted == expect and a.value is None and a.width is None and a.level is None
        if expect:
            inc = p
        answers.append(expect)
        assert abs(a.total_bits - tr.sequence_bits(answers)) < 1e-12 and abs(a.bar - tr.z_bar(0.05, a.total_bits)) < 1e-12 and abs(a.clean_share - tr.clean_share(0.05, a.total_bits)) < 1e-12
    st = ho.state
    assert st["asks"] == 6 and st["accepts"] == sum(answers) and answers[:2] == [True, True]
    return f"answers {''.join('A' if a else 'R' for a in answers)}; {st['bits']:.3f} bits; z_L {st['bar']:.3f}; s(L) {st['clean_share']:.3f}"


@lock("TW4", "retirement: under the fixed code the tuning slice retires itself with the answer that takes s(L) to 0.05 (bit 39, as the kit predicts, in the same log operation); a later ask refuses with the cause; retire releases the tuning rows, then the exam rows too once opened, idempotently")
def tw4():
    import math
    panel, S, H, S_lin, H_lin, d, ho = _ho_basic(seed=3, code="fixed")
    f0 = float(panel.loc[S, "y"].mean())
    k_ret = math.ceil(tr.bits_for_share(0.05, 0.05))
    for k in range(1, k_ret + 1):
        a = ho.ask(f0, lineage=S_lin)
        assert (a.tuning == "retired") == (k == k_ret)
    with pytest.raises(ValueError, match=r"ask refused: the tuning slice of HO-.* retired at #39 \(threshold: s\(L\) 0\.0[0-9]+ <= retire_at 0\.05\)"):
        ho.ask(f0, lineage=S_lin)
    rel = ho.retire()
    assert rel.slices == ("tuning",) and len(rel.X) == 320 and rel.lineage == ho.tuning_lineage() and rel.cause.startswith("threshold")
    n_ops = len(ho._ops())
    cert = ho.open(_ols(panel.loc[S, HFEATS], panel.loc[S, "y"], ["h0"]), lineage=S_lin)
    rel2 = ho.retire()
    assert rel2.slices == ("tuning", "exam") and len(rel2.X) == 400 and len(ho._ops()) == n_ops + 1
    return f"retired at ask {k_ret} ({tr.clean_share(0.05, k_ret):.4f}); released 320 then 400 rows; exam opened after retirement"


@lock("TW5", "open: the exam slice once at the clean bar, a second open refused; without an exam slice the transcript certificate at z_L on the tuning slice, which then closes; bounds recomputed; one charge at delta; performance kind, FWER, the tuning transcript carried as a peek; the Hoeffding tail exact for a declared range")
def tw5():
    panel, S, H, S_lin, H_lin, d, ho = _ho_basic(seed=5, claim="gain")
    X, y = panel.loc[S, HFEATS], panel.loc[S, "y"]
    for cols in (["h0"], ["h0", "h1"], ["h0", "h1", "h2"]):
        ho.ask(_ols(X, y, cols), lineage=S_lin)
    L = ho.state["bits"]
    f = _ols(X, y, ["h0", "h1"])
    cert = ho.open(f, lineage=S_lin)
    _, yh, tune, ex = ho._data()
    p = f(panel.loc[H, HFEATS])
    dd = (yh[ex] - float(y.mean())) ** 2 - (yh[ex] - p[ex]) ** 2
    bound = dd.mean() - tr.z_bar(0.05, 0) * dd.std(ddof=1) / np.sqrt(len(dd))
    fam = cert.guarantee.family
    assert abs(cert.artifact["bound"] - bound) < 1e-12 and cert.kind == ct.CertificateKind.PERFORMANCE and cert.guarantee.kind == ct.GuaranteeKind.FWER and cert.guarantee.level == 0.05
    assert fam["slice"] == "exam" and fam["bar"] == tr.z_bar(0.05, 0) and abs(fam["bits"] - L) < 1e-12
    pk = ct.peeks_of(cert)
    assert len(pk) == 1 and pk[0]["holdout"] == ho.id and abs(pk[0]["bits"] - L) < 1e-12 and pk[0]["rows"] == ho.tuning_lineage()
    assert ct.Ledger(d).statement().charged == pytest.approx(0.05)
    with pytest.raises(ValueError, match=r"open refused: the exam slice of HO-.* was opened at #4 \(CD-[0-9A-F]{16}\); a slice is opened once"):
        ho.open(f, lineage=S_lin)
    panel2, S2, H2, S2_lin, H2_lin, d2, hk = _ho_basic(seed=6, exam_share=0.0)
    X2, y2 = panel2.loc[S2, HFEATS], panel2.loc[S2, "y"]
    for cols in (["h0"], ["h0", "h1"], ["h2"]):
        hk.ask(_ols(X2, y2, cols), lineage=S2_lin)
    L2 = hk.state["bits"]
    fk = _ols(X2, y2, ["h0", "h1"])
    ck = hk.open(fk, lineage=S2_lin)
    _, yk, tk, _ = hk._data()
    lk = (yk[tk] - fk(panel2.loc[H2, HFEATS])[tk]) ** 2
    bk = lk.mean() + tr.z_bar(0.05, L2) * lk.std(ddof=1) / np.sqrt(len(lk))
    assert abs(ck.artifact["bound"] - bk) < 1e-12 and ck.guarantee.family["slice"] == "tuning" and ck.guarantee.family["level_at_bar"] == pytest.approx(0.05 * 2 ** -L2)
    with pytest.raises(ValueError, match="ask refused: the tuning slice of HO-.* closed at #4, when its transcript certificate was issued"):
        hk.ask(fk, lineage=S2_lin)
    with pytest.raises(ValueError, match="open refused: the transcript certificate of HO-.* was issued at #4"):
        hk.open(fk, lineage=S2_lin)
    yb = (panel2["y"] > 0).astype(float)
    hb = pv.seal(panel2.loc[H2, HFEATS], yb.loc[H2], tempfile.mkdtemp(prefix="ho_"), lineage=H2_lin, loss="zero_one", tail="hoeffding", baseline=1.0, baseline_lineage=ct.Lineage(), exam_share=0.25)
    cb = hb.open(lambda Z: (Z["h0"] > 0).astype(float).to_numpy(), lineage=ct.Lineage())
    _, yb_h, _, eb = hb._data()
    lb = ((panel2.loc[H2, "h0"] > 0).astype(float).to_numpy()[eb] != yb_h[eb]).astype(float)
    assert abs(cb.artifact["bound"] - (lb.mean() + tr.hoeffding_width(0.05, 0, len(lb), 1.0))) < 1e-12
    return f"exam bound {bound:+.4f} at z_0; transcript bound {bk:.4f} at z_L {tr.z_bar(0.05, L2):.3f} ({L2:.2f} bits); Hoeffding exact"


def _null_holdout(rng, h, k, exam_share, seed):
    X = pd.DataFrame(rng.normal(size=(h, k)), columns=[f"c{j}" for j in range(k)])
    y = rng.normal(size=h)
    ho = pv.seal(X, y, tempfile.mkdtemp(prefix="adv_"), lineage="self", baseline=0.0, baseline_lineage=ct.Lineage(), claim="gain", eta=0.0, exam_share=exam_share, seed=seed)
    return X, y, ho


def _sign_adversary(ho, k, n_tune):
    cols = [f"c{j}" for j in range(k)]
    lin = lambda w: (lambda Z: Z[cols].to_numpy() @ w)
    inc, signs = np.zeros(k), []
    for j in range(k):
        w = inc.copy(); w[j] += 1e-4
        a = ho.ask(lin(w), lineage=ct.Lineage())
        signs.append(1.0 if a.accepted else -1.0)
        if a.accepted:
            inc = w
    c = np.sqrt(2.0 / (np.pi * n_tune))
    return lin(c * np.asarray(signs)), c


@lock("TW6", "coverage of the transcript certificate against the sign-aggregation adversary through ask: 20 sign queries on 400 null rows per holdout, the aggregated model's gain bound at z_L fails in at most delta of 200 holdouts (Monte Carlo slack), while the unpriced bar fails in nearly all of them")
def tw6():
    rng = np.random.default_rng(20261005)
    reps, k, h = 200, 20, 400
    fail_k = fail_n = 0
    bits = []
    for rep in range(reps):
        X, y, ho = _null_holdout(rng, h, k, 0.0, rep)
        f, c = _sign_adversary(ho, k, h)
        cert = ho.open(f, lineage=ct.Lineage())
        true = -c * c * k
        fail_k += int(cert.artifact["bound"] > true)
        dd = y ** 2 - (y - f(X)) ** 2
        fail_n += int(dd.mean() - tr.z_bar(0.05, 0) * dd.std(ddof=1) / np.sqrt(h) > true)
        bits.append(ho.state["bits"])
    slack = 3 * np.sqrt(0.05 * 0.95 / reps)
    assert fail_k / reps <= 0.05 + slack and fail_n / reps >= 0.8
    return f"Kraft bar fails {fail_k}/{reps}; unpriced bar fails {fail_n}/{reps}; mean {np.mean(bits):.1f} bits"


@lock("TW7", "coverage of the exam slice: the same adversary tunes on 400 tuning rows; the exam certificate at the clean bar on 100 sealed rows, under the Gaussian bar (no loss range, so a CLT approximation), fails in at most delta of 200 holdouts up to Monte Carlo slack, while the unpriced tuning-slice bound fails in nearly all of them; the Gaussian miss rate itself is measured by TW21, and with a range the betting tail is exact")
def tw7():
    rng = np.random.default_rng(20261006)
    reps, k, h = 200, 20, 500
    fail_e = fail_n = 0
    for rep in range(reps):
        X, y, ho = _null_holdout(rng, h, k, 0.2, rep)
        _, _, tune, ex = ho._data()
        f, c = _sign_adversary(ho, k, len(tune))
        cert = ho.open(f, lineage=ct.Lineage())
        true = -c * c * k
        fail_e += int(cert.artifact["bound"] > true)
        dd = y[tune] ** 2 - (y[tune] - f(X.iloc[tune])) ** 2
        fail_n += int(dd.mean() - tr.z_bar(0.05, 0) * dd.std(ddof=1) / np.sqrt(len(tune)) > true)
    slack = 3 * np.sqrt(0.05 * 0.95 / reps)
    assert fail_e / reps <= 0.05 + slack and fail_n / reps >= 0.8
    return f"exam bound fails {fail_e}/{reps}; unpriced tuning bound fails {fail_n}/{reps}"


@lock("TW8", "audit recomputes everything from the log and custody (hash chain, custody head, dates, id, salted commitments, every answer and its bits, every certificate rebuilt against the book, models re-run once every row has left custody) and FAILs on a tampered prediction file, target or clock; a rewritten log line FAILs without being replayed")
def tw8():
    import shutil
    panel, S, H, S_lin, H_lin, d, ho = _ho_basic(seed=8)
    X, y = panel.loc[S, HFEATS], panel.loc[S, "y"]
    models = {}
    for cols in (["h0"], ["h0", "h1"], ["h3"]):
        f = _ols(X, y, cols)
        a = ho.ask(f, lineage=S_lin)
        models[a.seq] = f
    fo = _ols(X, y, ["h0", "h1"])
    ho.open(fo, lineage=S_lin)
    models[4] = fo
    ho.retire()
    au = ho.audit(models=models)
    assert au.ok and all(r[1] == "PASS" for r in au.rows) and sum(1 for r in au.rows if r[0].endswith("model")) == 4
    def tampered(fn, src=d, hid=ho.id):
        d2 = tempfile.mkdtemp(prefix="tamper_")
        shutil.rmtree(d2)
        shutil.copytree(src, d2)
        fn(d2)
        return pv.Holdout(d2, hid).audit()
    def pred(d2):
        pth = os.path.join(d2, "holdouts", ho.id, "pred", "2.npy")
        v = np.load(pth); v[0] += 1e-6; np.save(pth, v)
    a1 = tampered(pred)
    assert not a1.ok and [r for r in a1.rows if r[0] == "#2 ask"][0][2] == "stored predictions do not match the logged hash"
    def target(d2):
        pth = os.path.join(d2, "holdouts", ho.id, "y.npy")
        v = np.load(pth); v[5] += 0.5; np.save(pth, v)
    a2 = tampered(target)
    assert not a2.ok and "the target changed after sealing" in [r for r in a2.rows if r[0] == "custody rows"][0][2]
    def log(d2):
        pth = os.path.join(d2, "ledger.jsonl")
        lines = open(pth).read().splitlines()
        i = [j for j, l in enumerate(lines) if '"action":"ask"' in l][0]
        o = json.loads(lines[i]); o["answer"] = "reject" if o["answer"] == "accept" else "accept"
        lines[i] = json.dumps(o, sort_keys=True, separators=(",", ":"))
        open(pth, "w").write("\n".join(lines) + "\n")
    a3 = tampered(log)
    assert not a3.ok and [r for r in a3.rows if r[0] == "hash chain"][0][1] == "FAIL" and [r for r in a3.rows if r[0] == "replay"][0][2].startswith("not replayed") and not any(r[0].startswith("#") for r in a3.rows)
    dc = tempfile.mkdtemp(prefix="ho_")
    hc = pv.seal(panel.loc[H, HFEATS], panel.loc[H, "y"], dc, lineage=H_lin, baseline=0.0, baseline_lineage=ct.Lineage(), clock=(panel.loc[H, "start_utc"], panel.loc[H, "end_utc"]))
    assert hc.commit["clock_sha"] is not None and hc.audit().ok
    def clock(d2):
        pth = os.path.join(d2, "holdouts", hc.id, "clock.npy")
        v = np.load(pth); v[0, 0] -= 3600 * 10 ** 9; np.save(pth, v)
    a4 = tampered(clock, dc, hc.id)
    assert not a4.ok and "the clock changed after sealing" in [r for r in a4.rows if r[0] == "custody rows"][0][2]
    return f"{len(au.rows)} items PASS incl. 4 models re-run; four tampers each FAIL"


@lock("TW9", "two handles on one book agree: answers through one handle are seen by the other and the chain stays whole; the statement's holdout block reports asks, bits, z_L, s(L), tuning and exam status")
def tw9():
    panel, S, H, S_lin, H_lin, d, ho = _ho_basic(seed=9)
    X, y = panel.loc[S, HFEATS], panel.loc[S, "y"]
    other = pv.Holdout(d, ho.id)
    ho.ask(_ols(X, y, ["h0"]), lineage=S_lin)
    assert other.state["asks"] == 1
    other.ask(_ols(X, y, ["h0", "h1"]), lineage=S_lin)
    assert ho.state["asks"] == 2 and ho.audit().ok
    st = ct.Ledger(d).statement()
    row = st.holdouts[0]
    assert row["id"] == ho.id and row["asks"] == 2 and row["tuning"] == "open" and row["exam"] == "sealed" and abs(row["bits"] - ho.state["bits"]) < 1e-12
    txt = st.render()
    assert "-- holdouts" in txt and ho.id in txt
    return f"two handles, one chain; statement row {row['asks']} asks, {row['bits']:.2f} bits"


@lock("TW10", "no answer channel outside the priced answers: a probe that would read a sealed outcome from an out-of-range or non-finite refusal is answered and priced instead (brier and zero-one losses are total, declared ranges clip, non-finite predictions fall back to the baseline or the top of the range), so the transcript counts every look")
def tw10():
    rng = np.random.default_rng(10)
    n = 200
    X = pd.DataFrame({"a": rng.normal(size=n)})
    yb = (rng.random(n) < 0.5).astype(float)
    ho = pv.seal(X, yb, tempfile.mkdtemp(prefix="chan_"), lineage="self", loss="brier", baseline=0.5, baseline_lineage=ct.Lineage(), exam_share=0.25, seed=1)
    Xh, yh, tune, ex = ho._data()
    j = Xh.index[tune[3]]
    def probe(Z):
        p = np.full(len(Z), 1e6)
        if j in Z.index:
            p[list(Z.index).index(j)] = 2.0
        return p
    a = ho.ask(probe, lineage=ct.Lineage())
    nan_model = lambda Z: np.full(len(Z), np.nan)
    b = ho.ask(nan_model, lineage=ct.Lineage())
    assert ho.state["asks"] == 2 and not b.accepted and a.total_bits > 0
    cert = ho.open(probe, lineage=ct.Lineage())
    assert ho.state["exam"] == "opened" and np.isfinite(cert.artifact["bound"])
    with pytest.raises(ValueError, match="a slice is opened once"):
        ho.open(probe, lineage=ct.Lineage())
    p_ex = np.clip(probe(Xh.iloc[ex]), 0, 1)
    assert abs(cert.artifact["estimate"] - float(((yh[ex] - p_ex) ** 2).mean())) < 1e-12
    yc = rng.normal(size=n)
    hn = pv.seal(X, yc, tempfile.mkdtemp(prefix="chan_"), lineage="self", loss="squared", mode="numeric", loss_range=(0.0, 4.0), exam_share=0.0)
    _, ycn, tn, _ = hn._data()
    a1 = hn.check(lambda Z: np.full(len(Z), 1e6), 4.0, hn.min_width() * 2, lineage=ct.Lineage())
    assert a1.confirmed and hn.state["checks"] == 1
    a2 = hn.check(lambda Z: np.where(np.arange(len(Z)) % 2 == 0, np.nan, 0.0), 2.0, 2.0, lineage=ct.Lineage())
    stat = np.where(np.arange(len(tn)) % 2 == 0, 4.0, np.clip(ycn[tn] ** 2, 0, 4))
    assert hn.state["checks"] == 2 and (a2.confirmed or abs(a2.value - stat.mean()) <= (2.0 - hn._ops()[-1]["tol"] * 4.0) / 2 + 1e-9)
    assert "loss clipped to [0, 4]" in ct.Ledger(hn.ledger.path).active()[0].claim
    return f"probes answered and priced ({ho.state['bits']:.2f} bits); the exam opened once; ranges clip, gaps fall back"


def _lag_world(n=1400, seed=11):
    rng = np.random.default_rng(seed)
    panel = pd.DataFrame({"x": rng.normal(size=n)})
    panel["y"] = 0.3 * panel["x"] + rng.normal(size=n)
    panel["y_lag1"] = panel["y"].shift(1).fillna(0.0)
    panel["start_utc"] = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    panel["end_utc"] = panel["start_utc"] + pd.Timedelta(hours=1)
    S, H = panel.index[:600], panel.index[600:]
    lin = lambda idx: ct.Lineage.of(ct.slice_from_frame(panel, "lagdemo", rows=idx))
    return panel, S, H, lin(S), lin(H)


@lock("TW11", "outcome columns declared at seal: a lagged outcome column that reaches other sealed rows switches every prediction to one row per call, so a model that reads a row's own outcome from its successor certifies nothing it cannot earn row by row; with a random exam split the seal refuses (exam outcomes would ride into tuning rows) and exam='last' is accepted")
def tw11():
    from datetime import timedelta
    panel, S, H, S_lin, H_lin = _lag_world()
    clk = (panel.loc[H, "start_utc"], panel.loc[H, "end_utc"])
    lag = ct.ColumnDeclaration(column="y_lag1", dataset_tag="lagdemo", lag=timedelta(hours=1), outcome_derived=True)
    cols = ["x", "y_lag1"]
    with pytest.raises(ValueError, match=r"seal refused: declared outcome columns carry exam outcomes into tuning rows \(y_lag1 on \d+ rows\); seal with exam='last'"):
        pv.seal(panel.loc[H, cols], panel.loc[H, "y"], tempfile.mkdtemp(), lineage=H_lin, baseline=0.0, baseline_lineage=ct.Lineage(), clock=clk, columns=[lag])
    ho = pv.seal(panel.loc[H, cols], panel.loc[H, "y"], tempfile.mkdtemp(prefix="lag_"), lineage=H_lin, baseline=0.0, baseline_lineage=ct.Lineage(), clock=clk, columns=[lag], exam="last", exam_share=0.25)
    assert ho.protocol["rowwise"] is True
    _, _, tune, ex = ho._data()
    assert max(panel.loc[H].index[tune]) < min(panel.loc[H].index[ex])
    cheat = lambda Z: Z["y_lag1"].reindex(Z.index + 1).fillna(0.0).to_numpy()
    honest = lambda Z: 0.3 * Z["x"].to_numpy()
    ho.ask(cheat, lineage=S_lin, columns=[lag])
    cert = ho.open(cheat, lineage=S_lin, columns=[lag])
    assert cert.artifact["estimate"] > 0.8
    hf = pv.seal(panel.loc[H, cols], panel.loc[H, "y"], tempfile.mkdtemp(prefix="lag_"), lineage=H_lin, baseline=0.0, baseline_lineage=ct.Lineage(), clock=clk, exam="last", exam_share=0.25)
    with pytest.raises(ValueError, match=r"\[y_lag1\] an outcome-derived lag column .* declare it when sealing"):
        hf.ask(honest, lineage=S_lin, columns=[lag])
    return f"one row per call: the successor-reading model's exam loss estimate {cert.artifact['estimate']:.3f} (noise variance 1)"


@lock("TW12", "custody is salted and pickle-free: the public log's target hash is not the plain hash of the target (an 18-row binary target is not recoverable by enumeration), x_sha moves with any cell, the custody files parse only after their bytes match, and mixed column types round-trip exactly")
def tw12():
    import itertools, hashlib as _h
    from provable.holdout import _npy_bytes
    rng = np.random.default_rng(12)
    X = pd.DataFrame({"a": rng.normal(size=18)})
    yb = (rng.random(18) < 0.5).astype(float)
    ho = pv.seal(X, yb, tempfile.mkdtemp(prefix="salt_"), lineage="self", loss="zero_one", mode="numeric", exam_share=0.0)
    target = ho.commit["y_sha"]
    found = any(_h.sha256(_npy_bytes(np.asarray(v, float))).hexdigest() == target for v in itertools.product((0.0, 1.0), repeat=18))
    assert not found
    X2 = X.copy(); X2.iloc[4, 0] += 1e-9
    ho2 = pv.seal(X2, yb, tempfile.mkdtemp(prefix="salt_"), lineage="self", loss="zero_one", mode="numeric", exam_share=0.0)
    assert ho2.commit["x_sha"] != ho.commit["x_sha"]
    Xm = pd.DataFrame({"f": rng.normal(size=12), "i": np.arange(12, dtype=np.int32), "b": rng.random(12) < 0.5, "s": [f"c{i % 3}" for i in range(12)], "k": pd.Categorical([("lo", "hi")[i % 2] for i in range(12)]), "t": pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC")}, index=pd.Index(range(100, 112)))
    hm = pv.seal(Xm, rng.normal(size=12), tempfile.mkdtemp(prefix="salt_"), lineage="self", baseline=0.0, baseline_lineage=ct.Lineage(), exam_share=0.0)
    assert hm._data()[0].equals(Xm) and list(hm._data()[0].dtypes.astype(str)) == list(Xm.dtypes.astype(str))
    Xo = pd.DataFrame({"o": [object()] * 6})
    with pytest.raises(ValueError, match="custody stores strings, numbers and booleans"):
        pv.seal(Xo, np.zeros(6), tempfile.mkdtemp(), lineage="self", baseline=0.0, baseline_lineage=ct.Lineage(), exam_share=0.0)
    pth = os.path.join(ho.root, "x", "x0.npy")
    v = np.load(pth); v[0] += 1.0; np.save(pth, v)
    with pytest.raises(ValueError, match="the rows changed after sealing"):
        pv.Holdout(ho.ledger.path, ho.id)._data()
    return "2^18 candidate targets miss the salted hash; any cell moves x_sha; six column types round-trip"


@lock("TW13", "the audit replays the state machine and the custody head anchors the log: a forged op, a forged un-retirement, an edited last op and a same-size edit seen through a live handle all FAIL, and after a truncation the holdout refuses to answer until the log matches its head again")
def tw13():
    import shutil
    from provable.holdout import _op_hash
    rng = np.random.default_rng(13)
    n = 300
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    yv = 0.5 * X["a"].to_numpy() + rng.normal(size=n)
    m1 = lambda Z: 0.5 * Z["a"].to_numpy()
    def fresh(**kw):
        d = tempfile.mkdtemp(prefix="replay_")
        return d, pv.seal(X, yv, d, lineage="self", baseline=0.0, baseline_lineage=ct.Lineage(), seed=2, **kw)
    def forge(d, hid, action, patch):
        ops = [json.loads(l) for l in open(os.path.join(d, "ledger.jsonl")).read().splitlines()]
        last = [o for o in ops if o.get("op") == "holdout" and o.get("holdout") == hid][-1]
        st = dict(last["state"]); st.update(patch)
        op = {"op": "holdout", "holdout": hid, "action": action, "seq": last["seq"] + 1, "prev": _op_hash(last), "at": last["at"], "state": st}
        with open(os.path.join(d, "ledger.jsonl"), "a") as fh:
            fh.write(json.dumps(op, sort_keys=True, separators=(",", ":")) + "\n")
    d, ho = fresh(exam_share=0.25)
    ho.ask(m1, lineage=ct.Lineage())
    ho.open(m1, lineage=ct.Lineage())
    forge(d, ho.id, "note", {"exam": "sealed"})
    h2 = pv.Holdout(d, ho.id)
    assert not h2.audit().ok
    with pytest.raises(ValueError, match="does not match its custody head"):
        h2.open(m1, lineage=ct.Lineage())
    d, ho = fresh(code="fixed")
    for _ in range(39):
        ho.ask(0.0, lineage=ct.Lineage())
    forge(d, ho.id, "retire", {"tuning": "open"})
    a = pv.Holdout(d, ho.id).audit()
    assert not a.ok and any(r[1] == "FAIL" and ("second retirement" in str(r[2]) or "custody" in r[0]) for r in a.rows)
    d, ho = fresh()
    ho.ask(m1, lineage=ct.Lineage())
    ho.ask(lambda Z: m1(Z) + 0.3 * Z["b"].to_numpy(), lineage=ct.Lineage())
    lines = open(os.path.join(d, "ledger.jsonl")).read().splitlines()
    open(os.path.join(d, "ledger.jsonl"), "w").write("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="edited or truncated after the last answer"):
        pv.Holdout(d, ho.id).ask(m1, lineage=ct.Lineage())
    d, ho = fresh(exam_share=0.25)
    ho.ask(m1, lineage=ct.Lineage())
    ho.open(m1, lineage=ct.Lineage())
    lines = open(os.path.join(d, "ledger.jsonl")).read().splitlines()
    o = json.loads(lines[-2] if json.loads(lines[-1]).get("op") == "register" else lines[-1])
    idx = lines.index(json.dumps(o, sort_keys=True, separators=(",", ":")))
    o["state"]["exam"] = "sealed"
    lines[idx] = json.dumps(o, sort_keys=True, separators=(",", ":"))
    open(os.path.join(d, "ledger.jsonl"), "w").write("\n".join(lines) + "\n")
    assert not pv.Holdout(d, ho.id).audit().ok
    d, ho = fresh()
    ho.ask(m1, lineage=ct.Lineage())
    pth = os.path.join(d, "ledger.jsonl")
    txt = open(pth).read()
    flipped = txt.replace('"answer":"accept"', '"answer":"reject"') if '"answer":"accept"' in txt else txt.replace('"answer":"reject"', '"answer":"accept"')
    assert len(flipped) == len(txt)
    open(pth, "w").write(flipped)
    assert not ho.audit().ok
    return "forged op, forged un-retirement, edited last op and live same-size edit FAIL; truncation stops the holdout"


@lock("TW14", "audit is no oracle: models are refused while any row is in custody, an edited log is not replayed, the book around the holdout is checked (a voided certificate of the log, or a foreign certificate naming the holdout, FAILs), and a declared retirement worded like the threshold replays as declared")
def tw14():
    rng = np.random.default_rng(14)
    n = 300
    X = pd.DataFrame({"a": rng.normal(size=n)})
    yv = 0.5 * X["a"].to_numpy() + rng.normal(size=n)
    m1 = lambda Z: 0.5 * Z["a"].to_numpy()
    d = tempfile.mkdtemp(prefix="oracle_")
    ho = pv.seal(X, yv, d, lineage="self", baseline=0.0, baseline_lineage=ct.Lineage(), exam_share=0.25, seed=3)
    a = ho.ask(m1, lineage=ct.Lineage())
    with pytest.raises(ValueError, match="audit refused: re-running models on rows still in custody"):
        ho.audit(models={a.seq: m1})
    ho.retire(cause="threshold reached by eye")
    with pytest.raises(ValueError, match="audit refused: re-running models"):
        ho.audit(models={a.seq: m1})
    cert = ho.open(m1, lineage=ct.Lineage())
    ho.retire()
    au = ho.audit(models={a.seq: m1})
    assert au.ok and any(r[0].endswith("retire") and r[2].startswith("declared: threshold reached by eye") for r in au.rows)
    led = ct.Ledger(d)
    led.void(cert.id, "withdrawn by hand")
    a2 = pv.Holdout(d, ho.id).audit()
    assert not a2.ok and "voided in the book (withdrawn by hand)" in [r for r in a2.rows if r[0] == "book"][0][2]
    d2 = tempfile.mkdtemp(prefix="oracle_")
    h2 = pv.seal(X, yv, d2, lineage="self", baseline=0.0, baseline_lineage=ct.Lineage(), exam_share=0.25, seed=3)
    fake = ct.issue(claim="names it", guarantee=g(), ell_test=ct.Lineage(), proposer_rows=ct.Lineage(), artifact={"holdout": h2.id}, support=("a",), gram_context={}, kind=ct.CertificateKind.PERFORMANCE)
    ct.Ledger(d2).register(fake)
    a3 = h2.audit()
    assert not a3.ok and "names " + h2.id + " but is not in its log" in [r for r in a3.rows if r[0] == "book"][0][2]
    return "models refused in custody; voided and foreign certificates FAIL the book row; declared wording replays"


@lock("TW15", "claims say what was certified: a declared range narrower than the loss's natural range is named as clipping, brier names its probability clip, non-finite predictions are counted and named with their substitute, and a prediction file changed during operation refuses before it is used, its bytes checked before they are parsed (the same numbers in other bytes refuse too)")
def tw15():
    rng = np.random.default_rng(15)
    n = 400
    X = pd.DataFrame({"a": rng.normal(size=n)})
    yb = (rng.random(n) < 0.5).astype(float)
    hz = pv.seal(X, yb, tempfile.mkdtemp(prefix="claim_"), lineage="self", loss="zero_one", baseline=0.0, baseline_lineage=ct.Lineage(), loss_range=(0.0, 0.25), exam_share=0.25)
    cz = hz.open(lambda Z: np.ones(len(Z)), lineage=ct.Lineage())
    assert "loss clipped to [0, 0.25]" in cz.claim and cz.guarantee.family["clipped"] is True
    hb = pv.seal(X, yb, tempfile.mkdtemp(prefix="claim_"), lineage="self", loss="brier", baseline=0.5, baseline_lineage=ct.Lineage(), exam_share=0.25)
    cb = hb.open(lambda Z: np.where(Z["a"].to_numpy() > 0, np.nan, 0.4), lineage=ct.Lineage())
    k = cb.artifact["substituted"]
    assert k > 0 and "probabilities clipped to [0, 1]" in cb.claim and f"{k} of {cb.guarantee.family['n']} predictions not finite and scored as the sealed baseline's" in cb.claim
    yv = rng.normal(size=n)
    hs = pv.seal(X, yv, tempfile.mkdtemp(prefix="claim_"), lineage="self", baseline=0.0, baseline_lineage=ct.Lineage(), exam_share=0.25)
    a = hs.ask(lambda Z: 0.1 * Z["a"].to_numpy(), lineage=ct.Lineage())
    pth = os.path.join(hs.root, "pred", "baseline.npy")
    v = np.load(pth); v[0] += 1.0; np.save(pth, v)
    with pytest.raises(ValueError, match="custody of HO-.* does not match its seal: the baseline's predictions changed after sealing"):
        pv.Holdout(hs.ledger.path, hs.id).ask(lambda Z: 0.2 * Z["a"].to_numpy(), lineage=ct.Lineage())
    yr = X["a"].to_numpy() + 0.5 * rng.normal(size=n)
    hr = pv.seal(X, yr, tempfile.mkdtemp(prefix="claim_"), lineage="self", baseline=0.0, baseline_lineage=ct.Lineage(), exam_share=0.25)
    acc = hr.ask(lambda Z: Z["a"].to_numpy(), lineage=ct.Lineage())
    assert acc.accepted
    pth = os.path.join(hr.root, "pred", f"{acc.seq}.npy")
    same = np.load(pth)
    with open(pth, "wb") as fh:
        np.lib.format.write_array(fh, same, version=(2, 0))
    assert np.array_equal(np.load(pth), same)
    with pytest.raises(ValueError, match=f"custody of HO-.* does not match its log: prediction file {acc.seq} changed after it was written"):
        pv.Holdout(hr.ledger.path, hr.id).ask(lambda Z: 0.9 * Z["a"].to_numpy(), lineage=ct.Lineage())
    au = pv.Holdout(hr.ledger.path, hr.id).audit()
    assert not au.ok and [r for r in au.rows if r[0] == f"#{acc.seq} ask"][0][2] == "stored predictions do not match the logged hash"
    return f"clip named; brier note; {k} substituted predictions named; a changed baseline file refuses, and so does a prediction file rewritten to the same numbers in other bytes"


@lock("TW16", "outcome columns whatever their tag: a lagged outcome column declared under another dataset tag still switches predictions to one row per call (the baseline included) and still forces exam='last'; after retirement a column built from released rows clears the gate at the exam open; lag refusals, at the seal and after it, name the column and the rule, never a row's window; a DatetimeIndex frame seals and round-trips and duplicate column names refuse")
def tw16():
    import re as _re
    from datetime import timedelta
    panel, S, H, S_lin, H_lin = _lag_world(seed=16)
    clk = (panel.loc[H, "start_utc"], panel.loc[H, "end_utc"])
    other = ct.ColumnDeclaration(column="y_lag1", dataset_tag="some_other_tag", lag=timedelta(hours=1), outcome_derived=True)
    cols = ["x", "y_lag1"]
    with pytest.raises(ValueError, match="carry exam outcomes into tuning rows"):
        pv.seal(panel.loc[H, cols], panel.loc[H, "y"], tempfile.mkdtemp(), lineage=H_lin, baseline=0.0, baseline_lineage=ct.Lineage(), clock=clk, columns=[other])
    calls = []
    frame_baseline = lambda Z: (calls.append(len(Z)), np.zeros(len(Z)))[1]
    ho = pv.seal(panel.loc[H, cols], panel.loc[H, "y"], tempfile.mkdtemp(prefix="tag_"), lineage=H_lin, baseline=frame_baseline, baseline_lineage=ct.Lineage(), clock=clk, columns=[other], exam="last")
    assert ho.protocol["rowwise"] is True and set(calls) == {1}
    own = ct.ColumnDeclaration(column="y_recent", dataset_tag="lagdemo", lag=timedelta(minutes=30), outcome_derived=True)
    with pytest.raises(ValueError) as e:
        ho.ask(lambda Z: 0.3 * Z["x"].to_numpy(), lineage=S_lin, columns=[own])
    assert "[y_recent] an outcome-derived lag column is checked against the sealed clock" in str(e.value) and not _re.search(r"\d{4}-\d{2}-\d{2}T", str(e.value))
    with pytest.raises(ValueError) as e:
        pv.seal(panel.loc[H, cols], panel.loc[H, "y"], tempfile.mkdtemp(), lineage=H_lin, baseline=0.0, baseline_lineage=ct.Lineage(), clock=clk, columns=[other, own], exam="last")
    assert "[y_recent] own outcome rows in column lineage: lag 0:30:00" in str(e.value) and not _re.search(r"\d{4}-\d{2}-\d{2}T", str(e.value))
    rel = ho.retire()
    built = ct.ColumnDeclaration(column="enc_released", dataset_tag="lagdemo", rows=rel.lineage, outcome_derived=True)
    ho.open(lambda Z: 0.3 * Z["x"].to_numpy(), lineage=S_lin.union(rel.lineage), columns=[built])
    Xd = pd.DataFrame({"a": np.arange(8, dtype=float)}, index=pd.date_range("2026-02-01", periods=8, freq="D", tz="UTC"))
    hd = pv.seal(Xd, np.arange(8, dtype=float), tempfile.mkdtemp(prefix="tag_"), lineage="self", baseline=0.0, baseline_lineage=ct.Lineage(), exam_share=0.0)
    assert hd._data()[0].equals(Xd) and hd._data()[0].index.equals(Xd.index)
    Xdup = pd.DataFrame(np.zeros((6, 2)), columns=["a", "a"])
    with pytest.raises(ValueError, match="duplicate column names"):
        pv.seal(Xdup, np.zeros(6), tempfile.mkdtemp(), lineage="self", baseline=0.0, baseline_lineage=ct.Lineage(), exam_share=0.0)
    return "another tag still rowwise (baseline too); released column clears; refusals carry no window; DatetimeIndex round-trips"


@lock("TW17", "the book lock: a model runs outside it, so a model that writes to the same book through another handle returns; two threads on one handle are serialized and each answer is priced against the state it is written to; an audit started while a write is in flight waits for it and passes; a seal racing another seal of the same rows is refused under the lock and leaves no custody behind; reservations decide under the lock; arguments are read before the lock is taken; a reader leaves out a line still being written, and a dead writer's tail is moved aside before the next append")
def tw17():
    import threading, time
    import provable.holdout as Hm
    rng = np.random.default_rng(17)
    n = 1200
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    yv = X["a"].to_numpy() + rng.normal(size=n)
    d = tempfile.mkdtemp(prefix="lock_")
    ho = pv.seal(X, yv, d, lineage="self", baseline=0.0, baseline_lineage=ct.Lineage(), exam_share=0.0)
    def logging_model(Z):
        ct.Ledger(d).note("model evaluated")
        return Z["a"].to_numpy()
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("a", ho.ask(logging_model, lineage=ct.Lineage())), daemon=True)
    t.start(); t.join(60)
    assert not t.is_alive() and out["a"].accepted and "model evaluated" in ct.Ledger(d).statement().notes
    def slow(Z):
        time.sleep(0.2)
        return 0.9 * Z["a"].to_numpy() + 0.1 * Z["b"].to_numpy()
    answers = []
    ts = [threading.Thread(target=lambda: answers.append(ho.ask(slow, lineage=ct.Lineage())), daemon=True) for _ in range(2)]
    [t_.start() for t_ in ts]; [t_.join(60) for t_ in ts]
    st = pv.Holdout(d, ho.id).state
    assert sorted(a.seq for a in answers) == [2, 3] and st["asks"] == 3 and abs(st["bits"] - (out["a"].bits + sum(a.bits for a in answers))) < 1e-12
    assert sorted(a.total_bits for a in answers)[-1] == pytest.approx(st["bits"]) and pv.Holdout(d, ho.id).audit().ok
    orig = Hm._write_atomic
    seen, aud = [], []
    def slow_head(path, data):
        th = threading.Thread(target=lambda: seen.append(pv.Holdout(d, ho.id).audit().ok), daemon=True)
        th.start(); aud.append(th)
        time.sleep(0.3)
        seen.append(th.is_alive())
        return orig(path, data)
    Hm._write_atomic = slow_head
    try:
        ho.ask(lambda Z: Z["b"].to_numpy(), lineage=ct.Lineage())
    finally:
        Hm._write_atomic = orig
    aud[0].join(60)
    assert seen == [True, True]
    d3 = tempfile.mkdtemp(prefix="lock_")
    inner = {}
    def racing_baseline(Z):
        if not inner:
            inner["ho"] = pv.seal(X, yv, d3, lineage="self", baseline=0.0, baseline_lineage=ct.Lineage(), exam_share=0.0)
        return np.zeros(len(Z))
    with pytest.raises(ValueError, match=r"seal refused: rows already under seal in HO-"):
        pv.seal(X, yv, d3, lineage="self", baseline=racing_baseline, baseline_lineage=ct.Lineage(), exam_share=0.0)
    assert ct.Ledger(d3).holdout_ids() == [inner["ho"].id] and os.listdir(os.path.join(d3, "holdouts")) == [inner["ho"].id]
    led = ct.Ledger(tempfile.mkdtemp(prefix="lock_"), budget=0.10)
    got, errs = [], []
    def reserve():
        try:
            got.append(led.reserve(0.06))
        except ValueError as e:
            errs.append(str(e))
    rs = [threading.Thread(target=reserve, daemon=True) for _ in range(4)]
    [t_.start() for t_ in rs]; [t_.join(60) for t_ in rs]
    assert len(got) == 1 and len(errs) == 3 and led.statement().reserved == pytest.approx(0.06)
    pth = os.path.join(led.path, "ledger.jsonl")
    with open(pth, "a") as fh:
        fh.write('{"op": "note", "text": "half writ')
    assert [o["op"] for o in ct.Ledger(led.path)._ops] == ["reserve"]
    with open(pth, "a") as fh:
        fh.write('ten"}\n')
    assert [o["op"] for o in ct.Ledger(led.path)._ops] == ["reserve", "note"]
    def noting_columns():
        ct.Ledger(d).note("column builder ran")
        yield ct.ColumnDeclaration(column="b", dataset_tag="t", outcome_derived=False)
    def noting_vocabulary():
        ct.Ledger(led.path).note("vocabulary builder ran")
        yield "pair-1"
    done = {}
    t2 = threading.Thread(target=lambda: done.update(a=ho.ask(lambda Z: Z["a"].to_numpy(), lineage=ct.Lineage(), columns=noting_columns()), r=led.reserve(0.01, vocabulary=noting_vocabulary())), daemon=True)
    t2.start(); t2.join(60)
    assert not t2.is_alive() and "a" in done and "r" in done
    with open(pth, "a") as fh:
        fh.write('{"op": "note", "text": "a writer that died mid-wri')
    led2 = ct.Ledger(led.path)
    led2.note("the next append")
    assert [o.get("text") for o in ct.Ledger(led.path)._ops if o["op"] == "note"][-1] == "the next append" and open(pth + ".torn").read().startswith('{"op": "note", "text": "a writer that died')
    return "a book-writing model returns; two threads give #2 and #3, both priced; an audit waits out a write and passes; a racing seal refused under the lock; four racing reservations leave one; arguments that write to the book are read before the lock; a half-written line is left out until whole, and a dead writer's tail moves aside"


@lock("TW18", "the book around a holdout: a certificate whose test rows meet rows in custody is refused at register unless it is the holdout's own registration (a performance certificate whose content hashes to a log id, registered once), and clears once retire() has released those rows (a book without holdouts registers as before); an artifact taking a log id is refused; the seal refuses rows an active certificate names, uninstrumented ones included as far as known; a record written around the API FAILs the audit by its rows (voided or not), by naming the holdout in any field but its peeks while active, by preceding the seal on its rows, or by taking a log id")
def tw18():
    rng = np.random.default_rng(18)
    n = 300
    X = pd.DataFrame({"a": rng.normal(size=n)})
    yv = X["a"].to_numpy() + rng.normal(size=n)
    m1 = lambda Z: Z["a"].to_numpy()
    d = tempfile.mkdtemp(prefix="book_")
    ho = pv.seal(X, yv, d, lineage="self", baseline=0.0, baseline_lineage=ct.Lineage(), exam_share=0.25, seed=5)
    cert = ho.open(m1, lineage=ct.Lineage())
    fam = {k: v for k, v in cert.guarantee.family.items() if k != "holdout"}
    fake = ct.issue(claim="performance: a bound computed outside the holdout", guarantee=ct.Guarantee(ct.GuaranteeKind.FWER, 0.05, fam), ell_test=ho.exam_lineage(), proposer_rows=ct.Lineage(), artifact={"bound": 0.01}, support=("model:0000",), gram_context={}, kind=ct.CertificateKind.PERFORMANCE)
    with pytest.raises(ValueError, match=r"register refused: the certificate's test rows meet rows in the custody of HO-"):
        ct.Ledger(d).register(fake)
    fid = ct.content_id(ct.to_dict(fake))
    with open(os.path.join(d, "ledger.jsonl"), "a") as fh:
        fh.write(json.dumps({"op": "register", "id": fid, "record": ct.to_dict(fake), "reservation": None}, sort_keys=True, separators=(",", ":")) + "\n")
    a1 = pv.Holdout(d, ho.id).audit()
    assert not a1.ok and f"{fid} registered on {ho.commit['n_exam']} row(s) in the custody of {ho.id} and not in its log" in [r for r in a1.rows if r[0] == "book"][0][2]
    ct.Ledger(d).void(fid, "withdrawn")
    a2 = pv.Holdout(d, ho.id).audit()
    assert not a2.ok and "(voided: withdrawn)" in [r for r in a2.rows if r[0] == "book"][0][2]
    d2 = tempfile.mkdtemp(prefix="book_")
    h2 = pv.seal(X, yv, d2, lineage="self", baseline=0.0, baseline_lineage=ct.Lineage(), exam_share=0.25, seed=5)
    named = ct.issue(claim=f"performance: model 0000 at most 0.01 ({h2.id}, exam slice)", guarantee=g(), ell_test=ct.Lineage(), proposer_rows=ct.Lineage(), artifact={"source": "elsewhere"}, support=("a",), gram_context={}, kind=ct.CertificateKind.PERFORMANCE)
    ct.Ledger(d2).register(named)
    a3 = h2.audit()
    assert not a3.ok and "names " + h2.id + " but is not in its log" in [r for r in a3.rows if r[0] == "book"][0][2]
    d3 = tempfile.mkdtemp(prefix="book_")
    h3 = pv.seal(X, yv, d3, lineage="self", baseline=0.0, baseline_lineage=ct.Lineage(), exam_share=0.25, seed=5)
    c_tune = mk("on the tuning rows", h3.tuning_lineage(), ct.Lineage())
    with pytest.raises(ValueError, match="register refused"):
        ct.Ledger(d3).register(c_tune)
    h3.retire()
    ct.Ledger(d3).register(c_tune)
    with pytest.raises(ValueError, match="register refused"):
        ct.Ledger(d3).register(mk("on the exam rows", h3.exam_lineage(), ct.Lineage()))
    h3.open(m1, lineage=ct.Lineage())
    au = h3.audit()
    assert au.ok and len(ct.Ledger(d3).active()) == 2
    led0 = ct.Ledger(tempfile.mkdtemp(prefix="book_"))
    led0.register(c_tune)
    d4 = tempfile.mkdtemp(prefix="book_")
    h4 = pv.seal(X, yv, d4, lineage="self", baseline=0.0, baseline_lineage=ct.Lineage(), exam_share=0.25, seed=5)
    c4 = h4.open(m1, lineage=ct.Lineage())
    spoof = ct.issue(claim="an artifact that takes the holdout's id", guarantee=g(), ell_test=h4.tuning_lineage(), proposer_rows=ct.Lineage(), artifact={"certificate_id": c4.id}, support=("a",), gram_context={}, kind=ct.CertificateKind.ARTIFACT)
    assert spoof.id == c4.id
    with pytest.raises(ValueError, match=r"register refused: CD-[0-9A-F]+ is an id in the log of HO-[0-9A-F]+, which issues performance certificates only; a artifact certificate cannot take it"):
        ct.Ledger(d4).register(spoof)
    with pytest.raises(ValueError, match=r"register refused: CD-[0-9A-F]+, a certificate of HO-[0-9A-F]+, is already registered in the book"):
        ct.Ledger(d4).register(c4)
    pth = os.path.join(d4, "ledger.jsonl")
    lines = open(pth).read().splitlines()
    forged = ct.issue(claim="performance: a bound computed outside the holdout", guarantee=g(), ell_test=h4.exam_lineage(), proposer_rows=ct.Lineage(), artifact={"bound": 0.01}, support=("a",), gram_context={}, kind=ct.CertificateKind.PERFORMANCE)
    lines.insert(0, json.dumps({"op": "register", "id": forged.id, "record": ct.to_dict(forged), "reservation": None}, sort_keys=True, separators=(",", ":")))
    open(pth, "w").write("\n".join(lines) + "\n")
    a5 = pv.Holdout(d4, h4.id).audit()
    assert not a5.ok and f"{forged.id} precedes the seal of {h4.id}, spends its rows and was not voided before it" in [r for r in a5.rows if r[0] == "book"][0][2]
    lines = open(pth).read().splitlines()
    spoof_line = json.dumps({"op": "register", "id": c4.id, "record": ct.to_dict(spoof), "reservation": None}, sort_keys=True, separators=(",", ":"))
    open(pth, "w").write("\n".join(lines[1:] + [spoof_line]) + "\n")
    a6 = pv.Holdout(d4, h4.id).audit()
    assert not a6.ok and f"{c4.id}, an id of the log of {h4.id}, is registered again or as another kind" in [r for r in a6.rows if r[0] == "book"][0][2]
    d5 = tempfile.mkdtemp(prefix="book_")
    led5 = ct.Ledger(d5, budget=0.5)
    half = ct.issue(claim="existence on these rows, proposal rows unknown", guarantee=ct.Guarantee(ct.GuaranteeKind.FWER, 0.05, {}), ell_test=ct.Lineage.of(ct.Slice(source_id=ct.source_fingerprint(X.reset_index(drop=True)), dataset_tag="t", rows=frozenset(range(50)), t0=datetime(1970, 1, 1, tzinfo=UTC), t1=datetime(1970, 1, 1, tzinfo=UTC))), proposer_rows=unknown_lineage(), artifact={}, support=("a", "b"), gram_context={}, kind=ct.CertificateKind.EXISTENCE)
    led5.register(half)
    assert half.uninstrumented()
    with pytest.raises(ValueError, match="seal refused: rows already spent in the book"):
        pv.seal(X, yv, led5, lineage="self", baseline=0.0, baseline_lineage=ct.Lineage(), exam_share=0.25, seed=5)
    d6 = tempfile.mkdtemp(prefix="book_")
    h6 = pv.seal(X, yv, d6, lineage="self", baseline=0.0, baseline_lineage=ct.Lineage(), exam_share=0.25, seed=5)
    ct.Ledger(d6).register(ct.issue(claim="unrelated", guarantee=g(), ell_test=ct.Lineage(), proposer_rows=ct.Lineage(), artifact={}, support=(h6.id,), gram_context={}, kind=ct.CertificateKind.CARRIER))
    a7 = h6.audit()
    assert not a7.ok and f"names {h6.id} but is not in its log" in [r for r in a7.rows if r[0] == "book"][0][2]
    return "register refuses rows in custody and a log id taken by an artifact or registered twice, clears released rows; forged records FAIL by their rows (voided too), by name, before the seal, or under a log id"


@lock("TW19", "numpy's floating-point error state is not a channel: with np.seterr raising on underflow and invalid operations, probes built to raise on a sealed target are answered and priced by ask, check and open, the audit replays under the same setting, and the caller's setting stands afterwards; with warnings as errors, a sealed clock's nanoseconds decide nothing")
def tw19():
    rng = np.random.default_rng(19)
    n = 120
    X = pd.DataFrame({"a": rng.normal(size=n)})
    yb = (rng.random(n) < 0.5).astype(float)
    def probe(j):
        def f(Z):
            p = np.full(len(Z), 0.5)
            idx = list(Z.index)
            if j in idx:
                p[idx.index(j)] = 1e-160
            p[[i for i, lab in enumerate(idx) if lab != j][0]] = 1e200
            return p
        return f
    ho = pv.seal(X, yb, tempfile.mkdtemp(prefix="fp_"), lineage="self", loss="squared", baseline=0.5, baseline_lineage=ct.Lineage(), exam_share=0.25, seed=1)
    hn = pv.seal(X, yb, tempfile.mkdtemp(prefix="fp_"), lineage="self", loss="squared", loss_range=(0.0, 4.0), mode="numeric", exam_share=0.0)
    _, _, tune, ex = ho._data()
    _, _, tn, _ = hn._data()
    w = 2 * hn.min_width()
    old = np.geterr()
    try:
        np.seterr(all="ignore")
        np.seterr(under="raise", invalid="raise")
        answers = [ho.ask(probe(int(j)), lineage=ct.Lineage()) for j in tune[:20]]
        cert = ho.open(probe(int(ex[0])), lineage=ct.Lineage())
        checks = [hn.check(probe(int(j)), 0.3, w, lineage=ct.Lineage()) for j in tn[:4]]
        au, aun = ho.audit(), hn.audit()
        kept = np.geterr()
    finally:
        np.seterr(**old)
    assert len(answers) == 20 and ho.state["asks"] == 20 and ho.state["bits"] > 0 and ho.state["exam"] == "opened" and cert.id in ho.state["certificates"]
    assert len(checks) == 4 and hn.state["checks"] == 4 and au.ok and aun.ok and kept["under"] == "raise" and kept["invalid"] == "raise"
    import warnings
    start = pd.Series(pd.date_range("2026-01-01", periods=40, freq="h", tz="UTC", unit="ns"))
    lin = ct.Lineage.of(ct.Slice(source_id="ns-src", dataset_tag="t", rows=frozenset(range(40)), t0=datetime(2025, 1, 1, tzinfo=UTC), t1=datetime(2027, 1, 1, tzinfo=UTC)))
    opened = []
    for ns in (0, 7):
        end = start + pd.Timedelta(hours=1)
        end.iloc[-1] = end.iloc[-1] + pd.Timedelta(nanoseconds=ns)
        hc = pv.seal(X.iloc[:40], yb[:40], tempfile.mkdtemp(prefix="fp_"), lineage=lin, baseline=0.0, baseline_lineage=ct.Lineage(), exam_share=0.25, exam="last", clock=(start, end))
        hc.retire()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            hc.open(lambda Z: Z["a"].to_numpy(), lineage=ct.Lineage())
        opened.append(hc.state["exam"])
    assert opened == ["opened", "opened"]
    return f"20 asks priced ({ho.state['bits']:.2f} bits), the exam opened, 4 checks answered, both audits PASS under under='raise'; warnings as errors read nothing from a sealed clock's nanoseconds"


@lock("TW20", "the transcript is the public log: one row per operation in log order, its log numbers the keys audit(models=...) takes; an ask's bits and running codelength equal its Answer's, a check's bits are its query bits and its running codelength the next check's price, opens and the family carry their certificates; the numeric render prices the next check before the first answer")
def tw20():
    panel, S, H, S_lin, H_lin, d, ho = _ho_basic(seed=20)
    X, y = panel.loc[S, HFEATS], panel.loc[S, "y"]
    keep = {"h0": _ols(X, y, ["h0"]), "h0 h1": _ols(X, y, ["h0", "h1"]), "h3": _ols(X, y, ["h3"])}
    answers = [ho.ask(f, lineage=S_lin, label=name) for name, f in keep.items()]
    cert = ho.open(keep["h0 h1"], lineage=S_lin, label="h0 h1")
    ho.retire()
    rows = ho.transcript().rows
    assert [r["seq"] for r in rows] == list(range(len(rows))) and [r["action"] for r in rows] == ["seal", "ask", "ask", "ask", "open", "retire"]
    for a, r in zip(answers, rows[1:4]):
        assert r["seq"] == a.seq and r["label"] in keep and r["answer"] == ("ACCEPT" if a.accepted else "REJECT") and abs(r["bits"] - a.bits) < 1e-12 and abs(r["transcript_bits"] - a.total_bits) < 1e-12
    assert rows[4]["certificate"] == cert.id and rows[4]["bits"] == 0.0 and rows[4]["transcript_bits"] == pytest.approx(ho.state["bits"])
    au = ho.audit(models={r["seq"]: keep[r["label"]] for r in rows if r["label"] in keep})
    assert au.ok and sum(1 for r in au.rows if r[0].endswith("model")) == 4
    _, _, hn = _numeric(n=2000, seed=21)
    assert "next check priced at" in hn.render() and "1.436 bits (level 0.0185)" in hn.render()
    a1 = hn.check(_thr(0.5), _err(0.5), 2 * hn.min_width(), lineage=ct.Lineage(), label="t50")
    a2 = hn.check(_thr(0.5), _err(0.5) + 0.2, 2 * hn.min_width(), lineage=ct.Lineage(), label="t50")
    rows = hn.transcript().rows
    checks = [r for r in rows if r["action"] == "check"]
    assert [r["action"] for r in rows] == ["seal", "family", "check", "check"] and rows[1]["certificate"] == hn.state["family"]
    assert checks[0]["bits"] == pytest.approx(a1.bits) and checks[0]["transcript_bits"] == pytest.approx(a1.total_bits) and checks[1]["bits"] == pytest.approx(checks[0]["transcript_bits"]) and checks[1]["transcript_bits"] == pytest.approx(a2.total_bits)
    return f"{len(ho.transcript().rows)} comparative rows and 4 numeric rows match their Answers; audit keyed from the transcript re-runs 4 models"


def _clipped_chi2(rng, reps, n, hi=16.0):
    return np.minimum(rng.standard_normal((reps, n)) ** 2, hi)


CHI2_16_MEAN = 0.9998794954488747


@lock("TW21", "the betting tail is exact where the Gaussian bar is not: on squared standard normal errors clipped to [0, 16] (mean 0.99988), 100 rows a bound, the betting bound falls below the mean in at most its level of 6000 draws at the clean bar and at 10 bits (none here), while the Gaussian bar misses beyond its level at both")
def tw21():
    rng = np.random.default_rng(21)
    reps, n = 6000, 100
    x = _clipped_chi2(rng, reps, n)
    rows = []
    for bits in (0.0, 10.0):
        level = 0.05 * 2.0 ** -bits
        bet = np.array([tr.betting_bound(r, 0.0, 16.0, 0.05, bits, side="upper") for r in x])
        gau = x.mean(axis=1) + tr.z_bar(0.05, bits) * x.std(axis=1, ddof=1) / np.sqrt(n)
        miss_b, miss_g = int((bet < CHI2_16_MEAN).sum()), int((gau < CHI2_16_MEAN).sum())
        assert miss_b <= level * reps + 3 * np.sqrt(level * (1 - level) * reps)
        assert miss_g > level * reps + 3 * np.sqrt(level * (1 - level) * reps)
        rows.append(f"{bits:g} bits: betting misses {miss_b}/{reps}, Gaussian {miss_g}/{reps} (level {level:.2e})")
    return "; ".join(rows)


@lock("TW22", "tail auto resolves at the seal and is recorded resolved: a declared range or a bounded loss gives betting, an unbounded loss without a range gives gaussian; the claim names the tail (CLT approximation, or exact); hoeffding and betting without a range refuse by name; audit recomputes a betting certificate")
def tw22():
    panel = _ho_world(seed=22)
    S, H, S_lin, H_lin = _ho_split(panel, 800)
    ys = panel.loc[S, "y"]
    X, y = panel.loc[S, HFEATS], panel.loc[S, "y"]
    f = _ols(X, y, ["h0", "h1"])
    got = {}
    for name, kw in (("none", {}), ("range", {"loss_range": (0.0, 16.0)}), ("named gaussian", {"loss_range": (0.0, 16.0), "tail": "gaussian"})):
        ho = pv.seal(panel.loc[H, HFEATS], panel.loc[H, "y"], tempfile.mkdtemp(prefix="tail_"), lineage=H_lin, baseline=float(ys.mean()), baseline_lineage=S_lin, seed=22, **kw)
        cert = ho.open(f, lineage=S_lin)
        got[name] = (ho.protocol["tail"], cert)
        assert cert.guarantee.family["tail"] == ho.protocol["tail"]
    assert got["none"][0] == "gaussian" and "CLT approximation" in got["none"][1].claim
    assert got["range"][0] == "betting" and "exact, betting bound" in got["range"][1].claim
    assert got["named gaussian"][0] == "gaussian"
    assert got["range"][1].artifact["width"] > got["named gaussian"][1].artifact["width"] > 0
    yb = (panel["y"] > panel["y"].median()).astype(float)
    hb = pv.seal(panel.loc[H, HFEATS], yb.loc[H], tempfile.mkdtemp(prefix="tail_"), lineage=H_lin, loss="brier", baseline=float(yb.loc[S].mean()), baseline_lineage=S_lin, seed=22)
    assert hb.protocol["tail"] == "betting"
    for t in ("hoeffding", "betting"):
        with pytest.raises(ValueError, match=f"{'Hoeffding' if t == 'hoeffding' else 'betting'} tail needs a declared loss range"):
            pv.seal(panel.loc[H, HFEATS], panel.loc[H, "y"], tempfile.mkdtemp(prefix="tail_"), lineage=H_lin, baseline=0.0, baseline_lineage=S_lin, tail=t)
    ho = pv.seal(panel.loc[H, HFEATS], panel.loc[H, "y"], tempfile.mkdtemp(prefix="tail_"), lineage=H_lin, baseline=float(ys.mean()), baseline_lineage=S_lin, seed=23, loss_range=(0.0, 16.0))
    ho.ask(f, lineage=S_lin)
    ho.open(f, lineage=S_lin)
    ho.retire()
    aud = ho.audit()
    assert aud.ok, aud.render()
    return f"no range -> gaussian, range -> betting (width {got['range'][1].artifact['width']:.3f} vs Gaussian {got['named gaussian'][1].artifact['width']:.3f}), brier -> betting; audit {sum(1 for r in aud.rows if r[1] == 'PASS')} PASS"


@lock("TW23", "betting against Hoeffding: on squared standard normal errors clipped to [0, 16], the betting width is at most two thirds of the Hoeffding width at 100, 400 and 2000 rows, at the clean bar and at 10 bits, and its ratio to the Gaussian width falls as rows grow")
def tw23():
    rng = np.random.default_rng(23)
    out, ratios = [], {}
    for n in (100, 400, 2000):
        x = _clipped_chi2(rng, 60, n)
        for bits in (0.0, 10.0):
            bet = np.mean([tr.betting_bound(r, 0.0, 16.0, 0.05, bits) - r.mean() for r in x])
            gau = float(np.mean(tr.z_bar(0.05, bits) * x.std(axis=1, ddof=1) / np.sqrt(n)))
            hoe = tr.hoeffding_width(0.05, bits, n, 16.0)
            assert bet <= hoe * 2 / 3
            ratios[(n, bits)] = bet / gau
            out.append(f"n {n}, {bits:g} bits: betting/Gaussian {bet / gau:.2f}, Hoeffding/Gaussian {hoe / gau:.2f}")
    for bits in (0.0, 10.0):
        assert ratios[(2000, bits)] < ratios[(400, bits)] < ratios[(100, bits)]
    return "; ".join(out)


# ======================================================================
# TX: Guess and Check
# ======================================================================
def _bern_world(n, seed):
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    yb = (rng.random(n) < u).astype(float)
    return pd.DataFrame({"u": u}), yb


def _thr(t):
    return lambda Z: (Z["u"].to_numpy() > t).astype(float)


def _err(t):
    return t * t / 2 + (1 - t) ** 2 / 2


def _numeric(n=2000, seed=0, exam_share=0.0):
    X, yb = _bern_world(n, seed)
    return X, yb, pv.seal(X, yb, tempfile.mkdtemp(prefix="gnc_"), lineage="self", loss="zero_one", mode="numeric", exam_share=exam_share, seed=seed)


@lock("TX1", "Guess and Check rules recomputed from the rows: confirm iff the guess lies within width - tol of the value; a failure releases the value on the grid of step width - tol (error at most half a step); a width at min_width halts on failure and the mechanism stops")
def tx1():
    X, yb, ho = _numeric(seed=1)
    _, _, tune, _ = ho._data()
    pred = lambda t: _thr(t)(X)
    a_h = lambda t: float((pred(t)[tune] != yb[tune]).mean())
    mw = ho.min_width()
    beta = tr.gnc_level(0.05, 1, [])
    assert abs(mw - tr.hoeffding_tol(beta, len(tune))) < 1e-15
    a1 = ho.check(_thr(0.5), a_h(0.5), mw + 0.02, lineage=ct.Lineage())
    assert a1.confirmed and a1.value == a_h(0.5) and a1.level == pytest.approx(beta)
    mw2 = ho.min_width()
    a2 = ho.check(_thr(0.3), a_h(0.3) + mw2 + 0.05, mw2 + 0.04, lineage=ct.Lineage())
    gamma = 0.04
    assert not a2.confirmed and abs(a2.value - a_h(0.3)) <= gamma / 2 + 1e-12 and abs(a2.value / gamma - round(a2.value / gamma)) < 1e-9
    mw3 = ho.min_width()
    assert mw3 > mw2
    a3 = ho.check(_thr(0.7), a_h(0.7) + 0.2, mw3, lineage=ct.Lineage())
    assert a3.halted and a3.value is None and ho.state["tuning"] == "halted"
    with pytest.raises(ValueError, match=r"check refused: Guess and Check on HO-.* halted at #4"):
        ho.check(_thr(0.5), 0.3, 0.2, lineage=ct.Lineage())
    return f"min widths {mw:.4f} < {mw2:.4f} < {mw3:.4f}; confirm, fail on grid 0.04, halt"


@lock("TX2", "simultaneous coverage with failures: an adaptive analyst (guesses from its own rows, a third deliberately off, the next threshold chosen from released values) runs 12 checks per holdout on Bernoulli rows with known population error; every released interval covers its population value in all but at most delta of 100 holdouts")
def tx2():
    rng = np.random.default_rng(77)
    reps, q = 100, 12
    bad, fails, total = 0, 0, 0
    for rep in range(reps):
        X, yb, ho = _numeric(n=1500, seed=1000 + rep)
        Xs, ys = _bern_world(400, 5000 + rep)
        t, step, miss = 0.2, 0.15, False
        for j in range(q):
            g = float((_thr(t)(Xs) != ys).mean()) + (0.25 if j % 3 == 2 else 0.0)
            a = ho.check(_thr(t), g, ho.min_width() * 1.5, lineage=ct.Lineage())
            total += 1
            fails += int(not a.confirmed)
            lo, hi = a.interval
            miss = miss or not (lo <= _err(t) <= hi)
            t = min(max(t + (step if a.value > 0.26 else -step / 2) * (1 if t < 0.5 else -1), 0.05), 0.95)
        bad += int(miss)
    assert bad / reps <= 0.05 and fails >= 0.3 * total
    return f"{bad}/{reps} holdouts with any miss; {fails}/{total} checks failed"


@lock("TX3", "bits accounting: every check op logs level = delta 2^-bits with bits = log2(nu / (c_(i-1) c_f)) from the failures before it; the family certificate is registered before the first answer and charges the book delta once across all checks")
def tx3():
    X, yb, ho = _numeric(seed=3)
    _, _, tune, _ = ho._data()
    for j, t in enumerate((0.5, 0.4, 0.6, 0.3, 0.5)):
        a_h = float((_thr(t)(X)[tune] != yb[tune]).mean())
        ho.check(_thr(t), a_h + (0.3 if j % 2 else 0.0), ho.min_width() * 2, lineage=ct.Lineage())
    ops = ho._ops()
    assert [o["action"] for o in ops] == ["seal", "family", "check", "check", "check", "check", "check"]
    gam = []
    for i, o in enumerate([o for o in ops if o["action"] == "check"], start=1):
        assert abs(o["level"] - tr.gnc_level(0.05, i, gam)) < 1e-18 and abs(o["level"] - 0.05 * 2.0 ** (-o["query_bits"])) < 1e-15
        if o["outcome"] == "failed":
            gam.append(o["gamma"])
    st = ct.Ledger(ho.ledger.path).statement()
    assert st.charged == pytest.approx(0.05) and len(st.items) == 1 and "simultaneously" in st.items[0]["claim"]
    fam = [c for c in ct.Ledger(ho.ledger.path).active()][0]
    assert fam.kind == ct.CertificateKind.PERFORMANCE and fam.ell_test == ho.tuning_lineage() and ho.audit().ok
    return f"5 checks, {len(gam)} failures; levels {ops[2]['level']:.3g} .. {ops[-1]['level']:.3g}; one charge of 0.05"


@lock("TX4", "numeric mode with an exam slice: the exam opens at the clean bar after the checks (the zero-one loss has a range, so the betting bound, recomputed from the exam rows), carries the check transcript's bits as a peek on the tuning rows, and the book holds two charges (the check family and the exam certificate)")
def tx4():
    X, yb, ho = _numeric(seed=4, exam_share=0.25)
    _, _, tune, ex = ho._data()
    for t in (0.5, 0.45):
        ho.check(_thr(t), float((_thr(t)(X)[tune] != yb[tune]).mean()), ho.min_width() * 2, lineage=ct.Lineage())
    cert = ho.open(_thr(0.5), lineage=ct.Lineage())
    le = (_thr(0.5)(X)[ex] != yb[ex]).astype(float)
    assert ho.protocol["tail"] == "betting" and abs(cert.artifact["bound"] - tr.betting_bound(le, 0.0, 1.0, 0.05, 0.0)) < 1e-12
    pk = ct.peeks_of(cert)
    assert pk[0]["bits"] == pytest.approx(ho.state["bits"]) and pk[0]["rows"] == ho.tuning_lineage()
    assert ct.Ledger(ho.ledger.path).statement().charged == pytest.approx(0.10)
    return f"exam bound {cert.artifact['bound']:.4f}; peek {pk[0]['bits']:.2f} bits; charged 0.10"


@lock("TX5", "the Guess and Check family: registered before the first check even after the exam opened, superseded (predecessor voided, the charge held at delta) when a check widens the proposal lineage; a voided family stops the checks, and a family missing from the book refuses by name")
def tx5():
    X, yb, ho = _numeric(seed=5, exam_share=0.25)
    ho.open(_thr(0.5), lineage=ct.Lineage())
    _, _, tune, _ = ho._data()
    t_err = lambda t: float((_thr(t)(X)[tune] != yb[tune]).mean())
    ho.check(_thr(0.5), t_err(0.5), ho.min_width() * 2, lineage=ct.Lineage())
    led = ct.Ledger(ho.ledger.path)
    assert led.statement().charged == pytest.approx(0.10) and ho.state["family"] is not None
    first = ho.state["family"]
    other = ct.Lineage.of(ct.slice_from_frame(frame("dataset_2", 9), "dataset_2", range(0, 12)))
    ho.check(_thr(0.45), t_err(0.45), ho.min_width() * 2, lineage=other)
    second = ho.state["family"]
    st = ct.Ledger(ho.ledger.path).statement()
    assert second != first and first in {v["id"] for v in st.voids} and st.charged == pytest.approx(0.10)
    fam = [c for c in ct.Ledger(ho.ledger.path).active() if c.id == second][0]
    assert ct.identity_overlap(fam.ell_prop, other) and ho.audit().ok
    ct.Ledger(ho.ledger.path).void(second, "withdrawn")
    with pytest.raises(ValueError, match=r"check refused: the family certificate CD-[0-9A-F]+ of HO-.* was voided"):
        ho.check(_thr(0.5), t_err(0.5), ho.min_width() * 2, lineage=ct.Lineage())
    X2, yb2, h2 = _numeric(seed=6)
    _, _, tune2, _ = h2._data()
    h2.check(_thr(0.5), float((_thr(0.5)(X2)[tune2] != yb2[tune2]).mean()), h2.min_width() * 2, lineage=ct.Lineage())
    pth = os.path.join(h2.ledger.path, "ledger.jsonl")
    keep = [l for l in open(pth).read().splitlines() if not (json.loads(l).get("op") == "register" and json.loads(l).get("id") == h2.state["family"])]
    open(pth, "w").write("\n".join(keep) + "\n")
    with pytest.raises(ValueError, match=r"check refused: the family certificate CD-[0-9A-F]+ of HO-.* is missing from the book"):
        pv.Holdout(h2.ledger.path, h2.id).check(_thr(0.5), 0.3, 0.3, lineage=ct.Lineage())
    return f"family {first} superseded by {second}; book held at 0.10; voided family stops the checks"


# ======================================================================
# TY: PEEKED, R7, rows_for_bits, the performance kind
# ======================================================================
def _peeked_cert():
    if "pk" not in _STATE:
        panel, S, H, S_lin, H_lin, d, ho = _ho_basic(seed=11)
        X, y = panel.loc[S, HFEATS], panel.loc[S, "y"]
        for cols in (["h0"], ["h0", "h1"], ["h2"], ["h0", "h1", "h4"]):
            ho.ask(_ols(X, y, cols), lineage=S_lin)
        cert = ho.open(_ols(X, y, ["h0", "h1"]), lineage=S_lin)
        _STATE["pk"] = (panel, ho, cert, d)
    return _STATE["pk"]


@lock("TY1", "FRESH -> PEEKED(L) -> NOT FRESH: an exam certificate reused on fresh rows, on the tuning rows its transcript saw, and on its own exam rows; inflation exactly 2^L and pass level level 2^-L when PEEKED; the refusal names the rows")
def ty1():
    panel, ho, cert, d = _peeked_cert()
    L = ct.peeks_of(cert)[0]["bits"]
    fresh = ct.Lineage.of(ct.slice_from_frame(_ho_world(n=60, seed=99), "hdemo", rows=range(60)))
    a0 = pv.reuse([cert], fresh, ledger=d, level=0.05)
    a1 = pv.reuse([cert], ho.tuning_lineage(), ledger=d, level=0.05)
    assert a0.verdict == ct.Verdict.FRESH and a0.inflation == 1.0 and a0.level == 0.05 and a0.bits == 0.0
    assert a1.verdict == ct.Verdict.PEEKED and a1.admitted and a1.inflation == pytest.approx(2.0 ** L) and a1.level == pytest.approx(0.05 * 2.0 ** -L) and a1.bits == pytest.approx(L)
    assert a1.peeked()[0].peeks[0]["holdout"] == ho.id and a1.items[0].bits == pytest.approx(L) and ct.Verdict.PEEKED.label == "PEEKED"
    with pytest.raises(ct.FreshnessError, match=r"refused: certificate lineage meets the consuming rows. \[performance: .*rows \["):
        pv.reuse([cert], ho.exam_lineage(), ledger=d)
    a2 = pv.reuse([cert], ho.exam_lineage(), ledger=d, on_stale="demote")
    assert a2.verdict == ct.Verdict.NOT_FRESH and a2.demotion[0].status == "MISCALIBRATED"
    return f"FRESH at 0.05; PEEKED at {L:.2f} bits: inflation {a1.inflation:.2f}, pass level {a1.level:.4g}; NOT FRESH refused"


@lock("TY2", "peeks compose: one transcript per holdout at its largest bits, distinct holdouts add; a consuming certificate inherits its parent's peeks; a mixed reuse prices the sum")
def ty2():
    panel, ho, cert, d = _peeked_cert()
    tl = ho.tuning_lineage()
    mk_p = lambda hid, bits, rows: ct.issue(claim=f"c {hid} {bits}", guarantee=g(), ell_test=ct.Lineage(), proposer_rows=ct.Lineage(), artifact={}, support=("a",), gram_context={}, kind=ct.CertificateKind.CARRIER, peeks=[{"holdout": hid, "bits": bits, "rows": rows}])
    c1, c2, c3 = mk_p("HO-A", 3.0, tl), mk_p("HO-A", 5.0, tl), mk_p("HO-B", 2.0, tl)
    assert pv.reuse([c1, c2], tl, level=0.1).bits == pytest.approx(5.0)
    adm = pv.reuse([c1, c2, c3], tl, level=0.1)
    assert adm.bits == pytest.approx(7.0) and adm.level == pytest.approx(0.1 * 2 ** -7.0)
    child = ct.issue(claim="child", guarantee=g(), ell_test=ct.Lineage(), proposer_rows=ct.Lineage(), artifact={}, support=("a",), gram_context={}, kind=ct.CertificateKind.CARRIER, consumed=(c1, c3))
    hs = {p["holdout"]: p["bits"] for p in ct.peeks_of(child)}
    assert hs == {"HO-A": 3.0, "HO-B": 2.0}
    assert pv.reuse([child], tl, level=0.1).bits == pytest.approx(5.0)
    return "max within a holdout, sum across holdouts, inherited through consumption"


@lock("TY3", "R7: an unpriced test on transcript-seen rows is a VIOLATION at inflation 2^L with the repair q 2^-L, a bar that priced the bits passes, disjoint rows pass, unknown lineage returns UNKNOWN, and consumed certificates carry their peeks into the check")
def ty3():
    panel, ho, cert, d = _peeked_cert()
    L = ct.peeks_of(cert)[0]["bits"]
    tl = ho.tuning_lineage()
    rr = ct.RunRecord(name="tuned through asks, scored on the tuning rows", test_rows=tl, q=0.05, peeks=[ct.PeekDecl(rows=tl, bits=L, holdout=ho.id)])
    f = pv.lint(rr)
    r7 = [r for r in f.rows if r.check == "R7"][0]
    rep = f.repair("R7")
    assert r7.verdict == "VIOLATION" and r7.source == "RRSSTW20 Thm 3.2" and rep["repaired_level"] == pytest.approx(0.05 * 2 ** -L) and rep["inflation"] == pytest.approx(2 ** L)
    rr.priced_bits = L
    assert [r for r in pv.lint(rr).rows if r.check == "R7"][0].verdict == "PASS"
    rr2 = ct.RunRecord(name="other rows", test_rows=ho.exam_lineage(), q=0.05, peeks=[ct.PeekDecl(rows=tl, bits=L, holdout=ho.id)])
    assert [r for r in pv.lint(rr2).rows if r.check == "R7"][0].verdict == "PASS"
    rr3 = ct.RunRecord(name="unknown", test_rows=unknown_lineage(), q=0.05, peeks=[ct.PeekDecl(rows=tl, bits=L)])
    assert [r for r in pv.lint(rr3).rows if r.check == "R7"][0].verdict == "UNKNOWN"
    rr4 = ct.RunRecord(name="consumes the exam certificate on the tuning rows", test_rows=tl, consumed=[cert], q=0.05)
    f4 = pv.lint(rr4)
    r1 = [r for r in f4.rows if r.check == "R1"][0]
    assert r1.verdict == "PASS" and "transcript: priced under R7" in r1.detail and [r for r in f4.rows if r.check == "R7"][0].verdict == "VIOLATION"
    decl = pv.lint({"name": "declared", "test_rows": {"source_id": "s", "rows": [1, 2, 3]}, "proposal_rows": {"source_id": "p", "rows": [1]}, "q": 0.1, "peeks": [{"rows": {"source_id": "s", "rows": [2]}, "bits": 4}]})
    assert decl.repair("R7")["repaired_level"] == pytest.approx(0.1 / 16)
    return f"violation at {2 ** L:.3g}x, repair {rep['repaired_level']:.4g}; priced, disjoint, unknown, consumed and declared forms"


@lock("TY4", "rows_for_bits: n (z_L / z_0)^2 = n / s(L) at the one-sided bar, the identity at zero bits, registered in the rates register with its grade")
def ty4():
    for L in (0.0, 1.0, 8.3, 38.94, 100.0):
        assert ct.rows_for_bits(1000, L) == pytest.approx(1000 / tr.clean_share(0.05, L))
    assert ct.rows_for_bits(500, 0.0) == pytest.approx(500.0) and ct.rows_for_bits(1000, 1.0, delta=0.025) == pytest.approx(1000 * (tr.z_bar(0.025, 1) / tr.z_bar(0.025, 0)) ** 2)
    r = ct.REGISTER["rows_for_bits"]
    assert r.converts == "validity -> rows" and "one-sided" in r.grade and r.line().startswith("[rate] rows_for_bits")
    return f"1000 rows after 38.94 bits need {ct.rows_for_bits(1000, 38.94):.0f}"


@lock("TY5", "the performance kind and the ell_peek field are additive: certificates without peeks keep their pre-0.12 ids byte for byte; with peeks the field round-trips and sits inside the hashed content")
def ty5():
    start = pd.date_range(datetime(2026, 9, 3, tzinfo=UTC), periods=24, freq="h")
    f = pd.DataFrame({"start_utc": start, "end_utc": start + pd.Timedelta(hours=1), "x": np.random.default_rng(0).normal(size=24)})
    S = ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(0, 12)))
    P = ct.Lineage.of(ct.slice_from_frame(f, "dataset_1", range(12, 24)))
    gg = ct.Guarantee(kind=ct.GuaranteeKind.FDR, level=0.1, family={"rule": "BH", "q": 0.1, "m": 10})
    ts = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
    c0 = ct.issue(claim="neutral", guarantee=gg, ell_test=S, proposer_rows=P, artifact={"note": "neutral"}, support=("a1", "a2"), gram_context={}, kind=ct.CertificateKind.CARRIER, timestamp=ts)
    c1 = ct.issue(claim="child", guarantee=gg, ell_test=P, proposer_rows=ct.Lineage(), artifact={}, support=("a1",), gram_context={}, kind=ct.CertificateKind.EXISTENCE, consumed=(c0,), timestamp=ts)
    assert (c0.id, c1.id) == ("CD-2704BA639A1ADE30", "CD-C972B46EE91C69BA") and "ell_peek" not in to_dict(c0)
    cp = ct.issue(claim="neutral", guarantee=gg, ell_test=S, proposer_rows=P, artifact={"note": "neutral"}, support=("a1", "a2"), gram_context={}, kind=ct.CertificateKind.PERFORMANCE, timestamp=ts, peeks=[{"holdout": "HO-X", "bits": 2.5, "rows": P}])
    back = from_dict(json.loads(json.dumps(to_dict(cp))))
    assert back.id == cp.id and ct.peeks_of(back)[0]["bits"] == 2.5 and ct.peeks_of(back)[0]["rows"] == P and back.kind == ct.CertificateKind.PERFORMANCE
    moved = dict(to_dict(cp)); moved["ell_peek"] = [dict(moved["ell_peek"][0], bits=0.5)]
    assert content_id(moved) != cp.id
    return "pre-0.12 ids unchanged; ell_peek round-trips inside the hash"


@lock("TY6", "lint of a holdout: the exam route passes R1 and R7 (the transcript sits on other rows), the transcript route passes R7 priced at its own bits")
def ty6():
    panel, ho, cert, d = _peeked_cert()
    f = pv.lint(ho)
    assert {r.check: r.verdict for r in f.rows} == {"R1": "PASS", "R7": "PASS"} and "other rows" in [r for r in f.rows if r.check == "R7"][0].detail
    panel2, S2, H2, S2_lin, H2_lin, d2, hk = _ho_basic(seed=12, exam_share=0.0)
    X2, y2 = panel2.loc[S2, HFEATS], panel2.loc[S2, "y"]
    for cols in (["h0"], ["h1"]):
        hk.ask(_ols(X2, y2, cols), lineage=S2_lin)
    hk.open(_ols(X2, y2, ["h0", "h1"]), lineage=S2_lin)
    fk = pv.lint(hk)
    r7 = [r for r in fk.rows if r.check == "R7"][0]
    assert r7.verdict == "PASS" and r7.price.startswith("priced")
    return f"exam route R1/R7 PASS; transcript route R7 {r7.price}"


@lock("TY7", "PEEKED outside lint: grade on transcript-seen rows tests its cells at level 2^-L and charges the book the level itself; certify on transcript-seen rows refuses by name rather than run at the unpriced level")
def ty7():
    X, y = _grade_world(3)
    rows = ct.Lineage.of(ct.Slice(source_id=ct.source_fingerprint(X), dataset_tag="grade", rows=frozenset(range(len(X))), t0=datetime(1970, 1, 1, tzinfo=UTC), t1=datetime(1970, 1, 1, tzinfo=UTC)))
    fam = {"support_ordered": ["g0>=0.5"], "atoms_spec": [{"kind": "ge", "col": "g0", "thr": 0.5}], "directions": [1], "m": 1}
    cert = ct.issue(claim="peeked carrier", guarantee=ct.Guarantee(kind=ct.GuaranteeKind.FDR, level=0.1, family=fam), ell_test=ct.Lineage(), proposer_rows=ct.Lineage(), artifact={}, support=("g0>=0.5",), gram_context={}, kind=ct.CertificateKind.CARRIER, peeks=[{"holdout": "HO-GRADE", "bits": 4.0, "rows": rows}])
    led = ct.Ledger(tempfile.mkdtemp(prefix="ty7_"), budget=0.5)
    gr = pv.grade(cert, X, y, ledger=led, level=0.05)
    assert gr.admission_verdict == "PEEKED" and gr.level == pytest.approx(0.05 / 16) and led.statement().charged == pytest.approx(0.05)
    rep, _ = _certify_pair()
    s2 = rep.certificate.ell_test
    pk = ct.issue(claim="saw the certifying rows", guarantee=g(), ell_test=ct.Lineage(), proposer_rows=ct.Lineage(), artifact={}, support=("a",), gram_context={}, kind=ct.CertificateKind.CARRIER, peeks=[{"holdout": "HO-S2", "bits": 3.0, "rows": s2}])
    led2 = ct.Ledger(tempfile.mkdtemp(prefix="ty7_"), budget=0.5)
    led2.register(pk)
    with pytest.raises(ValueError, match=r"certify refused: the certifying rows were seen through counted transcripts \(CD-[0-9A-F]+ \(3.00 bits\)\)"):
        pv.certify(_STATE["X"], _STATE["y"], q=0.10, ledger=led2, profile="fast", seed=3, dataset_tag="synthetic")
    return f"grade PEEKED: cells at {gr.level:.4g}, book charged 0.05; certify refused"


@lock("TY8", "naive and aware clocks compare in reuse and lint; certify refuses when a consumed certificate saw its certifying rows through a transcript, as when the book's own did")
def ty8():
    naive = pd.DataFrame({"start_utc": pd.date_range("2026-03-01", periods=10, freq="h"), "x": np.arange(10.0)})
    naive["end_utc"] = naive["start_utc"] + pd.Timedelta(hours=1)
    aware = frame("dataset_2", 1)
    c = ct.issue(claim="aware", guarantee=g(), ell_test=ct.Lineage.of(ct.slice_from_frame(aware, "dataset_2", range(0, 4))), proposer_rows=ct.Lineage(), artifact={}, support=("a",), gram_context={}, kind=ct.CertificateKind.CARRIER)
    rows = ct.Lineage.of(ct.slice_from_frame(naive, "dataset_1", range(0, 10)))
    adm = pv.reuse([c], rows, level=0.1)
    assert adm.verdict in (ct.Verdict.FRESH, ct.Verdict.FRESH_DEP_FLAG)
    assert pv.lint(ct.RunRecord(name="mixed clocks", test_rows=rows, consumed=[c], q=0.1)).rows[0].verdict == "PASS"
    rep, _ = _certify_pair()
    s2 = rep.certificate.ell_test
    pk = ct.issue(claim="consumed, saw the certifying rows", guarantee=g(), ell_test=ct.Lineage(), proposer_rows=ct.Lineage(), artifact={}, support=("a",), gram_context={}, kind=ct.CertificateKind.CARRIER, peeks=[{"holdout": "HO-CONSUMED", "bits": 2.0, "rows": s2}])
    with pytest.raises(ValueError, match=r"certify refused: the certifying rows were seen through counted transcripts \(CD-[0-9A-F]+ \(2.00 bits\)\)"):
        pv.certify(_STATE["X"], _STATE["y"], q=0.10, ledger=tempfile.mkdtemp(prefix="ty8_"), profile="fast", seed=3, dataset_tag="synthetic", consumed=[pk])
    return f"mixed clocks compare ({adm.verdict.label}); consumed peek refused at certify"



# ------------------------------------------------------------------ TZ watch
def _watch_book(claim="risk", loss_range=(0.0, 16.0), loss="squared", budget=0.10, seed=1):
    rng = np.random.default_rng(seed)
    n = 400
    X = pd.DataFrame({"x": rng.normal(size=n)})
    if loss == "brier":
        y = (X["x"].to_numpy() + rng.normal(size=n) > 0).astype(float)
        model = lambda Z: 1.0 / (1.0 + np.exp(-1.6 * Z["x"].to_numpy()))
        base = 0.5
    else:
        y = 0.5 * X["x"].to_numpy() + rng.normal(size=n)
        model = lambda Z: 0.5 * Z["x"].to_numpy()
        base = 0.0
    start = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    led = ct.Ledger(tempfile.mkdtemp(prefix="tz_"), budget=budget)
    ho = pv.seal(X, y, led, lineage="self", loss=loss, claim=claim, baseline=base, baseline_lineage=ct.Lineage(), exam_share=0.0, loss_range=loss_range, tail="gaussian", clock=(start, start + pd.Timedelta(hours=1)))
    cert = ho.open(model, lineage=ct.Lineage())
    return led, cert, rng


def _fresh(rng, m, sd=1.0, t0="2026-02-01"):
    x = rng.normal(size=m)
    return 0.5 * x + sd * rng.normal(size=m), 0.5 * x, pd.date_range(t0, periods=m, freq="h", tz="UTC")


@lock("TZ1", "watch binds to a performance certificate registered and active in its book and charges delta before any row: a carrier, a certificate of another book, a voided one, a level above the remaining budget, an unbounded loss without a range, a range above the loss's minimum and a range other than the sealed one refuse; the statement carries the charge")
def tz1():
    led, cert, _ = _watch_book(loss_range=None, budget=0.2)
    with pytest.raises(ValueError, match="this certificate is carrier"):
        pv.watch(mk("carrier", ct.Lineage(), ct.Lineage()), led)
    with pytest.raises(ValueError, match="the squared loss is unbounded and the certificate declared no range"):
        pv.watch(cert, led)
    with pytest.raises(ValueError, match="above the squared loss's natural minimum 0"):
        pv.watch(cert, led, loss_range=(0.5, 16.0))
    other = ct.Ledger(tempfile.mkdtemp(prefix="tz1o_"), budget=0.5)
    with pytest.raises(ValueError, match="is not registered in"):
        pv.watch(cert, other, loss_range=(0.0, 16.0))
    with pytest.raises(ValueError, match=r"a watch at delta 0.2 charges the book, and .* has 0.1500 remaining"):
        pv.watch(cert, led, delta=0.2, loss_range=(0.0, 16.0))
    w = pv.watch(cert, led, delta=0.05, loss_range=(0.0, 16.0))
    st = led.statement()
    assert st.charged == pytest.approx(0.10) and st.watches[0]["id"] == w.id and st.watches[0]["level"] == 0.05 and "-- watches" in st.render()
    led2, cert2, _ = _watch_book()
    with pytest.raises(ValueError, match=r"sealed with the loss range \[0.0, 16.0\]"):
        pv.watch(cert2, led2, loss_range=(0.0, 9.0))
    led2.void(cert2.id, "test")
    with pytest.raises(ValueError, match="is voided"):
        pv.watch(cert2, led2)
    return f"{w.id}: charged 0.05 on top of the certificate; seven refusals by name"


@lock("TZ2", "the detector recomputed by brute force: every row's evidence equals the start-weighted mixture over bets and start rows plus the weight of the starts not yet begun, to 1e-12 in logs, under both weightings and across update boundaries; the start estimate is the unweighted CUSUM argmax")
def tz2():
    import math
    from provable.monitor import _advance, _start_of, GRID
    rng = np.random.default_rng(3)
    worst = 0.0
    for horizon in (None, 60):
        u = np.clip(rng.beta(2, 5, size=60) + np.r_[np.zeros(30), np.full(30, 0.25)], 0, 1)
        m = 0.3
        det = {"rows": 0, "logA": [None] * len(GRID), "logB": [None] * len(GRID), "start": [0] * len(GRID), "evidence": 0.0, "peak": 0.0}
        d1, _, _, e1 = _advance(det, u[:25], m, math.log(1e9), horizon)
        d2, _, _, e2 = _advance(d1, u[25:], m, math.log(1e9), horizon)
        ev = np.r_[e1, e2]
        for t in range(1, 61):
            tot, best, bs = 0.0, -math.inf, None
            for c in GRID:
                for s0 in range(1, t + 1):
                    w = 1.0 / (s0 * (s0 + 1)) if horizon is None else 1.0 / horizon
                    prod = float(np.prod(1.0 + (c / m) * (u[s0 - 1:t] - m)))
                    tot += w * prod / len(GRID)
                    if math.log(prod) > best:
                        best, bs = math.log(prod), s0
            rest = 1.0 / (t + 1) if horizon is None else (horizon - t) / horizon
            worst = max(worst, abs(ev[t - 1] - math.log(tot + rest)))
            if t in (25, 60):
                assert _start_of(d1 if t == 25 else d2) == bs
    assert worst < 1e-12
    return f"max log-evidence error {worst:.1e} over 120 rows; start estimates match"


@lock("TZ3", "the false-alarm guarantee under the worst null: Bernoulli losses with mean exactly at the certified level, 400 streams of 3,000 rows: alarms at most delta plus three standard errors, anytime and with a horizon")
def tz3():
    import math
    from provable.monitor import _advance, GRID
    rng = np.random.default_rng(11)
    out = []
    for horizon in (None, 3000):
        alarms = 0
        for r in range(400):
            m = (0.05, 0.3, 0.6)[r % 3]
            u = (rng.random(3000) < m).astype(float)
            det = {"rows": 0, "logA": [None] * len(GRID), "logB": [None] * len(GRID), "start": [0] * len(GRID), "evidence": 0.0, "peak": 0.0}
            alarms += _advance(det, u, m, math.log(20.0), horizon)[2] is not None
        assert alarms <= 400 * 0.05 + 3 * math.sqrt(400 * 0.05 * 0.95)
        out.append(alarms)
    return f"false alarms {out[0]} and {out[1]} of 400 at delta 0.05 (anytime, horizon 3000)"


@lock("TZ4", "an alarm after a shift: rows before the shift raise none, the alarm row follows the shift and the start estimate falls within 60 rows of it; rows after the alarm are stored but not monitored, the next update refuses, close keeps the charge")
def tz4():
    led, cert, rng = _watch_book()
    w = pv.watch(cert, led)
    y0, p0, t0 = _fresh(rng, 1000)
    s0 = w.update(y0, p0, times=t0)
    assert s0.status == "watching" and s0.rows == 1000 and s0.alarm is None
    y1, p1, t1 = _fresh(rng, 2000, sd=1.6, t0="2026-03-15")
    s1 = w.update(y1, p1, times=t1)
    assert s1.status == "alarm" and 1000 < s1.alarm["row"] < 3000 and abs(s1.alarm["start"] - 1001) <= 60
    assert s1.received == 3000 and s1.rows == s1.alarm["row"] and len(s1.path) == s1.rows - 1000
    with pytest.raises(ValueError, match="raised its alarm at row"):
        w.update(*_fresh(rng, 10, t0="2026-08-01")[:2], times=_fresh(rng, 10, t0="2026-08-01")[2])
    c = w.close("model retired")
    assert c.status == "alarm" and led.statement().charged == pytest.approx(0.10)
    return f"shift at row 1001, alarm at row {s1.alarm['row']}, start estimate {s1.alarm['start']}"


@lock("TZ5", "the clock runs forward: rows dated inside the certified slice refuse, times going backwards within or across updates refuse, times given at one update and not the next refuse; a horizon ends the watch at T rows and stores the rest unmonitored")
def tz5():
    led, cert, rng = _watch_book(budget=0.2)
    w = pv.watch(cert, led)
    y, p, _ = _fresh(rng, 20)
    with pytest.raises(ValueError, match="precedes the end of the certified rows"):
        w.update(y, p, times=pd.date_range("2026-01-10", periods=20, freq="h", tz="UTC"))
    with pytest.raises(ValueError, match="the times go backwards"):
        w.update(y, p, times=pd.date_range("2026-02-01", periods=20, freq="h", tz="UTC")[::-1])
    w.update(y, p, times=pd.date_range("2026-02-02", periods=20, freq="h", tz="UTC"))
    with pytest.raises(ValueError, match="precedes the last watched row"):
        w.update(y, p, times=pd.date_range("2026-02-01", periods=20, freq="h", tz="UTC"))
    with pytest.raises(ValueError, match="times come with every update or with none"):
        w.update(y, p)
    w2 = pv.watch(cert, led, horizon=50)
    s = w2.update(*_fresh(rng, 80)[:2])
    assert s.status == "ended" and s.rows == 50 and s.received == 80
    with pytest.raises(ValueError, match="reached its horizon of 50 rows"):
        w2.update(y, p)
    return "four clock refusals; horizon 50 ended the watch after 50 of 80 rows"


@lock("TZ6", "a gain claim is watched on the paired gain: the baseline is required and finite, a risk claim refuses one, and the alarm fires when the live gain falls below the certified lower bound")
def tz6():
    led, cert, rng = _watch_book(claim="gain")
    w = pv.watch(cert, led)
    y, p, t = _fresh(rng, 500)
    with pytest.raises(ValueError, match="pass baseline="):
        w.update(y, p, times=t)
    with pytest.raises(ValueError, match="must be finite, one per row"):
        w.update(y, p, times=t, baseline=np.r_[np.zeros(499), np.nan])
    s = w.update(y, p, times=t, baseline=np.zeros(500))
    assert s.status == "watching" and s.mean > cert.artifact["bound"]
    x = rng.normal(size=3000)
    yb = rng.normal(size=3000)
    s2 = w.update(yb, 0.5 * x, times=pd.date_range("2026-04-01", periods=3000, freq="h", tz="UTC"), baseline=np.zeros(3000))
    assert s2.status == "alarm"
    led2, cert2, rng2 = _watch_book()
    with pytest.raises(ValueError, match="it takes no baseline"):
        pv.watch(cert2, led2).update(y, p, times=t, baseline=np.zeros(500))
    return f"gain bound {cert.artifact['bound']:.3f}; live gain {s.mean:.3f} watched, alarm at row {s2.alarm['row']} once the signal left"


@lock("TZ7", "losses on the watch's range: a non-finite prediction scores the top of the range, losses above the range are clipped and the mean reported is the clipped one, a binary loss refuses a non-binary outcome")
def tz7():
    led, cert, rng = _watch_book(loss_range=(0.0, 4.0))
    w = pv.watch(cert, led)
    y = np.array([0.0, 0.0, 0.0, 0.0])
    p = np.array([1.0, 3.0, np.nan, 0.0])
    s = w.update(y, p, times=pd.date_range("2026-02-01", periods=4, freq="h", tz="UTC"))
    assert s.mean == pytest.approx((1.0 + 4.0 + 4.0 + 0.0) / 4)
    led2, cert2, _ = _watch_book(loss="brier", loss_range=None)
    w2 = pv.watch(cert2, led2)
    with pytest.raises(ValueError, match="takes a binary target"):
        w2.update(np.array([0.0, 0.5]), np.array([0.2, 0.3]), times=pd.date_range("2026-02-01", periods=2, freq="h", tz="UTC"))
    return f"mean {s.mean:.2f} with one loss clipped at 4 and one non-finite prediction scored at 4"


@lock("TZ8", "the audit replays the watch: PASS on an untouched book; an edited row file, an edited logged state and an edited last op (seen by the head) each FAIL")
def tz8():
    import json as _json
    led, cert, rng = _watch_book()
    w = pv.watch(cert, led)
    for k in range(3):
        y, p, t = _fresh(rng, 300, t0=f"2026-0{k + 2}-01")
        w.update(y, p, times=t)
    a = w.audit()
    assert a.ok and len(a.rows) == 6
    path = os.path.join(w.root, "2", "y.npy")
    keep = open(path, "rb").read()
    arr = np.load(path)
    arr[0] += 1.0
    np.save(path, arr)
    assert not w.audit().ok
    open(path, "wb").write(keep)
    assert w.audit().ok
    lp = os.path.join(led.path, "ledger.jsonl")
    lines = open(lp).read().splitlines()
    last = _json.loads(lines[-1])
    last["state"]["sum"] += 1.0
    open(lp, "w").write("\n".join(lines[:-1] + [_json.dumps(last, sort_keys=True, separators=(",", ":"))]) + "\n")
    led.refresh(force=True)
    a2 = w.audit()
    assert not a2.ok and any(r[0] == "head" and r[1] == "FAIL" for r in a2.rows)
    with pytest.raises(ValueError, match="does not match its head"):
        w.update(*_fresh(rng, 5, t0="2026-09-01")[:2], times=pd.date_range("2026-09-01", periods=5, freq="h", tz="UTC"))
    return f"{a.rows[-1][2]}; edited rows and an edited last op fail the audit"



# ------------------------------------------------------------------ TI route
def _route_rows(seed=0, M=48, per=168, regime=False):
    rng = np.random.default_rng(seed)
    blocks = np.repeat(np.arange(M), per)
    if regime:
        mean = np.repeat(rng.choice([-0.09, 0.09], size=M), per)
    else:
        mean = np.full(M * per, 0.16)
    return mean + rng.normal(scale=0.7, size=M * per), blocks


@lock("TI1", "a choice whose effect is the same in every block routes to VAULT and one whose effect changes sign with the regime routes to FORWARD, each with its reason; the threshold arithmetic reproduced: memory bound tau_ucb sqrt(2 ln r), bar z_0 sd(d) / sqrt(h)")
def ti1():
    d, b = _route_rows()
    ok = pv.route(d, b, h=500, r=2)
    assert ok.verdict == "VAULT" and "within one clean bar" in ok.reason
    assert ok.memory_bound == pytest.approx(ok.tau_ucb * np.sqrt(2 * np.log(2))) and ok.bar == pytest.approx(norm.isf(0.05) * d.std(ddof=1) / np.sqrt(500))
    d2, b2 = _route_rows(regime=True)
    no = pv.route(d2, b2, h=500, r=2)
    assert no.verdict == "FORWARD" and no.memory_bound > no.bar and no.q_p < 1e-6
    one = pv.route(d2, b2, h=500, r=1)
    assert one.memory_bound == 0.0 and one.verdict == ("FORWARD" if one.fluctuation > one.critical else "VAULT")
    return f"stable: {ok.memory_bound:.4f} <= {ok.bar:.4f}, VAULT; regime: {no.memory_bound:.4f} > {no.bar:.4f}, FORWARD"


@lock("TI2", "the Q-profile bound covers the spread: normal block effects with known errors, 2,000 replications at tau 0.1 and 12 blocks, coverage at least 1 - delta - 3 standard errors; the bound is zero when Q falls below the delta quantile and grows with the spread")
def ti2():
    from provable.transport import tau_upper
    rng = np.random.default_rng(7)
    s = np.full(12, 0.08)
    cov = sum(tau_upper(rng.normal(0.0, 0.1, 12) + rng.normal(0.0, s), s, 0.05) >= 0.1 for _ in range(2000)) / 2000
    assert cov >= 0.95 - 3 * np.sqrt(0.95 * 0.05 / 2000)
    assert tau_upper(np.zeros(12), s, 0.05) == 0.0
    assert tau_upper(np.linspace(-0.3, 0.3, 12), s, 0.05) > tau_upper(np.linspace(-0.1, 0.1, 12), s, 0.05) > 0
    return f"coverage {cov:.4f} at nominal 0.95"


@lock("TI3", "the fluctuation statistic: under a constant effect with independent rows, 600 runs reject at most delta plus three standard errors; a mid-sample shift of the effect is rejected; the critical value is the seeded Brownian-bridge quantile and falls as the window widens")
def ti3():
    from provable.transport import fluctuation_critical
    rng = np.random.default_rng(9)
    b = np.repeat(np.arange(12), 100)
    rej = sum(pv.route(0.1 + rng.normal(size=1200), b, h=300).fluctuation > fluctuation_critical(0.3, 0.05) for _ in range(600))
    assert rej <= 600 * 0.05 + 3 * np.sqrt(600 * 0.05 * 0.95)
    shift = pv.route(np.r_[np.full(600, -0.3), np.full(600, 0.3)] + rng.normal(size=1200), b, h=300)
    assert shift.verdict == "FORWARD" and "moves over time" in shift.reason
    c = [fluctuation_critical(w, 0.05) for w in (0.1, 0.3, 0.5)]
    assert c[0] > c[1] > c[2] > 1.5
    return f"{rej} of 600 rejected under a constant effect; critical values {c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}"


@lock("TI4", "refusals by name: fewer than three blocks, a block in two runs, a one-row block, a constant block, non-finite differences, a length mismatch, r below 1 and no h; holdout= reads h from its tuning slice and refuses differences computed on rows in its custody")
def ti4():
    d, b = _route_rows(M=6, per=10)
    for kw, msg in ((dict(d=d[:20], blocks=b[:20], h=10), "2 blocks"), (dict(d=d, blocks=np.r_[b[:5], b[10:20], b[5:10], b[20:]], h=10), "more than one run"),
                    (dict(d=d[:41], blocks=np.r_[b[:40], 9], h=10), "has 1 row"), (dict(d=np.r_[np.ones(10), d[10:]], blocks=b, h=10), "are constant"),
                    (dict(d=np.r_[np.nan, d[1:]], blocks=b, h=10), "must be finite"), (dict(d=d, blocks=b[:-1], h=10), "block labels for"),
                    (dict(d=d, blocks=b, h=10, r=0), "a whole number at least 1"), (dict(d=d, blocks=b), "pass h")):
        with pytest.raises(ValueError, match=msg):
            pv.route(kw.pop("d"), kw.pop("blocks"), **kw)
    rng = np.random.default_rng(2)
    panel = pd.DataFrame({"x": rng.normal(size=600)})
    panel["y"] = panel["x"] + rng.normal(size=600)
    panel["start_utc"] = pd.date_range("2026-01-01", periods=600, freq="h", tz="UTC")
    panel["end_utc"] = panel["start_utc"] + pd.Timedelta(hours=1)
    tr, hl = panel.iloc[:400], panel.iloc[400:]
    S, H = (ct.Lineage.of(ct.slice_from_frame(panel, "ti4", rows=f.index)) for f in (tr, hl))
    ho = pv.seal(hl[["x"]], hl["y"], tempfile.mkdtemp(prefix="ti4_"), lineage=H, baseline=0.0, baseline_lineage=ct.Lineage())
    r = pv.route(rng.normal(size=400), np.repeat(np.arange(4), 100), holdout=ho, lineage=S)
    assert r.h == ho.commit["n_tune"] == 160
    with pytest.raises(ValueError, match="meet rows in the custody of"):
        pv.route(rng.normal(size=400), np.repeat(np.arange(4), 100), holdout=ho, lineage=H)
    return f"eight refusals by name; h read from the tuning slice ({r.h} rows); custody rows refused"


@lock("TI5", "the record: with ledger= the verdict is appended as a route op carrying hashes of d and the blocks, the statement lists it, and the account is unchanged")
def ti5():
    from provable.holdout import _sha, _npy_bytes, _canon
    led = ct.Ledger(tempfile.mkdtemp(prefix="ti5_"), budget=0.10)
    d, b = _route_rows()
    r = pv.route(d, b, h=500, ledger=led, label="x0 in or out")
    ops = [o for o in led._ops if o["op"] == "route"]
    assert len(ops) == 1 and r.record.startswith(ops[0]["id"]) and ops[0]["verdict"] == r.verdict
    assert ops[0]["d_sha"] == _sha(_npy_bytes(np.asarray(d, float))) and ops[0]["blocks_sha"] == _sha(_canon([str(x) for x in b.tolist()]))
    st = led.statement()
    assert st.charged == 0.0 and st.routes[0]["id"] == ops[0]["id"] and "-- routes" in st.render()
    return f"{ops[0]['id']} recorded, {r.verdict}; account unchanged"


@pytest.mark.parametrize("lid,target,fn", LOCKS, ids=[l[0] for l in LOCKS])
def test_lock(lid, target, fn):
    fn()


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
