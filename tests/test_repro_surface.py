"""Repro-surface gate.

Freezes the provable surface consumed by the published paper-artifact
scripts (the two archived artifact bundles, md5 6e2b9cb80c3aed630d8ce9f
d1cf66309 and 0101ad64fee67541110465cce455bd37). Their live contract is
`from provable import CertifiedDiscovery`, the (seed, verbose, **expert)
constructor, run(X, y), and the post-run attributes below. Attributes
the scripts consume from paper-era sieve variants that never shipped in
0.9.0 (family_panel_, _core_families, cert_cols_)
reproduce through the archive's pinned environment, not through this
gate; they are listed so a future release that adds them knows they are
spoken for. Any change that drops a LIVE symbol fails here.
"""
import inspect

import numpy as np
import pandas as pd

from provable import CertifiedDiscovery

LIVE_ATTRS = ["P_", "Z", "y", "idx_s1", "idx_s2", "idx_train", "idx_test", "core_", "certified_", "cert_sel_", "core_t_", "diag_", "perf_", "refit_scope", "atom_names_", "elapsed_", "claims_", "report_", "khat_", "cert_pred_all_"]
ARCHIVE_ONLY = ["family_panel_", "_core_families", "cert_cols_"]


def _mini_cd():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(260, 5)), columns=[f"x{i}" for i in range(5)])
    y = rng.normal(size=260) + 1.3 * (X["x0"] > 0).astype(float)
    cd = CertifiedDiscovery(seed=1, verbose=False, rounds=1, stability_reps=0, n_harvest=10)
    cd.run(X, y)
    return cd


def test_constructor_and_run_signature():
    sig = list(inspect.signature(CertifiedDiscovery.__init__).parameters)
    assert "seed" in sig and "verbose" in sig
    assert list(inspect.signature(CertifiedDiscovery.run).parameters)[:3] == ["self", "X", "y"]


def test_live_surface_present_after_run():
    cd = _mini_cd()
    missing = [a for a in LIVE_ATTRS if not hasattr(cd, a)]
    assert not missing, f"live repro surface broken: {missing}"
    assert len(cd.idx_s1) and len(cd.idx_s2) and cd.P_ == cd.Z.shape[1]
    assert isinstance(cd.diag_, dict) and isinstance(cd.perf_, dict)


def test_archive_only_list_is_honest():
    cd = _mini_cd()
    shipped = [a for a in ARCHIVE_ONLY if hasattr(cd, a)]
    assert not shipped, f"attributes moved from archive-only to live; update the gate lists: {shipped}"
