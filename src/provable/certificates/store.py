"""The Ledger: the certificate book.

A ledger is a directory holding an append-only ledger.jsonl (one
operation per line: register, void, reserve, note, grade, refine, the
holdout operations of provable.holdout, hash-chained per holdout, the
watch operations of provable.monitor, hash-chained per watch, and the
routes of provable.transport)
and a budget_state.json (the validity account); .lock is the file the
book lock takes, and ledger.jsonl.torn keeps the tail of an append that
never completed. A sealed holdout keeps its rows in custody under
holdouts/<id>/ in the same directory. The book's realized guarantee under the additive
rule is the sum of active certificate levels (the proved additive budget
across certificate sets); EXACT artifact certificates and UNKNOWN
LINEAGE items carry no error budget and are excluded from the account.

The registered charge (the online layer) is reserve(): a level debited
against the account before outcomes are consulted; a later registration
may consume the reservation.

The batch step-up combination rule is wired but gated: activating it
requires the exact statement of the certificate-algebra paper's batch
theorem (Thm 6.5), which is not transcribed into this build; the method
refuses with that message rather than guess theorem content. The
additive rule is the active default meanwhile.
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from tabulate import tabulate

from .admission import Admission, admit as _admit
from .batch import SOURCE as BATCH_SOURCE, BatchSchedule
from .lineage import Lineage
from .records import Certificate, CertificateKind, Guarantee, GuaranteeKind, content_id, from_dict, issue, refines, to_dict


@dataclass
class Statement:
    """The ledger's account statement."""
    path: str
    budget: float
    charged: float
    reserved: float
    remaining: float
    n_active: int
    n_void: int
    items: list
    unknowns: list
    voids: list
    open_reservations: list
    notes: list = field(default_factory=list)
    grades: list = field(default_factory=list)
    refinements: list = field(default_factory=list)
    holdouts: list = field(default_factory=list)
    watches: list = field(default_factory=list)
    routes: list = field(default_factory=list)

    def render(self) -> str:
        head = [["budget (target)", f"{self.budget:.4f}"], ["charged (additive, active book)", f"{self.charged:.4f}"], ["reserved (registered charges open)", f"{self.reserved:.4f}"], ["remaining", f"{self.remaining:.4f}"], ["active certificates", self.n_active], ["voided", self.n_void], ["unknown-lineage items (not budget-bearing)", len(self.unknowns)]]
        out = ["-- ledger statement: " + self.path + " --", tabulate(head, tablefmt="psql")]
        if self.items:
            rows = [[i["id"], i["kind"], i["guarantee"], f"{i['level']:.4f}" if i["level"] is not None else "", i["claim"][:48]] for i in self.items]
            out += ["-- active certificates --", tabulate(rows, headers=["id", "kind", "guarantee", "level", "claim"], tablefmt="psql")]
        if self.unknowns:
            rows = [[i["id"], i["kind"], i["claim"][:48], "UNKNOWN LINEAGE"] for i in self.unknowns]
            out += ["-- unknown lineage (predates instrumentation; excluded from the account) --", tabulate(rows, headers=["id", "kind", "claim", "state"], tablefmt="psql")]
        if self.voids:
            rows = [[v["id"], v["cause"][:60]] for v in self.voids]
            out += ["-- voided --", tabulate(rows, headers=["id", "cause"], tablefmt="psql")]
        if self.grades:
            rows = [[g["certificate"], f"{g['level']:.4f}", f"{g['bar']:.4g}" if g.get("bar") is not None else "", g.get("k"), g.get("confirmed"), g.get("reversed"), g.get("not_confirmed"), g.get("reservation") or ""] for g in self.grades]
            out += ["-- grades (budget-bearing; each graded level charges the account unless it consumed a reservation) --", tabulate(rows, headers=["certificate", "level", "bar", "k", "confirmed", "reversed", "not conf", "reservation"], tablefmt="psql")]
        if self.refinements:
            rows = [[r["parent"], r["id"], r["member"], f"{r['p']:.3g}", f"{r['bar']:.4g}", r["multiplicity"]] for r in self.refinements]
            out += ["-- refinements (licensed carriers under an existence certificate; the charge rides the parent's level) --", tabulate(rows, headers=["parent", "carrier", "member", "p", "bar", "multiplicity"], tablefmt="psql")]
        if self.holdouts:
            rows = [[h["id"], f"{h['mode']}/{h['route']}", f"{h['n_tune']}/{h['n_exam']}", f"{h['asks']} ({h['accepts']})" if h["mode"] == "comparative" else f"{h['checks']} ({h['failures']} failed)", f"{h['bits']:.2f}", f"{h['bar']:.3f}" if h["mode"] == "comparative" else "-", f"{h['clean_share']:.3f}" if h["mode"] == "comparative" else "-", h["tuning"], h["exam"]] for h in self.holdouts]
            out += ["-- holdouts (bits = codelength of each holdout's own log; the transcript bar z_L is taken at delta 2^-bits; s(L) = (z_0/z_L)^2 is the clean share) --", tabulate(rows, headers=["holdout", "mode/route", "tune/exam rows", "asks (accepts) | checks", "bits", "z_L", "s(L)", "tuning", "exam"], tablefmt="psql")]
        if self.watches:
            rows = [[w["id"], w["certificate"], f"{w['level']:.4f}", f"{w['claim']} {'<=' if w['claim'] == 'risk' else '>='} {w['certified']:.4g}", w["rows"], "" if w["mean"] is None else f"{w['mean']:.4g}", f"{w['evidence']:.4g}" if w["evidence"] < 1e6 else f"{w['evidence']:.3e}", w["status"] + ("" if w["alarm"] is None else f" at row {w['alarm']}")] for w in self.watches]
            out += ["-- watches (budget-bearing: each watch charges its level when it opens; evidence against the certified level, alarm at 1/level) --", tabulate(rows, headers=["watch", "certificate", "level", "certified", "rows", "mean", "evidence", "status"], tablefmt="psql")]
        if self.routes:
            rows = [[o["id"], o.get("label", "")[:28], o["verdict"], f"{o['tau_ucb']:.4g}", o["r"], f"{o['memory_bound']:.4g}", f"{o['bar']:.4g}", f"{o['fluctuation']:.3f} / {o['critical']:.3f}"] for o in self.routes]
            out += ["-- routes (transport status of choices, read on training rows; not budget-bearing) --", tabulate(rows, headers=["route", "label", "verdict", "tau_c UCB", "r", "memory bias bound", "clean bar of noise", "fluctuation / critical"], tablefmt="psql")]
        if self.notes:
            out += ["-- notes --"] + [f"  {t}" for t in self.notes]
        return "\n".join(out)

    def __str__(self):
        return self.render()


class Ledger:
    LEDGER_FILE = "ledger.jsonl"
    STATE_FILE = "budget_state.json"

    def __init__(self, path: str, budget: Optional[float] = None):
        self.path = str(path)
        os.makedirs(self.path, exist_ok=True)
        self._state_path = os.path.join(self.path, self.STATE_FILE)
        self._ledger_path = os.path.join(self.path, self.LEDGER_FILE)
        if os.path.exists(self._state_path):
            with open(self._state_path) as fh:
                st = json.load(fh)
            self.budget = float(st.get("budget", 0.10)) if budget is None else float(budget)
            if budget is not None and float(budget) != float(st.get("budget", budget)):
                st["budget"] = float(budget)
                self._write_state(st)
        else:
            self.budget = 0.10 if budget is None else float(budget)
            self._write_state({"budget": self.budget, "rule": "additive"})
        self._ops_cache = []
        self._sig = None
        self._lock_fh = None
        self._lock_depth = 0
        self._rlock = threading.RLock()
        self._read_lock = threading.Lock()
        self.refresh()

    def _signature(self):
        try:
            st = os.stat(self._ledger_path)
        except FileNotFoundError:
            return (0, 0, 0)
        return (st.st_size, st.st_mtime_ns, st.st_ino)

    def refresh(self, force: bool = False) -> list:
        """Re-read the operation log when the file changed on disk (size,
        modification time or inode), so every handle on a book reads the
        same operations; force re-reads unconditionally. One reader at a
        time per handle, so a slower read never replaces a newer one; a
        line still being written (no newline yet) is left out until it is
        whole."""
        with self._read_lock:
            sig = self._signature()
            if not force and sig == self._sig:
                return self._ops_cache
            ops, whole = [], True
            if sig[0]:
                with open(self._ledger_path) as fh:
                    for line in fh:
                        if not line.endswith("\n"):
                            whole = False
                            break
                        line = line.strip()
                        if line:
                            ops.append(json.loads(line))
            self._ops_cache = ops
            self._sig = sig if whole else None
            return ops

    @property
    def _ops(self) -> list:
        return self.refresh()

    @contextmanager
    def locked(self):
        """Exclusive lock on the book: a thread lock for this handle, then an
        advisory POSIX flock on .lock in the book directory (no flock where
        the platform has none), reentrant for the thread that holds it.
        Every append takes it; the holdout takes it around read, check and
        append, never around a model's own code."""
        with self._rlock:
            if self._lock_depth == 0:
                try:
                    import fcntl
                    self._lock_fh = open(os.path.join(self.path, ".lock"), "a")
                    fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_EX)
                except ImportError:
                    self._lock_fh = None
            self._lock_depth += 1
            try:
                yield self
            finally:
                self._lock_depth -= 1
                if self._lock_depth == 0 and self._lock_fh is not None:
                    import fcntl
                    fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_UN)
                    self._lock_fh.close()
                    self._lock_fh = None

    # ------------------------------------------------------------ plumbing
    def _write_state(self, st: dict) -> None:
        with open(self._state_path, "w") as fh:
            json.dump(st, fh, indent=1, sort_keys=True)

    def _append(self, op: dict) -> None:
        """Append one operation under the book lock. A last line without its
        newline is the tail of an append that never completed (a writer
        that died mid-write; readers already leave it out): it is moved to
        ledger.jsonl.torn before the new line is written, so that the new
        line is never glued onto it."""
        line = json.dumps(op, sort_keys=True, separators=(",", ":")) + "\n"
        with self.locked():
            if os.path.exists(self._ledger_path) and os.path.getsize(self._ledger_path):
                with open(self._ledger_path, "rb+") as fh:
                    fh.seek(-1, os.SEEK_END)
                    if fh.read(1) != b"\n":
                        fh.seek(0)
                        data = fh.read()
                        cut = data.rfind(b"\n") + 1
                        with open(self._ledger_path + ".torn", "ab") as torn:
                            torn.write(data[cut:] + b"\n")
                        fh.truncate(cut)
            with open(self._ledger_path, "a") as fh:
                fh.write(line)
            self.refresh(force=True)

    # ------------------------------------------------------------ writes
    def register(self, cert: Certificate, reservation: Optional[str] = None) -> str:
        """Append a certificate to the book. In a book holding a sealed
        holdout (a book without holdouts is unchanged): an id the holdout's
        log names is taken once, by the performance certificate whose
        content hashes to it, which is the holdout's own registration; any
        other certificate whose test rows meet rows still in the holdout's
        custody refuses by name, since sealed rows are reached through the
        holdout until they are released."""
        d = to_dict(cert)
        cid = content_id(d)
        with self.locked():
            if self.holdout_ids():
                from ..holdout import register_guard
                register_guard(self, cert, cid)
            self._append({"op": "register", "id": cid, "record": d, "reservation": reservation})
        return cid

    def void(self, cert_id: str, cause: str) -> None:
        cert_id, cause = str(cert_id), str(cause)
        with self.locked():
            if cert_id not in {o["id"] for o in self._ops if o["op"] == "register"}:
                raise KeyError(f"no registered certificate {cert_id}")
            self._append({"op": "void", "id": cert_id, "cause": str(cause)})

    def note(self, text: str) -> None:
        """Append a free-text provenance note to the book. Notes are
        first-class append-only ops: they never touch the account and
        render in the statement."""
        self._append({"op": "note", "text": str(text)})

    def reserve(self, level: float, note: str = "", vocabulary=None) -> str:
        """The registered charge. With a vocabulary (a finite set of
        member/row-set pair ids fixed in advance), this is the
        registered-vocabulary charge of the certificate-algebra paper
        (Prop 6.3): claims issued only at p <= level/|V| carry familywise
        validity at the reserved level under arbitrary reuse, order, and
        lineage; it is the one guarantee that survives reuse of the test
        rows. Without a vocabulary it is a plain budget hold
        (bookkeeping, not Prop 6.3)."""
        level, note = float(level), str(note)
        vocab = None if vocabulary is None else sorted(str(v) for v in vocabulary)
        with self.locked():
            st = self.statement()
            if level <= 0:
                raise ValueError("reserved level must be positive")
            if st.remaining < level:
                raise ValueError(f"reservation {level:.4f} exceeds remaining budget {st.remaining:.4f}")
            rid = f"RES-{len([o for o in self._ops if o['op'] == 'reserve']) + 1:04d}"
            if vocab is not None and len(vocab) == 0:
                raise ValueError("a registered vocabulary must be nonempty")
            self._append({"op": "reserve", "id": rid, "level": level, "note": note, "vocabulary": vocab})
        return rid

    def claim_registered(self, reservation: str, member: str, p: float) -> dict:
        """Check a claim against a registered-vocabulary charge (Prop
        6.3): valid iff the member is one of the pre-registered pairs and
        its p-value clears the per-pair bar level/|V|. Lineage-free by
        the proposition: arbitrary reuse, order, and stopping are
        covered; the price is the fixed multiplicity |V|."""
        res = [o for o in self._ops if o["op"] == "reserve" and o["id"] == reservation]
        if not res:
            raise KeyError(f"no reservation {reservation}")
        r = res[0]
        vocab = r.get("vocabulary")
        if not vocab:
            raise ValueError(f"reservation {reservation} carries no registered vocabulary; a plain budget hold is not the Prop 6.3 charge")
        bar = float(r["level"]) / len(vocab)
        member = str(member)
        inside = member in vocab
        valid = bool(inside and float(p) <= bar)
        reason = "valid at the per-pair bar" if valid else ("member not in the registered vocabulary" if not inside else f"p {float(p):.3g} above the bar {bar:.3g}")
        return {"reservation": reservation, "member": member, "p": float(p), "bar": bar, "vocabulary_size": len(vocab), "valid": valid, "lineage_free": True, "source": "M2 Prop 6.3", "reason": reason}

    # ------------------------------------------------------------ online budget
    def refine(self, parent, member: str, p: float, direction=None, claim=None, multiplicity: str = "support") -> str:
        """The licensed refinement op (M2 Thm 7.2; Cor 7.6; Def 7.8):
        issue a carrier for one member of an active existence
        certificate's vocabulary V, at the per-member bar level / |V|
        (multiplicity="union" uses the book's certified-support union
        instead). Licensed on the certifying rows, so no fresh
        admission runs; charges nothing further, the parent's level
        already sits on the account and the carrier rides it. The
        result satisfies the stored refines() predicate, so a declared
        (parent, child) pair lints R6 PASS."""
        pid = parent if isinstance(parent, str) else content_id(to_dict(parent))
        member, p = str(member), float(p)
        direction = None if direction is None else int(direction)
        claim = None if claim is None else str(claim)
        with self.locked():
            dead = self._voided_ids()
            rec = {o["id"]: o for o in self._recorded()}
            if pid not in rec or pid in dead:
                raise ValueError(f"refine: no active certificate {pid} in {self.path}")
            par = from_dict(rec[pid]["record"])
            if par.kind != CertificateKind.EXISTENCE:
                raise ValueError("refine: parent must be an existence certificate (R6; M2 Def 7.8)")
            if member not in par.support:
                raise ValueError(f"refine: member {member!r} outside the licensed vocabulary V (|V|={len(par.support)})")
            V = par.support if multiplicity == "support" else self.certified_support_union()
            mult = len(V)
            if mult == 0:
                raise ValueError("refine: empty multiplicity set")
            bar = float(par.guarantee.level) / mult
            if p > bar:
                raise ValueError(f"refine refused: p {p:.4g} above the licensed bar {bar:.4g} = level {par.guarantee.level:g} / multiplicity {mult} (M2 Def 7.8)")
            fam = {"multiplicity": mult, "member": member, "p": p, "parent": pid}
            if direction is not None:
                fam["direction"] = int(direction)
            child = issue(claim=claim or f"refined carrier: {member} at p={p:.3g} under {pid}", guarantee=Guarantee(kind=par.guarantee.kind, level=bar, family=fam), ell_test=par.ell_test, proposer_rows=par.ell_prop, artifact={"p": p, "parent": pid}, support=(member,), gram_context=dict(par.gram_context), kind=CertificateKind.CARRIER, consumed=(par,))
            assert refines(child, par, ledger=self)
            cid = content_id(to_dict(child))
            if self.holdout_ids():
                from ..holdout import custody_meets
                hits = custody_meets(self, child.ell_test)
                if hits:
                    raise ValueError(f"refine refused: the carrier's test rows meet rows in the custody of {hits[0][0]}: {hits[0][1]}; sealed rows are reached through the holdout until retire() releases them")
            self._append({"op": "refine", "id": cid, "parent": pid, "record": to_dict(child), "member": member, "p": p, "bar": bar, "multiplicity": mult})
        return cid

    def _batch_schedule(self) -> BatchSchedule:
        from .records import GuaranteeKind as _GK
        sched = BatchSchedule(q=self.budget)
        for c in self.certificates(kind="carrier"):
            fam = c.guarantee.family
            if c.uninstrumented() or "p" not in fam:
                continue
            sched.record(fam["p"], float(c.guarantee.level))
        return sched

    def next_level(self, n_next: int) -> float:
        """The batch step-up quote for the next run of family size
        n_next, per the transcribed Algorithm 1 at the account target."""
        return self._batch_schedule().next_level(int(n_next))

    def record_grade(self, entry: dict, reservation: Optional[str] = None) -> None:
        """Append a grade scorecard entry (budget-bearing: the graded
        level charges the account unless it consumes a reservation)."""
        op = {"op": "grade", "reservation": reservation}
        op.update(entry)
        self._append(op)

    def combine_batches(self) -> dict:
        """The online budget statement over the book's carrier sequence:
        per-batch levels used vs prescribed, the Definition-1 estimator,
        and the governing bound. A conforming book carries the union
        guarantee at the account target (Thm 6.5); a deviating book
        falls back to the additive bound, which always holds."""
        sched = self._batch_schedule()
        probe = BatchSchedule(q=self.budget)
        rows = []
        for b in sched.batches:
            rows.append({"n": b.n, "level_used": b.level, "level_prescribed": probe.next_level(b.n), "R": b.R, "R_plus": b.R_plus})
            probe.batches.append(b)
        conforms = sched.conforms()
        additive = float(sum(b.level for b in sched.batches))
        return {"target": self.budget, "batches": rows, "fdp_hat": sched.fdp_hat(), "conforms": conforms, "guarantee": (f"union FDP <= {self.budget:g} at every t ({BATCH_SOURCE}); requires per-batch fresh lineage (M2 Lem 6.4) and a monotone schedule" if conforms else f"levels deviate from the batch step-up schedule; governing bound is additive: union FDP <= {additive:.4f} (M2 Prop 6.1)"), "additive_level": additive}

    # ------------------------------------------------------------ reads
    def holdout_ops(self, holdout_id: Optional[str] = None) -> List[dict]:
        """The holdout operations of the book, in log order, optionally for
        one holdout."""
        return [o for o in self._ops if o.get("op") == "holdout" and (holdout_id is None or o.get("holdout") == holdout_id)]

    def holdout_ids(self) -> List[str]:
        return [o["holdout"] for o in self.holdout_ops() if o.get("action") == "seal"]

    def _holdout_rows(self) -> list:
        rows = []
        for hid in self.holdout_ids():
            ops = self.holdout_ops(hid)
            seal, last = ops[0], ops[-1]
            pr, cm, st = seal["protocol"], seal["commit"], last.get("state", {})
            rows.append({"id": hid, "mode": pr["mode"], "route": pr["route"], "claim": pr["claim"], "n_tune": cm["n_tune"], "n_exam": cm["n_exam"], "asks": st.get("asks", 0), "accepts": st.get("accepts", 0), "checks": st.get("checks", 0), "failures": st.get("failures", 0), "bits": float(st.get("bits", 0.0)), "bar": float(st.get("bar", 0.0)), "clean_share": float(st.get("clean_share", 1.0)), "tuning": st.get("tuning", ""), "exam": st.get("exam", ""), "certificates": list(st.get("certificates", []))})
        return rows

    def watch_ops(self, watch_id: Optional[str] = None) -> List[dict]:
        """The watch operations of the book, in log order, optionally for
        one watch."""
        return [o for o in self._ops if o.get("op") == "watch" and (watch_id is None or o.get("watch") == watch_id)]

    def watch_ids(self) -> List[str]:
        return [o["watch"] for o in self.watch_ops() if o.get("action") == "open"]

    def _watch_rows(self) -> list:
        import math
        rows = []
        for wid in self.watch_ids():
            ops = self.watch_ops(wid)
            o, st = ops[0], ops[-1].get("state", {})
            n = int(st.get("rows", 0))
            mean = None if n == 0 else float(st.get("sum", 0.0)) / n
            if mean is not None and o.get("claim") == "gain":
                mean = -mean
            rows.append({"id": wid, "certificate": o.get("certificate"), "claim": o.get("claim"), "level": float(o.get("delta", 0.0)), "certified": float(o.get("level", 0.0)), "rows": n, "mean": mean, "evidence": math.exp(min(float(st.get("evidence", 0.0)), 700.0)), "status": st.get("status", ""), "alarm": (st.get("alarm") or {}).get("row")})
        return rows

    def _registered(self) -> List[dict]:
        return [o for o in self._ops if o["op"] == "register"]

    def _recorded(self) -> List[dict]:
        return [o for o in self._ops if o["op"] in ("register", "refine")]

    def _voided_ids(self) -> dict:
        return {o["id"]: o["cause"] for o in self._ops if o["op"] == "void"}

    def active(self) -> List[Certificate]:
        dead = self._voided_ids()
        return [from_dict(o["record"]) for o in self._recorded() if o["id"] not in dead]

    def certificates(self, support: Optional[Iterable[str]] = None, kind: Optional[str] = None) -> List[Certificate]:
        out = self.active()
        if kind is not None:
            k = kind.value if isinstance(kind, CertificateKind) else str(kind)
            out = [c for c in out if (c.kind.value if isinstance(c.kind, CertificateKind) else str(c.kind)) == k]
        if support is not None:
            s = set(str(x) for x in support)
            out = [c for c in out if s & set(c.support)]
        return out

    def certified_support_union(self) -> frozenset:
        out = set()
        for c in self.active():
            if c.kind == CertificateKind.CARRIER:
                out |= set(c.support)
        return frozenset(out)

    def spent_rows(self) -> Lineage:
        lin = Lineage()
        for c in self.active():
            if not c.uninstrumented():
                lin = lin.union(c.full_lineage())
        return lin

    # ------------------------------------------------------------ admission
    def admit(self, certs: Optional[Iterable[Certificate]] = None, rows: Optional[Lineage] = None, level: Optional[float] = None, mode: str = "guidance", on_stale: str = "refuse") -> Admission:
        """Reuse admission against the book. certs defaults to the whole
        active book; rows are the consuming rows; level defaults to the
        account target and is returned as the pass level for the next
        test."""
        certs = list(certs) if certs is not None else self.active()
        rows = rows if rows is not None else Lineage()
        lv = self.budget if level is None else float(level)
        return _admit(certs, rows, level=lv, mode=mode, on_stale=on_stale)

    # ------------------------------------------------------------ statement
    def _budget_bearing(self, c: Certificate) -> bool:
        if c.uninstrumented():
            return False
        return c.guarantee.kind != GuaranteeKind.EXACT

    def statement(self) -> Statement:
        dead = self._voided_ids()
        grade_ops = [o for o in self._ops if o["op"] == "grade"]
        consumed_res = {o.get("reservation") for o in self._registered() if o.get("reservation")} | {o.get("reservation") for o in grade_ops if o.get("reservation")}
        items, unknowns = [], []
        charged = 0.0
        for o in self._registered():
            if o["id"] in dead:
                continue
            c = from_dict(o["record"])
            row = {"id": o["id"], "kind": c.kind.value if isinstance(c.kind, CertificateKind) else str(c.kind), "guarantee": c.guarantee.kind.value, "level": float(c.guarantee.level) if c.guarantee.kind != GuaranteeKind.EXACT else None, "claim": c.claim}
            if c.uninstrumented():
                unknowns.append(row)
            else:
                items.append(row)
                if self._budget_bearing(c):
                    charged += float(c.guarantee.level)
        open_res = [{"id": o["id"], "level": o["level"], "note": o.get("note", "")} for o in self._ops if o["op"] == "reserve" and o["id"] not in consumed_res]
        reserved = float(sum(r["level"] for r in open_res))
        grades = []
        for o in grade_ops:
            charged += float(o.get("level", 0.0))
            sm = o.get("summary", {})
            grades.append({"certificate": o.get("certificate"), "level": float(o.get("level", 0.0)), "bar": o.get("bar"), "k": o.get("k"), "confirmed": sm.get("confirmed"), "reversed": sm.get("reversed"), "not_confirmed": sm.get("not_confirmed"), "reservation": o.get("reservation")})
        refine_rows = [{"parent": o["parent"], "id": o["id"], "member": o["member"], "p": float(o["p"]), "bar": float(o["bar"]), "multiplicity": int(o["multiplicity"])} for o in self._ops if o["op"] == "refine" and o["id"] not in dead]
        voids = [{"id": i, "cause": cz} for i, cz in dead.items()]
        notes = [str(o.get("text", o.get("note", ""))) for o in self._ops if o.get("op") == "note"]
        watches = self._watch_rows()
        charged += float(sum(w["level"] for w in watches))
        return Statement(path=self.path, budget=self.budget, charged=charged, reserved=reserved, remaining=self.budget - charged - reserved, n_active=len(items) + len(unknowns) + len(refine_rows), n_void=len(voids), items=items, unknowns=unknowns, voids=voids, open_reservations=open_res, notes=notes, grades=grades, refinements=refine_rows, holdouts=self._holdout_rows(), watches=watches, routes=[o for o in self._ops if o.get("op") == "route"])

    def __len__(self):
        return len(self._registered())
