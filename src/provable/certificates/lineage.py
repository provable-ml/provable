"""Row lineage for certificates.

Row identity (proposed default, pending Decision 0 ruling):
row_id = (source_id, row_index) where source_id is a content fingerprint of
the source table. Every row also carries a clock key
(dataset_tag, start_utc, end_utc).

Identity freshness follows register Def 2.9: a lineage is fresh for a row
set S iff the row_id sets are disjoint. Taint is tracked at slice
granularity and conservatively: any overlap means NOT fresh; false
staleness is acceptable, false freshness is not.

The row model treats disjoint row sets as independent (register Conv 2.1).
Rows from different dataset tags at overlapping clock times are not
independent in general, so identity freshness is necessary but not
sufficient for cross-dataset transport. This module detects and reports
such clock overlaps; nothing is enforced on the flag in v0 (Decision 1).
"""
from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import numpy as np
import pandas as pd

MAX_NAMED_ROWS = 12
UNIT_NS = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}
MAX_ALIAS_COMBINATIONS = 16
_NAT = np.iinfo(np.int64).min


def _aware(t: datetime) -> datetime:
    """Clock comparisons read a naive time as UTC, so naive and aware
    declarations compare instead of raising."""
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t


def _datetime_like(obj) -> bool:
    return pd.api.types.is_datetime64_any_dtype(obj.dtype) or pd.api.types.is_timedelta64_dtype(obj.dtype)


def _int_view(obj):
    """A datetime-like column or index as (int64 values, unit): clock
    values in UTC, NaT the int64 minimum. None for a unit outside s, ms,
    us and ns."""
    if isinstance(obj.dtype, pd.DatetimeTZDtype):
        obj = obj.tz_convert("UTC").tz_localize(None) if isinstance(obj, pd.Index) else obj.dt.tz_convert("UTC").dt.tz_localize(None)
    arr = np.asarray(obj.to_numpy())
    unit = np.datetime_data(arr.dtype)[0]
    return (arr.view("i8"), unit) if unit in UNIT_NS else None


def _in_unit(v: np.ndarray, unit: str, target: str) -> Optional[np.ndarray]:
    """The same instants as int64 in another unit, or None when the
    conversion is not exact (a coarser unit drops digits, a finer one
    overflows)."""
    f, g = UNIT_NS[unit], UNIT_NS[target]
    if f == g:
        return v
    nat = v == _NAT
    w = np.where(nat, 0, v)
    if f > g:
        k = f // g
        if np.abs(w).max(initial=0) > np.iinfo(np.int64).max // k:
            return None
        out = w * k
    else:
        k = g // f
        if (w % k).any():
            return None
        out = w // k
    return np.where(nat, _NAT, out)


def _views(frame: pd.DataFrame) -> list:
    """(position, (int64 values, unit)) for each datetime-like column, in
    column order, then the index as position -1."""
    out = []
    for j in range(frame.shape[1]):
        col = frame.iloc[:, j]
        if _datetime_like(col):
            v = _int_view(col)
            if v is not None:
                out.append((j, v))
    if _datetime_like(frame.index):
        v = _int_view(frame.index)
        if v is not None:
            out.append((-1, v))
    return out


def _with_units(frame: pd.DataFrame, views: list, units: tuple):
    """The frame with each datetime-like part replaced by its int64 values
    in the given unit, or None when a part cannot hold its values exactly
    in that unit. Hashing an int64 column is hashing a datetime-like column
    with the same values, so the replacement changes nothing else."""
    g = frame.copy(deep=False)
    for (pos, (v, unit)), target in zip(views, units):
        w = _in_unit(v, unit, target)
        if w is None:
            return None
        if pos < 0:
            g.index = pd.Index(w)
        else:
            g.isetitem(pos, w)
    return g


def legacy_fingerprint(frame: pd.DataFrame) -> str:
    """The fingerprint of releases up to 0.12.0.dev0, which hashed each
    datetime-like column in whatever unit the running pandas stores it."""
    h = hashlib.sha256()
    h.update(repr([str(c) for c in frame.columns]).encode())
    h.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return h.hexdigest()[:16]


def _canonical_units(views: list) -> tuple:
    return tuple("us" if _in_unit(v, unit, "us") is not None else "ns" for _, (v, unit) in views)


def source_fingerprint(frame: pd.DataFrame) -> str:
    """Content fingerprint of a source table. Same content, same id,
    whatever pandas version or storage unit holds it: date and time-span
    columns (and a datetime-like index) are hashed as int64 microseconds,
    or as nanoseconds when their values carry sub-microsecond parts.
    Tables without such columns keep the fingerprint of earlier releases,
    and so do tables whose datetime-like parts pandas stores in
    microseconds."""
    views = _views(frame)
    if not views:
        return legacy_fingerprint(frame)
    return legacy_fingerprint(_with_units(frame, views, _canonical_units(views)))


def source_identity(frame: pd.DataFrame) -> tuple:
    """(canonical fingerprint, aliases) of a source table; see
    source_fingerprint and source_aliases. The aliases come from one pass
    over the columns that keeps a row-hash accumulator per unit
    combination and combines the column hashes the way hash_pandas_object
    does, so memory stays at a few arrays of the table's length whatever
    its width. The pass is used only when its accumulator for the units
    the table is stored in reproduces the table's legacy fingerprint;
    otherwise each combination is rehashed in full."""
    views = _views(frame)
    canonical = source_fingerprint(frame)
    native = legacy_fingerprint(frame)
    if not views or frame.shape[0] == 0:
        return canonical, frozenset((canonical, native))
    k = len(views)
    combos = list(itertools.product(UNIT_NS, repeat=k)) if 4 ** k <= MAX_ALIAS_COMBINATIONS else [(u,) * k for u in UNIT_NS]
    rows = [tuple(unit for _, (v, unit) in views)] + combos
    width = frame.shape[1]
    view_at = {(width if pos < 0 else pos): i for i, (pos, _) in enumerate(views)}
    n_items = width + 1
    acc = np.full((len(rows), frame.shape[0]), 0x345678, dtype=np.uint64)
    ok = np.ones(len(rows), dtype=bool)
    mult = np.uint64(1000003)
    with np.errstate(over="ignore"):
        for item in range(n_items):
            if item in view_at:
                i = view_at[item]
                v, unit = views[i][1]
                hashed = {}
                for r, units in enumerate(rows):
                    target = units[i]
                    if target not in hashed:
                        w = _in_unit(v, unit, target)
                        hashed[target] = None if w is None else pd.util.hash_array(w)
                    if hashed[target] is None:
                        ok[r] = False
                    else:
                        acc[r] ^= hashed[target]
            else:
                part = frame.index if item == width else frame.iloc[:, item]
                acc ^= pd.util.hash_pandas_object(part, index=False).to_numpy()[None, :]
            acc *= mult
            mult += np.uint64(82520 + 2 * (n_items - item))
        acc += np.uint64(97531)
    names = repr([str(c) for c in frame.columns]).encode()

    def digest(r: int) -> str:
        h = hashlib.sha256()
        h.update(names)
        h.update(acc[r].tobytes())
        return h.hexdigest()[:16]

    ids = {canonical, native}
    if ok[0] and digest(0) == native:
        ids.update(digest(r) for r in range(1, len(rows)) if ok[r])
    else:
        for units in combos:
            g = _with_units(frame, views, units)
            if g is not None:
                ids.add(legacy_fingerprint(g))
    return canonical, frozenset(ids)


def source_aliases(frame: pd.DataFrame) -> frozenset:
    """Every id this table may carry in a book: the canonical fingerprint,
    the legacy fingerprint as the running pandas stores the table, and the
    legacy fingerprint with the datetime-like columns and index in each
    combination of the units s, ms, us and ns that holds their values
    exactly (each part in one shared unit when there are more than
    MAX_ALIAS_COMBINATIONS combinations). Overlap checks match a recorded
    id against this set, so a book written by an earlier release, under
    either pandas, still recognises its rows."""
    return source_identity(frame)[1]


@dataclass(frozen=True)
class Slice:
    """A set of rows of one source table, with its clock envelope.

    rows is a frozenset of row indices, or None meaning every row of the
    source (the conservative whole-source taint). t0 and t1 bound the
    clock windows of the rows (UTC). aliases holds the other ids the same
    table may carry in a book written by an earlier release (see
    source_aliases); it is never stored, compared or hashed, and overlap
    checks match a recorded id against it.
    """
    source_id: str
    dataset_tag: str
    rows: Optional[frozenset]
    t0: datetime
    t1: datetime
    aliases: frozenset = field(default=frozenset(), compare=False, repr=False)

    def row_count(self):
        return None if self.rows is None else len(self.rows)

    def ids(self) -> frozenset:
        return self.aliases | {self.source_id}


def slice_from_frame(frame: pd.DataFrame, dataset_tag: str, rows: Optional[Iterable[int]] = None, start_col: str = "start_utc", end_col: str = "end_utc") -> Slice:
    sid, aliases = source_identity(frame)
    if rows is None:
        sub = frame
        rset = None
    else:
        rset = frozenset(int(r) for r in rows)
        sub = frame.loc[sorted(rset)]
    t0 = pd.Timestamp(sub[start_col].min()).to_pydatetime()
    t1 = pd.Timestamp(sub[end_col].max()).to_pydatetime()
    return Slice(source_id=sid, dataset_tag=dataset_tag, rows=rset, t0=t0, t1=t1, aliases=aliases)


def same_source(a: Slice, b: Slice) -> bool:
    """Two slices name the same source table: equal ids, or one side's id
    among the other's aliases (a book written by an earlier release)."""
    return a.source_id == b.source_id or a.source_id in b.aliases or b.source_id in a.aliases


@dataclass(frozen=True)
class Lineage:
    """An immutable set of slices. The empty lineage is Lineage()."""
    slices: frozenset = frozenset()

    @staticmethod
    def of(*slices: Slice) -> "Lineage":
        return Lineage(frozenset(slices))

    def union(self, *others: "Lineage") -> "Lineage":
        s = set(self.slices)
        for o in others:
            s |= o.slices
        return Lineage(frozenset(s))

    def is_empty(self) -> bool:
        return len(self.slices) == 0

    def __contains__(self, sl: Slice) -> bool:
        return sl in self.slices


def hereditary_union(proposer_rows: Lineage, consumed_lineages: Iterable[Lineage]) -> Lineage:
    """Definition A.5 as amended: the proposer lineage is the proposer's
    raw rows united with the full lineage of every consumed certificate."""
    return proposer_rows.union(*list(consumed_lineages))


def _slice_overlap(a: Slice, b: Slice):
    if not same_source(a, b):
        return None
    sid = a.source_id if a.source_id == b.source_id else f"{a.source_id} (= {b.source_id})"
    if a.rows is None or b.rows is None:
        shared = a.rows if b.rows is None else b.rows
        return {"source_id": sid, "rows": None if shared is None else sorted(shared), "whole_source": True}
    inter = a.rows & b.rows
    if inter:
        return {"source_id": sid, "rows": sorted(inter), "whole_source": False}
    return None


def identity_overlap(a: Lineage, b: Lineage) -> list:
    """Row_id overlaps between two lineages, named per source. Empty list
    means identity fresh (register Def 2.9)."""
    seen = {}
    for sa in a.slices:
        for sb in b.slices:
            ov = _slice_overlap(sa, sb)
            if ov is None:
                continue
            key = ov["source_id"]
            if key in seen:
                prev = seen[key]
                if prev["rows"] is None or ov["rows"] is None:
                    prev["rows"] = None
                    prev["whole_source"] = True
                else:
                    prev["rows"] = sorted(set(prev["rows"]) | set(ov["rows"]))
            else:
                seen[key] = dict(ov)
    return list(seen.values())


def identity_fresh(a: Lineage, b: Lineage) -> bool:
    return len(identity_overlap(a, b)) == 0


def clock_overlaps(a: Lineage, b: Lineage) -> list:
    """Cross-dataset contemporaneous windows (Decision 1 flag).

    Reports slice pairs with different dataset tags whose clock
    windows intersect. Report only; nothing is enforced in v0.
    """
    flags = []
    for sa in a.slices:
        for sb in b.slices:
            if sa.dataset_tag == sb.dataset_tag:
                continue
            lo = max(_aware(sa.t0), _aware(sb.t0))
            hi = min(_aware(sa.t1), _aware(sb.t1))
            if lo < hi:
                flags.append({
                    "tags": (sa.dataset_tag, sb.dataset_tag),
                    "sources": (sa.source_id, sb.source_id),
                    "window": (lo, hi),
                    "rows": (sa.row_count(), sb.row_count()),
                })
    return flags


def overlap_report(overlaps: list, max_rows: int = MAX_NAMED_ROWS) -> str:
    lines = []
    for ov in overlaps:
        if ov["rows"] is None:
            lines.append(f"source {ov['source_id']}: whole-source overlap")
        else:
            shown = ov["rows"][:max_rows]
            more = len(ov["rows"]) - len(shown)
            tail = f" (+{more} more)" if more > 0 else ""
            lines.append(f"source {ov['source_id']}: rows {shown}{tail}")
    return "; ".join(lines)


@dataclass(frozen=True)
class ColumnDeclaration:
    """Builder-declared lineage of a frame column relative to the target row.

    Either a lag declaration (source rows of dataset_tag whose clock
    window is the target window shifted back by lag; window widens it) or
    explicit rows. outcome_derived marks columns computed from realized
    outcomes of their source rows; such values exist only after the source
    rows deliver.
    """
    column: str
    dataset_tag: str
    lag: Optional[timedelta] = None
    window: Optional[timedelta] = None
    rows: Optional[Lineage] = None
    outcome_derived: bool = False

    def source_window(self, target: Slice):
        if self.lag is None:
            return None
        end = target.t1 - self.lag
        if self.window is None:
            start = target.t0 - self.lag
        else:
            start = end - self.window
        return (start, end)


class ColumnFreshnessError(AssertionError):
    pass


def check_columns(columns: Iterable[ColumnDeclaration], targets: Iterable[Slice], gate_time: datetime) -> list:
    """Column-lineage rule: for a target row r scored or tested at gate
    time g, no consumed column may have lineage containing r's own outcome
    rows or any outcome row with clock after g. Declaration plus
    assertion only; automatic propagation through frame operations is out
    of scope in v0.
    """
    violations = []
    tlist = list(targets)
    for col in columns:
        if not col.outcome_derived:
            continue
        if col.rows is not None:
            tgt_lin = Lineage(frozenset(tlist))
            ov = identity_overlap(col.rows, tgt_lin)
            if ov:
                violations.append({"column": col.column, "reason": "own outcome rows in column lineage", "detail": overlap_report(ov)})
            for sl in col.rows.slices:
                if _aware(sl.t1) > _aware(gate_time):
                    violations.append({"column": col.column, "reason": "outcome row delivers after gate time", "detail": f"source {sl.source_id} window ({sl.t0}, {sl.t1}) vs gate {gate_time}"})
        if col.lag is not None:
            for r in tlist:
                sw = col.source_window(r)
                own = (col.dataset_tag == r.dataset_tag) and (max(_aware(sw[0]), _aware(r.t0)) < min(_aware(sw[1]), _aware(r.t1)))
                win = f"[{sw[0].isoformat()} .. {sw[1].isoformat()})"
                if own:
                    violations.append({"column": col.column, "reason": "own outcome rows in column lineage", "detail": f"lag {col.lag} source window {win} overlaps target [{r.t0.isoformat()} .. {r.t1.isoformat()})"})
                elif _aware(sw[1]) > _aware(gate_time):
                    violations.append({"column": col.column, "reason": "outcome row delivers after gate time", "detail": f"source window ends {sw[1].isoformat()} after gate {gate_time.isoformat()}"})
    return violations


def assert_column_freshness(columns: Iterable[ColumnDeclaration], targets: Iterable[Slice], gate_time: datetime) -> None:
    v = check_columns(columns, targets, gate_time)
    if v:
        msgs = "; ".join(f"[{x['column']}] {x['reason']}: {x['detail']}" for x in v)
        raise ColumnFreshnessError(msgs)
