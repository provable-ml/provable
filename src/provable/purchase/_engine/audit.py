"""SelfAudit: post-purchase reconciliation (P-8.3). Fields live on the order object.

Realized Delta nu-hat and realized R^2_Z are reconciled against the pre-purchase forecast within
its band. The verdict RECONCILED holds iff |realized - forecast| <= band for every reconciled
quantity; otherwise RECONCILE_FAIL, with the failing quantities named.
"""


class SelfAudit:
    def reconcile(self, items):
        checks, ok = [], True
        for it in items:
            hit = abs(it["realized"] - it["forecast"]) <= it["band"]
            ok = ok and hit
            checks.append({"quantity": it["quantity"], "forecast": float(it["forecast"]), "realized": float(it["realized"]), "band": float(it["band"]), "within": bool(hit)})
        return {"phase": "post-purchase", "checks": checks, "verdict": "RECONCILED" if ok else "RECONCILE_FAIL", "forecast": checks[0]["forecast"] if checks else None, "realized": checks[0]["realized"] if checks else None, "band": checks[0]["band"] if checks else None}
