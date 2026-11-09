"""Tier 2 risk classification behind one swappable interface.

    level, reason = classifier.predict(features)      # level in {"Low", "Medium", "High"}

`features` is a flat dict built from the Tier 1 signals (see FEATURES). RuleClassifier is the
Phase 3 baseline. The learned model in Phase 5 only has to implement the same `predict`
method and be selected with `classifier.type` in the config; nothing else in the pipeline
changes.
"""
import math

LEVELS = ["Low", "Medium", "High"]

# Columns written to the incident log; this is also the training table for Phase 5.
FEATURES = ["min_ttc", "max_h_frac", "risk_duration_s", "max_rel_growth",
            "min_abs_u", "n_tracks", "is_vru"]


class RuleClassifier:
    """Severity from how soon contact would happen (TTC), how close the object got
    (box size) and how long the danger lasted."""

    def __init__(self, cfg):
        self.c = cfg["classifier"]

    def predict(self, f):
        c, hi, me = self.c, self.c["high"], self.c["medium"]
        ttc = f.get("min_ttc")
        ttc = math.inf if ttc is None or math.isnan(ttc) else ttc
        level, why = 0, []

        if ttc <= hi["max_ttc_s"]:
            level = 2; why.append(f"min TTC {ttc:.2f}s <= {hi['max_ttc_s']}s")
        if f["max_h_frac"] >= hi["min_h_frac"]:
            level = 2; why.append(f"object filled {f['max_h_frac']:.0%} of frame height (>= {hi['min_h_frac']:.0%})")
        if level < 2:
            if ttc <= me["max_ttc_s"]:
                level = 1; why.append(f"min TTC {ttc:.2f}s <= {me['max_ttc_s']}s")
            if f["risk_duration_s"] >= me["min_duration_s"]:
                level = 1; why.append(f"risk lasted {f['risk_duration_s']:.2f}s (>= {me['min_duration_s']}s)")
        if f.get("is_vru") and c["vru_escalate"] and level < 2:
            level += 1; why.append("pedestrian/cyclist: escalated one level")
        if not why:
            why.append("trigger fired but no severity rule was met")
        return LEVELS[level], "; ".join(why)


def load_classifier(cfg):
    kind = cfg["classifier"]["type"]
    if kind == "rules":
        return RuleClassifier(cfg)
    raise SystemExit(f"Unknown classifier.type '{kind}' (Phase 5 adds 'learned')")
