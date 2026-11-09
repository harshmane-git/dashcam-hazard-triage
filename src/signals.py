"""Phase 2: TTC / proximity signals, adaptive ego-lane, triggers, incident merging.

Usage (from project root):
    python src/signals.py data/raw/clip.mp4
Reads outputs/phase1/<name>_tracks.csv and writes to outputs/phase2/:
    <name>_signals.csv     one row per tracked box per frame (TTC, lane offset, trigger flags)
    <name>_incidents.csv   triggers merged into incidents (input for Phase 3)
    <name>_calib.json      the horizon / vanishing point used for this video

TTC model: box height is inversely proportional to distance, so
    TTC ~= h / (dh/dt)
with dh/dt from a least-squares line over the last `window_s` seconds.

Adaptive lane (roi.mode = auto): for any object on the road plane
    y2 - y_h = camH * (box_h / H_real)      (bottom edge vs. size  -> horizon row y_h)
    cx - x_vp = X   * (box_h / H_real)      (centre column vs. size -> vanishing column x_vp)
so both are fitted per video from the tracks. An object's lateral offset in the road
plane, in camera-heights, is then  u = (cx - x_vp) / (y2 - y_h), independent of distance.
It counts when |u| <= lane OR its predicted u at the time of collision is inside the lane
(cut-ins), so parked / adjacent-lane cars that keep their lane are ignored.
"""
import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
NAN = float("nan")
INCIDENT_COLS = ["incident_id", "start_frame", "end_frame", "start_t", "end_t", "kinds",
                 "primary_track", "primary_cls", "min_ttc", "max_h_frac", "risk_duration_s", "n_tracks"]


# ---------- static geometry (static mode / fallback drawing) ----------
def roi_polygon(roi, w, h):
    cx, ty, by = roi["center_x"], roi["horizon_y"], roi["bottom_y"]
    tw, bw = roi["top_half_width"], roi["bottom_half_width"]
    pts = [(cx - tw, ty), (cx + tw, ty), (cx + bw, by), (cx - bw, by)]
    return np.array([[int(x * w), int(y * h)] for x, y in pts], np.int32)


def in_roi(cx, y2, roi, widen=1.0):
    """cx, y2 are fractions of frame width/height (bottom-center of the box)."""
    ty, by = roi["horizon_y"], roi["bottom_y"]
    if y2 < ty:
        return False
    t = min(max((y2 - ty) / (by - ty), 0.0), 1.0)
    hw = (roi["top_half_width"] + t * (roi["bottom_half_width"] - roi["top_half_width"])) * widen
    return abs(cx - roi["center_x"]) <= hw


def lane_polygon(calib, roi, w, h, widen=1.0):
    """Perspective lane drawn from the calibrated vanishing point."""
    y_top = calib["y_h"] + 0.06 * h
    y_bot = roi["bottom_y"] * h
    u = calib["u_lane"] * widen
    pts = {}
    for y in (y_top, y_bot):
        dy = y - calib["y_h"]
        c = calib["x_vp"] + roi["lane_offset_u"] * dy
        pts[y] = (c - u * dy, c + u * dy)
    return np.array([[pts[y_top][0], y_top], [pts[y_top][1], y_top],
                     [pts[y_bot][1], y_bot], [pts[y_bot][0], y_bot]], np.int32)


# ---------- calibration ----------
def _robust_line(x, y, iters=3, keep=0.8):
    """y = a*x + b by least squares with iterative trimming of the worst residuals."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    idx = np.arange(len(x))
    for _ in range(iters):
        a, b = np.polyfit(x[idx], y[idx], 1)
        res = np.abs(y - (a * x + b))
        idx = np.argsort(res)[: max(10, int(keep * len(x)))]
    a, b = np.polyfit(x[idx], y[idx], 1)
    return a, b


def calibrate(df, w, h, cfg):
    roi = cfg["roi"]
    cc = roi["calibration"]
    out = {"mode": "auto", "y_h": roi["horizon_y"] * h, "x_vp": roi["center_x"] * w,
           "u_lane": roi["u_lane"], "min_dy": 0.03 * h,
           "y_source": "config", "x_source": "config", "n_points": 0, "n_tracks": 0}
    veh = df[df.cls.isin(cfg["fcw"]["vehicle_classes"]) & (df.conf >= cc["min_conf"])].copy()
    veh["bh"] = veh.y2 - veh.y1
    veh["cx"] = (veh.x1 + veh.x2) / 2
    m = 8   # boxes cut by the frame border have unreliable size / bottom edge
    veh = veh[(veh.x1 > m) & (veh.x2 < w - m) & (veh.y2 < h - m) & (veh.bh > 0.03 * h)]

    # horizon row: fit y2 against class-normalised box height, intercept = horizon
    if len(veh) >= cc["min_points"]:
        hn = veh.bh / veh.cls.map(cc["class_height_m"]).fillna(1.5)
        a, b = _robust_line(hn.values, veh.y2.values)
        if a > 0 and 0.15 * h <= b <= 0.85 * h:
            out.update(y_h=float(b), y_source="fitted", n_points=int(len(veh)))

    # vanishing column: per-track fit of cx against box height, intercept = x_vp, median vote
    votes = []
    for _, g in veh.groupby("track_id"):
        if len(g) < cc["min_track_frames"] or g.bh.max() / g.bh.min() < cc["min_size_ratio"]:
            continue
        votes.append(np.polyfit(g.bh.values, g.cx.values, 1)[1])
    if len(votes) >= cc["min_tracks"]:
        x = float(np.median(votes))
        if 0.2 * w <= x <= 0.8 * w:
            out.update(x_vp=x, x_source="fitted", n_tracks=len(votes))

    if roi.get("force_horizon_y") is not None:
        out.update(y_h=roi["force_horizon_y"] * h, y_source="manual")
    if roi.get("force_center_x") is not None:
        out.update(x_vp=roi["force_center_x"] * w, x_source="manual")
    return out


# ---------- per-track state ----------
class State:
    __slots__ = ("t", "h", "tu", "u", "age", "roi_run", "risk_run", "last_t")

    def __init__(self):
        self.t, self.h, self.tu, self.u = deque(), deque(), deque(), deque()
        self.age = self.roi_run = self.risk_run = 0
        self.last_t = None


def fit_window(ts, vs, window_s, min_samples):
    """Line through (t, v) over the window -> (value_now, slope) or None if too little data."""
    if len(ts) < min_samples:
        return None
    t = np.array(ts) - ts[0]
    if t[-1] < 0.5 * window_s:
        return None
    slope, icpt = np.polyfit(t, np.array(vs), 1)
    return slope * t[-1] + icpt, slope


def estimate(state, window_s, min_samples, k):
    """Return (ttc, relative_growth_per_s, ttc_conservative), or Nones if too little data.

    ttc_conservative uses the growth slope minus k standard errors, so detector jitter on
    small / distant boxes cannot produce a short TTC; the trigger uses this one."""
    if len(state.t) < min_samples:
        return None, None, None
    t = np.array(state.t) - state.t[0]
    if t[-1] < 0.5 * window_s:
        return None, None, None
    y = np.array(state.h)
    slope, icpt = np.polyfit(t, y, 1)
    h_now = slope * t[-1] + icpt
    if h_now <= 1.0:
        return None, None, None
    res = y - (slope * t + icpt)
    se = math.sqrt(float(res @ res) / (len(t) - 2) / float(((t - t.mean()) ** 2).sum()))
    ttc = h_now / slope if slope > 1e-6 else math.inf
    lo = slope - k * se
    ttc_c = h_now / lo if lo > 1e-6 else math.inf
    return ttc, slope / h_now, ttc_c


def _trim(ts, vs, now, window_s):
    while ts and now - ts[0] > window_s:
        ts.popleft(); vs.popleft()


# ---------- core ----------
def analyze(df, w, h, fps, cfg, verbose=True):
    roi, fc, pc = cfg["roi"], cfg["fcw"], cfg["pedestrian"]
    veh, vru = set(fc["vehicle_classes"]), set(pc["classes"])
    auto = roi["mode"] == "auto"
    calib = calibrate(df, w, h, cfg) if auto else {"mode": "static"}
    if verbose and auto:
        print(f"Calibration: horizon y={calib['y_h']:.0f}px ({calib['y_h'] / h:.2f} H, {calib['y_source']}), "
              f"vanishing x={calib['x_vp']:.0f}px ({calib['x_vp'] / w:.2f} W, {calib['x_source']}); "
              f"{calib['n_points']} points, {calib['n_tracks']} tracks")
    states = defaultdict(State)
    rows = []

    for frame, g in df.groupby("frame", sort=True):
        for r in g.itertuples(index=False):
            s = states[r.track_id]
            if s.last_t is not None:
                gap = r.t_sec - s.last_t
                if gap > cfg["gap_reset_s"]:
                    for q in (s.t, s.h, s.tu, s.u):
                        q.clear()
                    s.roi_run = s.risk_run = 0
                elif gap > 2.5 / fps:      # skipped frames: the risk must be re-confirmed
                    s.risk_run = 0
            s.last_t = r.t_sec
            s.age += 1
            hpx = r.y2 - r.y1
            mg = cfg["edge_margin_px"]
            lat_cut = r.x1 <= mg or r.x2 >= w - mg   # beside / overtaking: size and position unreliable
            top_cut = r.y1 <= mg                      # height clipped by the frame
            if not (lat_cut or top_cut):
                s.t.append(r.t_sec); s.h.append(hpx)
            _trim(s.t, s.h, r.t_sec, fc["window_s"])

            is_veh = r.cls in veh
            cxp = (r.x1 + r.x2) / 2
            ttc, rel, ttc_c = ((None, None, None) if (lat_cut or top_cut)
                               else estimate(s, fc["window_s"], fc["min_samples"], fc["slope_k"]))

            u = u_pred = NAN
            if auto:
                dy = r.y2 - calib["y_h"]
                inside = False
                if dy >= calib["min_dy"]:
                    u = (cxp - calib["x_vp"]) / dy - roi["lane_offset_u"]
                    s.tu.append(r.t_sec); s.u.append(u)
                    _trim(s.tu, s.u, r.t_sec, roi["u_window_s"])
                    fit = fit_window(s.tu, s.u, roi["u_window_s"], fc["min_samples"])
                    u_s = fit[0] if fit else u
                    u_pred = u_s
                    if fit and ttc is not None and 0 < ttc < math.inf:
                        u_pred = u_s + fit[1] * min(ttc, roi["pred_cap_s"])
                    lane = roi["u_lane"] * (1.0 if is_veh else pc["roi_widen"])
                    inside = abs(u_s) <= lane or abs(u_pred) <= lane
            else:
                inside = in_roi(cxp / w, r.y2 / h, roi, 1.0 if is_veh else pc["roi_widen"])
            hf = hpx / h
            if lat_cut:
                inside = False
            s.roi_run = s.roi_run + 1 if inside else 0

            raw, kind = False, ""
            if s.age >= cfg["min_track_len"] and s.roi_run >= cfg["roi_persist_frames"]:
                if is_veh:
                    if hf >= fc["min_h_frac"] and ttc_c is not None and ttc_c <= fc["ttc_thresh_s"]:
                        raw, kind = True, "collision"
                elif r.cls in vru:
                    close = hf >= pc["prox_h_frac"]
                    closing = (ttc_c is not None and ttc_c <= pc["ttc_thresh_s"]
                               and hf >= pc["min_h_frac"])
                    if close or closing:
                        raw, kind = True, "proximity"
            s.risk_run = s.risk_run + 1 if raw else 0
            fired = raw and s.risk_run >= max(2, round(cfg["persist_s"] * fps))

            rows.append({
                "frame": frame, "t_sec": r.t_sec, "track_id": r.track_id, "cls": r.cls,
                "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2,
                "h_px": round(hpx, 1), "h_frac": round(hf, 4), "in_roi": inside,
                "u": round(u, 3) if not math.isnan(u) else NAN,
                "u_pred": round(u_pred, 3) if not math.isnan(u_pred) else NAN,
                "ttc": round(ttc, 3) if ttc is not None and math.isfinite(ttc) else NAN,
                "ttc_cons": round(ttc_c, 3) if ttc_c is not None and math.isfinite(ttc_c) else NAN,
                "rel_growth": round(rel, 4) if rel is not None else NAN,
                "kind": kind, "raw": raw, "fired": fired,
            })

    sig = pd.DataFrame(rows)
    incidents = merge_incidents(sig[sig.fired] if len(sig) else sig, fps, cfg["cooldown_s"], verbose)
    return sig, incidents, calib


def merge_incidents(fired, fps, cooldown_s, verbose=True):
    incs, cur = [], None

    def close(inc, idx):
        tr = inc["tracks"]
        finite = {k: v for k, v in tr.items() if math.isfinite(v["min_ttc"])}
        pid = (min(finite, key=lambda k: finite[k]["min_ttc"]) if finite
               else max(tr, key=lambda k: tr[k]["max_h"]))
        mt = min(v["min_ttc"] for v in tr.values())
        return {
            "incident_id": idx, "start_frame": inc["start_frame"], "end_frame": inc["end_frame"],
            "start_t": inc["start_t"], "end_t": inc["end_t"],
            "kinds": "+".join(sorted(inc["kinds"])),
            "primary_track": pid, "primary_cls": tr[pid]["cls"],
            "min_ttc": round(mt, 3) if math.isfinite(mt) else NAN,
            "max_h_frac": round(max(v["max_h"] for v in tr.values()), 4),
            "risk_duration_s": round(len(inc["frames"]) / fps, 3),
            "n_tracks": len(tr),
        }

    for r in fired.sort_values("frame").to_dict("records"):
        if cur is None or r["t_sec"] - cur["end_t"] > cooldown_s:
            if cur:
                incs.append(close(cur, len(incs) + 1))
            cur = {"start_frame": r["frame"], "start_t": r["t_sec"], "end_frame": r["frame"],
                   "end_t": r["t_sec"], "kinds": set(), "frames": set(), "tracks": {}}
            if verbose:
                label = "COLLISION RISK" if r["kind"] == "collision" else "PROXIMITY RISK"
                extra = (f"TTC={r['ttc']:.1f}s" if not math.isnan(r["ttc"])
                         else f"size={r['h_frac']:.0%}")
                print(f"[ALERT] t={r['t_sec']:6.2f}s  {label}  {r['cls']}#{r['track_id']}  {extra}")
        cur["end_t"], cur["end_frame"] = r["t_sec"], r["frame"]
        cur["kinds"].add(r["kind"]); cur["frames"].add(r["frame"])
        tr = cur["tracks"].setdefault(r["track_id"], {"cls": r["cls"], "min_ttc": math.inf, "max_h": 0.0})
        if not math.isnan(r["ttc"]):
            tr["min_ttc"] = min(tr["min_ttc"], r["ttc"])
        tr["max_h"] = max(tr["max_h"], r["h_frac"])
    if cur:
        incs.append(close(cur, len(incs) + 1))
    return pd.DataFrame(incs, columns=INCIDENT_COLS)   # fixed columns, so 0 incidents still gives a valid CSV


def video_meta(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return w, h, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--mode", choices=["auto", "static"], help="override roi.mode from the config")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.mode:
        cfg["roi"]["mode"] = args.mode
    src = Path(args.video)
    tracks = ROOT / "outputs" / "phase1" / f"{src.stem}_tracks.csv"
    if not tracks.exists():
        raise SystemExit(f"Run Phase 1 first, missing {tracks}")
    w, h, fps = video_meta(src)
    df = pd.read_csv(tracks)
    print(f"{src.name}: {w}x{h} @ {fps:.1f} fps, {len(df)} tracked boxes, roi mode: {cfg['roi']['mode']}")

    sig, inc, calib = analyze(df, w, h, fps, cfg)
    out = ROOT / "outputs" / "phase2"
    out.mkdir(parents=True, exist_ok=True)
    sig.to_csv(out / f"{src.stem}_signals.csv", index=False)
    inc.to_csv(out / f"{src.stem}_incidents.csv", index=False)
    json.dump(calib, open(out / f"{src.stem}_calib.json", "w"), indent=1)
    dur_min = (df.t_sec.max() or 1) / 60
    print(f"\n{len(inc)} incidents in {dur_min * 60:.0f}s of video ({len(inc) / dur_min:.1f} per minute)")
    if len(inc):
        print(inc.to_string(index=False))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()