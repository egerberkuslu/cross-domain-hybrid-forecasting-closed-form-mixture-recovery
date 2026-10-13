"""Round 7 diagnosis, supplementary table (DIAGNOSIS7.md section 3): per-trace drift of the injected z against K, the validation-to-test change of the
scale of q, and the degenerate-q check (share of windows where the Bolt or E-Q 60-window MSE is below 1e-10 so that q is clipped). Reads diag7_rows.csv,
the round 6 validation streams and the REVEALED round 6 test streams (design only). Writes diag7_extra.csv and diag7_extra_summary.json."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
R6 = HERE.parent / "round6"
sys.path.insert(0, str(R6))
import r6_lib as L6
A2, M3 = L6.A2, L6.M3
d = pd.read_csv(HERE / "diag7_rows.csv")
fr = json.loads((R6 / "frozen_r6.json").read_text())
out = []
for name in d.series.unique():
    rec = {"series": name}
    for sp, dd in (("val", R6 / "r6_val"), ("test", R6 / "r6_test")):
        y, fE, fB = M3.streams_from(M3.load_h6(dd, name))
        t, a, r, q, e = A2.streams(y, fE, fB)
        rec[f"{sp}_q_median"] = float(np.median(q))
        rec[f"{sp}_q_clip_share"] = float(np.mean(q < np.log(1e-10)) + np.mean(q > -np.log(1e-10)))
        rec[f"{sp}_flat_y_share"] = float(np.mean(np.abs(np.diff(y)) < 1e-9))
        rec[f"{sp}_q_sd"] = float(np.std(q))
    rec["q_sd_ratio_test_over_val"] = rec["test_q_sd"] / rec["val_q_sd"]
    for arm in ("V1", "V3", "STB"):
        r_ = d[(d.series == name) & (d.arm == arm)].iloc[0]
        inj = json.loads(r_.inj)
        K = r_.K
        rec[f"{arm}_K"] = K
        rec[f"{arm}_level"] = r_.level
        rec[f"{arm}_floor_binds_test"] = bool(fr["params"][name][arm].get("floor", 0) > rec["test_q_sd"]) if arm != "STB" else None
        rec[f"{arm}_zinj_minus_K"] = float(np.mean([x["z_inj_mean"] for x in inj]) - K)
        rec[f"{arm}_cps_zinj_below_K"] = int(sum(x["z_inj_mean"] < K for x in inj))
        rec[f"{arm}_det"] = r_.r6_det
        rec[f"{arm}_c2"] = bool(r_.r6_c2)
        rec[f"{arm}_joint"] = bool(r_.r6_joint)
    out.append(rec)
x = pd.DataFrame(out)
x.to_csv(HERE / "diag7_extra.csv", index=False)
s = {}
for arm in ("V1", "V3", "STB"):
    f, p = x[~x[f"{arm}_c2"]], x[x[f"{arm}_c2"]]
    s[arm] = {"c2_failed_n": int(len(f)), "c2_failed_zinj_below_K_all_cps": int((f[f"{arm}_cps_zinj_below_K"] == 3).sum()),
              "c2_passed_n": int(len(p)), "c2_passed_zinj_below_K_all_cps": int((p[f"{arm}_cps_zinj_below_K"] == 3).sum()),
              "c2_failed_median_q_sd_ratio": float(f.q_sd_ratio_test_over_val.median()), "c2_passed_median_q_sd_ratio": float(p.q_sd_ratio_test_over_val.median()),
              "spearman_det_vs_zinj_minus_K": float(x[f"{arm}_det"].rank().corr(x[f"{arm}_zinj_minus_K"].rank())),
              "floor_binds_test_failed": int(f[f"{arm}_floor_binds_test"].fillna(False).astype(bool).sum()) if arm != "STB" else None,
              "floor_binds_test_passed": int(p[f"{arm}_floor_binds_test"].fillna(False).astype(bool).sum()) if arm != "STB" else None}
s["q_clip"] = {"traces_val_clip_share_gt_1pct": x[x.val_q_clip_share > 0.01].series.tolist(), "traces_test_clip_share_gt_1pct": x[x.test_q_clip_share > 0.01].series.tolist()}
(HERE / "diag7_extra_summary.json").write_text(json.dumps(s, indent=1))
print(json.dumps(s, indent=1))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(x[["series", "val_q_median", "test_q_median", "val_q_clip_share", "test_q_clip_share", "val_flat_y_share", "test_flat_y_share", "q_sd_ratio_test_over_val", "V1_K", "V1_zinj_minus_K", "V3_K", "V3_zinj_minus_K", "V3_floor_binds_test", "V3_det", "STB_K", "STB_zinj_minus_K", "STB_det"]].round(3).to_string())
