"""Plot walk1_subject1 drift / arm-tracking curves for tier A (release weights), B (InstantTrack) and C (Cuboi pc-h32).

python3 -I tools/offline/plot_drift_walk1.py <g1.xml> <spec.json> <out.png>
Uses the same loaders/metrics as drift_timeseries.py (stored real closed-loop trajectories, no simulation).
"""
import json, sys
from pathlib import Path

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drift_timeseries import DT, load, quat_yaw, wrap  # noqa: E402

LABEL = {"C_final_full": "C: Cuboi paper contract H32 from scratch (it7900, 3144 upd)", "A_release_LAFAN": "A: Cuboi released weights (BONES, 39968 upd)",
         "B_instanttrack": "B: InstantTrack v1.3 seed0 (0.5 h)"}
COLOR = {"C_final_full": "#d1495b", "A_release_LAFAN": "#edae49", "B_instanttrack": "#00798c"}


def main(xml, spec_path, out):
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    specs = [s for s in json.load(open(spec_path)) if s["tag"] in LABEL]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for s in specs:
        s = dict(s, clips=["walk1_subject1"])
        for _, q, r, _ in load(s):
            t = np.arange(q.shape[0]) * DT
            yaw = np.degrees(np.abs(wrap(quat_yaw(q[:, 3:7]) - quat_yaw(r[:, 3:7]))))
            xy = np.linalg.norm(q[:, :2] - r[:, :2], axis=-1)
            arms = np.abs(q[:, 22:36] - r[:, 22:36]).mean(-1)
            m = t <= 60
            ax[0].plot(t[m], yaw[m], color=COLOR[s["tag"]], label=LABEL[s["tag"]], lw=1.2)
            ax[1].plot(t[m], xy[m], color=COLOR[s["tag"]], lw=1.2)
            m = t <= 10
            ax[2].plot(t[m], arms[m], color=COLOR[s["tag"]], lw=1.2)
    ax[0].axhline(15, color="gray", ls=":", lw=1)
    ax[0].set(title="pelvis yaw error |Δyaw| (°), 0–60 s", xlabel="t (s)")
    ax[1].set(title="pelvis horizontal position error (m), 0–60 s", xlabel="t (s)")
    ax[2].axvspan(0, 2.46, color="gray", alpha=0.15)
    ax[2].text(0.1, ax[2].get_ylim()[1] * 0.92, "reference arm-raise 0–2.46 s", fontsize=9)
    ax[2].set(title="arm, mean absolute error over 14 joints (rad), 0–10 s", xlabel="t (s)")
    ax[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("walk1_subject1 · fullclip nominal · closed-loop trajectory vs reference", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
