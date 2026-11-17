"""Small geometry summary of a closed-loop trajectories.npz, so the big file need not be kept.

  python traj_geom.py <trajectories.npz> <out.json>

Per rollout (window or clip): yaw error and horizontal root error at the last executed step, mean joint error.
Same definitions as tools/offline/session_summary.py::window_geom.
"""
from __future__ import annotations

import json
import sys

import numpy as np


def quat_yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main(src: str, dst: str) -> None:
    z = np.load(src)
    q, rq, ex = z["qpos"], z["ref_qpos"], z["executed_steps"]
    names = [str(x) for x in z["names"]]
    yaw, xy, joint = [], [], []
    for i in range(q.shape[0]):
        T = int(ex[i])
        yaw.append(float(np.degrees(abs(wrap(quat_yaw(q[i, T, 3:7]) - quat_yaw(rq[i, T, 3:7]))))))
        xy.append(float(np.linalg.norm(q[i, T, :2] - rq[i, T, :2])))
        joint.append(float(np.abs(wrap(q[i, : T + 1, 7:] - rq[i, : T + 1, 7:])).mean()))
    out = {"n": len(names), "end_yaw_deg": float(np.mean(yaw)), "end_xy_m": float(np.mean(xy)), "joint_err_rad": float(np.mean(joint)),
           "end_yaw_deg_median": float(np.median(yaw)), "end_xy_m_median": float(np.median(xy)),
           "per_rollout": {"name": names, "executed_steps": [int(x) for x in ex], "end_yaw_deg": [round(v, 3) for v in yaw], "end_xy_m": [round(v, 4) for v in xy], "joint_err_rad": [round(v, 5) for v in joint]}}
    json.dump(out, open(dst, "w"))
    print(json.dumps({k: round(out[k], 3) for k in ("end_yaw_deg", "end_xy_m", "joint_err_rad")}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
