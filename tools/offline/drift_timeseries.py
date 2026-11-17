"""Offline drift / joint-tracking time series from stored closed-loop trajectories (no simulation, no policy).

python3 -I tools/offline/drift_timeseries.py <g1.xml> <out.json> <spec.json>
spec.json = [{"tag": str, "kind": "cuboi"|"instanttrack", "traj": path, "clips": [names]}, ...]

kind=cuboi       : trajectories.npz from eval_fullclip_traj.py (qpos/ref_qpos (n,T+1,36), env-origin relative, wxyz)
kind=instanttrack: traj_final.npz from instanttrack.evaluate --record (time-major, ref indexed by absolute frame f)

Per clip it reports, at fixed times (s) and as means over alive frames:
  pelvis xy / yaw / height error (from the free joint, no FK), torso_link xy / yaw (FK),
  joint abs error (all 29, legs 0-11, waist 12-14, arms 15-28), pelvis-relative yaw-aligned body position error (FK).
Also linear fits of yaw error and xy error vs time (drift rate). 50 Hz control assumed (dt 0.02).
"""
import json, sys
import numpy as np
import mujoco

DT = 0.02
TIMES = (0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 180, 240)
TRACKED = ("pelvis", "left_knee_link", "left_ankle_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
           "left_elbow_link", "left_wrist_yaw_link", "right_elbow_link", "right_wrist_yaw_link")
GROUPS = {"legs": slice(0, 12), "waist": slice(12, 15), "arms": slice(15, 29), "all": slice(0, 29)}


def quat_yaw(q):  # wxyz
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rotm_to_quat(R):
    out = np.zeros(R.shape[:-2] + (4,))
    for i in np.ndindex(R.shape[:-2]):
        mujoco.mju_mat2Quat(out[i], R[i].reshape(-1).astype(np.float64))
    return out


def fk(m, d, qpos, bid):
    T = qpos.shape[0]
    pos = np.zeros((T, len(bid), 3)); yaw = np.zeros((T, len(bid)))
    for k in range(T):
        d.qpos[:] = qpos[k]; mujoco.mj_kinematics(m, d)
        pos[k] = d.xpos[bid]; R = d.xmat[bid].reshape(-1, 3, 3); yaw[k] = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    return pos, yaw


def analyse(m, d, bid, q, r):
    """q, r: (T,36) robot / reference qpos, wxyz."""
    T = q.shape[0]
    t = np.arange(T) * DT
    pxy = np.linalg.norm(q[:, :2] - r[:, :2], axis=-1)
    pyaw = np.degrees(np.abs(wrap(quat_yaw(q[:, 3:7]) - quat_yaw(r[:, 3:7]))))
    signed_yaw = np.degrees(np.unwrap(quat_yaw(q[:, 3:7])) - np.unwrap(quat_yaw(r[:, 3:7])))
    pz = np.abs(q[:, 2] - r[:, 2])
    je = np.abs(wrap(q[:, 7:] - r[:, 7:]))
    P, Y = fk(m, d, q.astype(np.float64), bid)
    PR, YR = fk(m, d, r.astype(np.float64), bid)
    j = TRACKED.index("torso_link")
    txy = np.linalg.norm(P[:, j, :2] - PR[:, j, :2], axis=-1)
    tyaw = np.degrees(np.abs(wrap(Y[:, j] - YR[:, j])))

    def local(p, yaw):  # pelvis-relative, rotated into pelvis heading frame
        rel = p - p[:, :1]
        c, s = np.cos(-yaw[:, :1]), np.sin(-yaw[:, :1])
        x = c * rel[..., 0] - s * rel[..., 1]; y = s * rel[..., 0] + c * rel[..., 1]
        return np.stack([x, y, rel[..., 2]], -1)

    body_loc = np.linalg.norm(local(P, Y) - local(PR, YR), axis=-1)[:, 1:].mean(-1)
    # reference path length (how far the reference walks), to normalise drift
    path = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(r[:, :2], axis=0), axis=-1))])
    series = {"pelvis_xy_m": pxy, "pelvis_yaw_deg": pyaw, "pelvis_signed_yaw_deg": signed_yaw, "pelvis_z_m": pz,
              "torso_xy_m": txy, "torso_yaw_deg": tyaw, "body_local_m": body_loc,
              **{f"joint_{g}_rad": je[:, s].mean(-1) for g, s in GROUPS.items()}}
    at = {}
    for tt in TIMES:
        k = int(round(tt / DT))
        if k < T:
            at[str(tt)] = {kk: float(v[k]) for kk, v in series.items()} | {"ref_path_m": float(path[k])}
    win = lambda v, a, b: float(v[int(a / DT):min(T, int(b / DT))].mean()) if int(a / DT) < T else None  # noqa: E731
    windows = {f"{a}-{b}s": {kk: win(v, a, b) for kk, v in series.items()} for a, b in ((0, 1), (0, 2), (2, 10), (10, 60), (60, 1e9))}
    fit = {}
    for kk in ("pelvis_signed_yaw_deg", "pelvis_xy_m"):
        if T > 100:
            fit[kk + "_per_s"] = float(np.polyfit(t, series[kk], 1)[0])
    first = {}
    for thr_name, kk, thr in (("yaw>15deg", "pelvis_yaw_deg", 15), ("xy>0.5m", "pelvis_xy_m", 0.5), ("xy>1m", "pelvis_xy_m", 1.0)):
        idx = np.nonzero(series[kk] > thr)[0]
        first[thr_name] = float(idx[0] * DT) if idx.size else None
    per_joint_0_2s = je[: int(2 / DT)].mean(0).tolist()
    per_joint_all = je.mean(0).tolist()
    return {"frames": T, "seconds": T * DT, "at_time_s": at, "window_means": windows, "linear_drift": fit,
            "first_exceed_s": first, "mean": {kk: float(v.mean()) for kk, v in series.items()},
            "per_joint_abs_err_0_2s_rad": per_joint_0_2s, "per_joint_abs_err_all_rad": per_joint_all,
            "ref_path_total_m": float(path[-1])}


def load(spec):
    t = np.load(spec["traj"])
    names = list(t["names"])
    for c in spec["clips"]:
        i = names.index(c)
        if spec["kind"] == "cuboi":
            T = int(t["executed_steps"][i]) + 1
            yield c, t["qpos"][i, :T], t["ref_qpos"][i, :T], {"executed_steps": int(t["executed_steps"][i])}
        else:
            alive = t["alive"][:, i]
            T = int(alive.sum())  # alive[t] before step t -> states 0..T-1 valid
            f = t["f"][:T, i]
            q = np.concatenate([t["root_p"][:T, i], t["root_quat"][:T, i], t["q"][:T, i]], -1)
            rq = rotm_to_quat(t["ref_root_R"][f])
            r = np.concatenate([t["ref_root_p"][f], rq, t["ref_q"][f]], -1)
            yield c, q, r, {"executed_steps": int(t["steps"][i]), "failed": bool(t["failed_eval"][i]),
                            "ret": float(t["ret"][i]), "clip_len": int(t["clip_len"][i])}


def main(xml, out, spec_path):
    m = mujoco.MjModel.from_xml_path(xml); d = mujoco.MjData(m)
    assert m.nq == 36
    bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in TRACKED]
    assert min(bid) >= 0
    specs = json.load(open(spec_path))
    res = []
    for s in specs:
        for c, q, r, meta in load(s):
            a = analyse(m, d, bid, q, r)
            res.append({"tag": s["tag"], "clip": c, "traj": s["traj"], **meta, **a})
            w = a["window_means"]
            print(f"{s['tag']:28s} {c:34s} T={a['seconds']:7.1f}s  xy@10s={a['at_time_s'].get('10',{}).get('pelvis_xy_m',float('nan')):.2f} "
                  f"yaw0-2={w['0-2s']['pelvis_yaw_deg']:.1f} j0-2={w['0-2s']['joint_all_rad']:.3f} arms0-2={w['0-2s']['joint_arms_rad']:.3f} "
                  f"jall={a['mean']['joint_all_rad']:.3f} loc={a['mean']['body_local_m']:.3f} first15deg={a['first_exceed_s']['yaw>15deg']}")
    json.dump({"schema_version": 1, "tool": "tools/offline/drift_timeseries.py", "xml": xml, "mujoco": mujoco.__version__,
               "dt": DT, "joint_groups": {k: [v.start, v.stop] for k, v in GROUPS.items()}, "results": res}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
