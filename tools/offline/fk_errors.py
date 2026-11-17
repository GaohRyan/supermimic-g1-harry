"""Offline FK tracking errors for fullclip-nominal trajectories (protocol v2 extra metrics).

python3 -I tools/offline/fk_errors.py <g1.xml> <trajectories.npz> <results.json> <out.json>
qpos layout = Cuboi StateIO (env-origin relative [pelvis free joint 7 | 29 joints]); ref_qpos same layout.
"""
import json, sys
import numpy as np
import mujoco

TRACKED = ("pelvis","left_hip_roll_link","left_knee_link","left_ankle_roll_link","right_hip_roll_link","right_knee_link",
           "right_ankle_roll_link","torso_link","left_shoulder_roll_link","left_elbow_link","left_wrist_yaw_link",
           "right_shoulder_roll_link","right_elbow_link","right_wrist_yaw_link")


def yaw_of(R):  # R (...,3,3) world rotation; heading of body x axis
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def main(xml, traj, res, out):
    m = mujoco.MjModel.from_xml_path(xml); d = mujoco.MjData(m)
    assert m.nq == 36, m.nq
    bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in TRACKED]
    assert min(bid) >= 0
    t = np.load(traj); r = json.load(open(res)); names = list(t["names"]); ex = t["executed_steps"]
    rows = []
    for i, nm in enumerate(names):
        T = int(ex[i]) + 1
        P = {}; Rm = {}
        for key in ("qpos", "ref_qpos"):
            q = t[key][i, :T].astype(np.float64)
            pos = np.zeros((T, len(bid), 3)); rot = np.zeros((T, len(bid), 3, 3))
            for k in range(T):
                d.qpos[:] = q[k]; mujoco.mj_kinematics(m, d)
                pos[k] = d.xpos[bid]; rot[k] = d.xmat[bid].reshape(-1, 3, 3)
            P[key], Rm[key] = pos, rot
        e = {}
        for bname in ("pelvis", "torso_link"):
            j = TRACKED.index(bname)
            dxy = np.linalg.norm(P["qpos"][:, j, :2] - P["ref_qpos"][:, j, :2], axis=-1)
            dz = np.abs(P["qpos"][:, j, 2] - P["ref_qpos"][:, j, 2])
            dyaw = np.degrees(np.abs((yaw_of(Rm["qpos"][:, j]) - yaw_of(Rm["ref_qpos"][:, j]) + np.pi) % (2 * np.pi) - np.pi))
            e[bname] = {"xy_m_mean": float(dxy.mean()), "xy_m_end": float(dxy[-1]), "yaw_deg_mean": float(dyaw.mean()),
                        "yaw_deg_end": float(dyaw[-1]), "height_m_mean": float(dz.mean()), "height_m_max": float(dz.max()),
                        "xy_m_at_10s": float(dxy[500]) if T > 500 else None, "yaw_deg_at_10s": float(dyaw[500]) if T > 500 else None}
        wb = np.linalg.norm(P["qpos"] - P["ref_qpos"], axis=-1).mean(-1)
        rel = lambda p: p - p[:, :1]  # noqa: E731  pelvis-relative, no yaw alignment
        wb_rel = np.linalg.norm(rel(P["qpos"]) - rel(P["ref_qpos"]), axis=-1).mean(-1)
        rows.append({"clip": nm, "executed_steps": int(ex[i]), "alive_frames_used": T, **{f"{k}": v for k, v in e.items()},
                     "whole_body_pos_err_world_m_mean": float(wb.mean()), "whole_body_pos_err_pelvis_rel_m_mean": float(wb_rel.mean())})
    summ = {k: float(np.mean([row[k] for row in rows])) for k in ("whole_body_pos_err_world_m_mean", "whole_body_pos_err_pelvis_rel_m_mean")}
    for b in ("pelvis", "torso_link"):
        for k in ("xy_m_mean", "yaw_deg_mean", "height_m_mean"):
            summ[f"{b}_{k}"] = float(np.mean([row[b][k] for row in rows]))
    json.dump({"schema_version": 1, "protocol_extra": "fullclip-nominal-v2 extra metrics (offline FK)", "xml": xml,
               "mujoco": mujoco.__version__, "source_trajectories": traj, "note": "alive-only means (frames 0..executed_steps); compare with executed_steps for survival bias; pelvis-relative whole-body error has no yaw alignment",
               "macro": summ, "clips": rows}, open(out, "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main(*sys.argv[1:5])
