"""The G1 on the generated terrain as one plain-MuJoCo model, built on a laptop.

The training simulator (mjlab / MuJoCo-Warp) compiles its scene on the GPU machine. For local checks this module
rebuilds the same scene without mjlab's runtime:

  terrain  mjlab's real terrain generator (tools/terrain/mjlab_terrain_local.py), same config and seed as training;
  robot    mjlab's G1 MJCF, attached with the same "robot/" prefix; every robot number that mjlab sets at build time
           (actuator gains and limits, armature, collision properties, masses ...) is copied by name from the model
           the simulator exported (scripts/export_sim2sim_contract.py -> g1_nominal.mjb), and the solver options too.

build_scene() returns the model, the tile origins and a report of what was copied and how much it changed.
"""
from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

import mjlab_terrain_local as L

L.install()  # makes mjlab.terrains.* importable (needed before a terrain config can be built)

BODY, JOINT, GEOM, ACT = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_ACTUATOR

BODY_FIELDS = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat", "body_gravcomp")
JOINT_FIELDS = ("jnt_range", "jnt_limited", "jnt_stiffness", "jnt_margin", "jnt_solref", "jnt_solimp", "jnt_pos", "jnt_axis", "jnt_actfrcrange", "jnt_actfrclimited")
DOF_FIELDS = ("dof_armature", "dof_damping", "dof_frictionloss", "dof_solref", "dof_solimp")
GEOM_FIELDS = ("geom_contype", "geom_conaffinity", "geom_condim", "geom_priority", "geom_friction", "geom_solref", "geom_solimp", "geom_solmix", "geom_margin", "geom_gap", "geom_size", "geom_pos", "geom_quat")
ACT_FIELDS = ("actuator_gainprm", "actuator_biasprm", "actuator_dynprm", "actuator_forcerange", "actuator_ctrlrange", "actuator_gear", "actuator_forcelimited", "actuator_ctrllimited", "actuator_gaintype", "actuator_biastype", "actuator_dyntype")
OPT_FIELDS = ("timestep", "integrator", "solver", "iterations", "ls_iterations", "cone", "impratio", "tolerance", "ls_tolerance", "noslip_iterations", "ccd_iterations", "disableflags", "enableflags", "o_margin")


def _name(m, t, i):
    return mujoco.mj_id2name(m, t, i)


def _copy(dst, src, field, i_dst, i_src, changed):
    a, b = getattr(dst, field), getattr(src, field)
    d = float(np.abs(np.asarray(a[i_dst], dtype=np.float64) - np.asarray(b[i_src], dtype=np.float64)).max())
    if d > 0:
        changed[field] = max(changed.get(field, 0.0), d)
    a[i_dst] = b[i_src]


def build_scene(contract_dir: Path, gen_cfg, ccd_iterations: int | None = 500, foot_friction: float | None = 1.0, foot_geom_regex: str = r"^(left|right)_foot[1-7]_collision$"):
    """-> (model, terrain_origins (rows, cols, 3), report)."""
    import re

    L.install()
    from mjlab.terrains.terrain_generator import TerrainGenerator

    mc = mujoco.MjModel.from_binary_path(str(Path(contract_dir) / "g1_nominal.mjb"))
    spec = mujoco.MjSpec()
    gen = TerrainGenerator(gen_cfg, device="cpu")
    gen.compile(spec)
    robot = mujoco.MjSpec.from_file(os.environ["G1_XML"])
    spec.attach(robot, prefix="robot/", frame=spec.worldbody.add_frame())
    # Position actuators in the exported order (the numbers are copied below).
    for i in range(mc.nu):
        a = spec.add_actuator()
        a.name = _name(mc, ACT, i)
        a.target = _name(mc, JOINT, int(mc.actuator_trnid[i, 0]))
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
    m = spec.compile()
    assert (m.nq, m.nv, m.nu) == (mc.nq, mc.nv, mc.nu), (m.nq, m.nv, m.nu)
    changed: dict[str, float] = {}
    # ---- bodies, their geoms (matched per body, in order), joints and dofs
    n_geom = 0
    for bc in range(mc.nbody):
        nm = _name(mc, BODY, bc)
        if not (nm or "").startswith("robot/"):
            continue
        bn = m.body(nm).id
        for f in BODY_FIELDS:
            _copy(m, mc, f, bn, bc, changed)
        gc = [g for g in range(mc.ngeom) if mc.geom_bodyid[g] == bc]
        gn = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bn]
        assert len(gc) == len(gn) and all(mc.geom_type[a] == m.geom_type[b] for a, b in zip(gc, gn)), f"geoms of {nm} differ"
        for a, b in zip(gc, gn):
            assert _name(mc, GEOM, a) == _name(m, GEOM, b), (nm, _name(mc, GEOM, a), _name(m, GEOM, b))
            for f in GEOM_FIELDS:
                _copy(m, mc, f, b, a, changed)
            n_geom += 1
    for jc in range(mc.njnt):
        nm = _name(mc, JOINT, jc)
        jn = m.joint(nm).id
        assert mc.jnt_type[jc] == m.jnt_type[jn] and mc.jnt_qposadr[jc] == m.jnt_qposadr[jn] and mc.jnt_dofadr[jc] == m.jnt_dofadr[jn], nm
        for f in JOINT_FIELDS:
            _copy(m, mc, f, jn, jc, changed)
        nd = 6 if mc.jnt_type[jc] == mujoco.mjtJoint.mjJNT_FREE else 1
        for k in range(nd):
            for f in DOF_FIELDS:
                _copy(m, mc, f, m.jnt_dofadr[jn] + k, mc.jnt_dofadr[jc] + k, changed)
    for i in range(mc.nu):
        assert _name(mc, ACT, i) == _name(m, ACT, i) and mc.actuator_trnid[i, 0] == m.actuator_trnid[i, 0]
        for f in ACT_FIELDS:
            _copy(m, mc, f, i, i, changed)
    m.qpos0[:] = mc.qpos0
    m.qpos_spring[:] = mc.qpos_spring
    for f in OPT_FIELDS:
        setattr(m.opt, f, getattr(mc.opt, f))
    m.opt.gravity[:] = mc.opt.gravity
    if ccd_iterations is not None:
        m.opt.ccd_iterations = ccd_iterations
    ex = lambda mm: sorted((_name(mm, BODY, int(s) >> 16), _name(mm, BODY, int(s) & 0xFFFF)) for s in mm.exclude_signature)  # noqa: E731
    assert ex(m) == ex(mc), "contact excludes differ"
    foot = [g for g in range(m.ngeom) if re.match(foot_geom_regex, (_name(m, GEOM, g) or "").split("/")[-1])]
    assert len(foot) == 14, len(foot)
    if foot_friction is not None:
        m.geom_friction[foot, 0] = foot_friction  # nominal evaluation protocol (the export carries the MJCF default 0.6)
    mujoco.mj_setConst(m, mujoco.MjData(m))
    report = {"robot_bodies": sum(1 for b in range(mc.nbody) if (_name(mc, BODY, b) or "").startswith("robot/")), "robot_geoms": n_geom, "actuators": int(mc.nu),
              "terrain_geoms": int(m.ngeom - n_geom), "fields_changed_by_copy_maxabs": changed, "foot_friction": foot_friction, "ccd_iterations": int(m.opt.ccd_iterations)}
    return m, gen.terrain_origins.copy(), report
