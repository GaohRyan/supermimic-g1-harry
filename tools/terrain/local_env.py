"""A small stand-in for the mjlab environment on plain MuJoCo (CPU): runs the variant's own simulator-side code on a laptop.

What is real here
  * the variant's MultiClipMotionCommand (reset placement, reference lifts, level curriculum, clip restriction, sampler),
    its termination functions, its ReplayRecorder / ReplayBuffer, StateIO and ActorScan: imported unchanged;
  * mjlab's tracking reward functions: the reference source file is executed unchanged (its two imports of the mjlab
    runtime are stubbed);
  * the terrain: mjlab's generator; the robot: the model the training simulator exported (tools/terrain/local_scene.py).
What is a stand-in
  * the environment object itself: N MjData on one shared model, stepped in mjlab's order
    (action -> physics -> termination -> reward -> reset -> forward -> command update -> observation -> recorder);
  * mjlab's manager base classes and the terrain entity's origin bookkeeping (same logic, re-written here);
  * physics is MuJoCo's C engine, not MuJoCo-Warp; no action noise / delay / domain randomisation (nominal context).
It is for checking logic and for local closed-loop runs on terrain, not a replacement of the GPU parity checks.

  source tools/offline/env.sh; export PYTHONPATH="$PYTHONPATH:$PWD/tools/terrain:$PWD/tools/offline"
"""
from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

import sm_offline  # noqa: F401  registers the import bypass for supermimic.envs (no mjlab runtime)
import local_scene as LS

MJLAB_SRC = LS.L.MJLAB_SRC


# ----------------------------------------------------------------------------- stand-ins for mjlab's manager base classes
class _TermBase:
    def __init__(self, env):
        self._env = env

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device


@dataclass(kw_only=True)
class CommandTermCfg:
    resampling_time_range: tuple[float, float]
    debug_vis: bool = False


class CommandTerm(_TermBase):
    """Same reset / compute / resample flow as mjlab.managers.command_manager.CommandTerm."""

    def __init__(self, cfg, env):
        self.cfg = cfg
        super().__init__(env)
        self.metrics = dict()
        self.time_left = torch.zeros(self.num_envs)
        self.command_counter = torch.zeros(self.num_envs, dtype=torch.long)

    def reset(self, env_ids):
        extras = {}
        for k, v in self.metrics.items():
            extras[k] = torch.mean(v[env_ids]).item()
            v[env_ids] = 0.0
        self.command_counter[env_ids] = 0
        self._resample(env_ids)
        return extras

    def compute(self, dt, env_ids=None):
        self._update_metrics()
        if env_ids is None:
            self.time_left -= dt
            resample = (self.time_left <= 0.0).nonzero().flatten()
        else:
            self.time_left[env_ids] -= dt
            resample = env_ids[self.time_left[env_ids] <= 0.0]
        if len(resample) > 0:
            self._resample(resample)
        self._update_command(env_ids)

    def _resample(self, env_ids):
        if len(env_ids) != 0:
            self.time_left[env_ids] = self.time_left[env_ids].uniform_(*self.cfg.resampling_time_range)
            self._resample_command(env_ids)
            self.command_counter[env_ids] += 1


@dataclass
class RecorderTermCfg:
    pass


class RecorderTerm(_TermBase):
    def __init__(self, cfg, env):
        self.cfg = cfg
        super().__init__(env)


def _install() -> None:
    if "mjlab.managers" in sys.modules and getattr(sys.modules["mjlab.managers"], "_local_stand_in", False):
        return
    import mjlab  # noqa: F401  the offline stub

    man = types.ModuleType("mjlab.managers")
    man.CommandTerm, man.CommandTermCfg, man._local_stand_in = CommandTerm, CommandTermCfg, True
    man.__path__ = []
    rec = types.ModuleType("mjlab.managers.recorder_manager")
    rec.RecorderTerm, rec.RecorderTermCfg = RecorderTerm, RecorderTermCfg
    sys.modules["mjlab.managers"] = man
    sys.modules["mjlab.managers.recorder_manager"] = rec


_install()

from supermimic.data.recorder import ReplayRecorderCfg  # noqa: E402
from supermimic.data import recorder as _rec_mod  # noqa: E402
from supermimic.envs.mdp import terminations as sm_term  # noqa: E402
from supermimic.envs.mdp.commands import MultiClipMotionCommandCfg  # noqa: E402
from supermimic.kinematics.rotations import quat_apply_inverse  # noqa: E402
from supermimic.terrain.terrain_cfg import TerrainArgs, make_terrain_generator_cfg  # noqa: E402
from supermimic.world_model.heads import analytic_observation, flatten_history_term_major  # noqa: E402


def _mjlab_tracking_rewards() -> dict:
    """mjlab's tasks/tracking/mdp/rewards.py, executed unchanged apart from its two runtime imports."""
    src = (MJLAB_SRC / "tasks" / "tracking" / "mdp" / "rewards.py").read_text()
    src = src.replace("from mjlab.sensor import ContactSensor\n", "ContactSensor = object\n").replace("from .commands import MotionCommand\n", "MotionCommand = object\n")
    ns: dict = {"__name__": "mjlab_tracking_rewards_local"}
    exec(compile(src, "mjlab/tasks/tracking/mdp/rewards.py", "exec"), ns)
    return ns


TRACK = _mjlab_tracking_rewards()
# (function, weight, std) of the variant's reward config (envs/tracking_env_cfg.py, Table A.2)
REWARD_TERMS = (
    ("motion_global_root_pos", "motion_global_anchor_position_error_exp", 0.5, 0.3),
    ("motion_global_root_ori", "motion_global_anchor_orientation_error_exp", 0.5, 0.4),
    ("motion_body_pos", "motion_relative_body_position_error_exp", 1.0, 0.3),
    ("motion_body_ori", "motion_relative_body_orientation_error_exp", 1.0, 0.4),
    ("motion_body_lin_vel", "motion_global_body_linear_velocity_error_exp", 1.0, 1.0),
    ("motion_body_ang_vel", "motion_global_body_angular_velocity_error_exp", 1.0, 3.14),
)
TRACKED_BODIES = sm_offline.TRACKED
XI_NOMINAL = None  # filled by nominal_xi()


def nominal_xi(n: int) -> torch.Tensor:
    """38-d sim context of an unrandomised env: foot friction 1, no encoder bias / COM offset, sim2real scales 1, 1, 1, 0, 0."""
    xi = torch.zeros(n, 38)
    xi[:, 0] = 1.0
    xi[:, 33:36] = 1.0
    return xi


class _Ctx:
    """Stand-in for SimContext (the local env is not randomised)."""

    dim = 38

    def __init__(self, env, *a, **k):
        self.env = env

    def read(self):
        return nominal_xi(self.env.num_envs)


# ----------------------------------------------------------------------------- terrain entity stand-in (mjlab TerrainEntity logic)
def _proportional_counts(num_envs: int, proportions: np.ndarray) -> np.ndarray:
    n = len(proportions)
    counts, remaining = (np.ones(n, dtype=int), num_envs - n) if num_envs >= n else (np.zeros(n, dtype=int), num_envs)
    if remaining > 0:
        ideal = proportions * remaining
        floor = np.floor(ideal).astype(int)
        counts += floor
        left = remaining - floor.sum()
        if left > 0:
            counts[np.argsort(-(ideal - floor))[:left]] += 1
    return counts


class TerrainStandIn:
    def __init__(self, gen_cfg, terrain_origins: np.ndarray, num_envs: int, max_init_level: int | None):
        self.cfg = types.SimpleNamespace(terrain_generator=gen_cfg)
        self.terrain_origins = torch.as_tensor(terrain_origins, dtype=torch.float32)
        rows, cols = self.terrain_origins.shape[:2]
        self.max_terrain_level = rows
        top = rows - 1 if max_init_level is None else min(max_init_level, rows - 1)
        self.terrain_levels = torch.randint(0, top + 1, (num_envs,))
        p = np.array([s.proportion for s in gen_cfg.sub_terrains.values()], dtype=float)
        self.terrain_types = torch.repeat_interleave(torch.arange(cols), torch.from_numpy(_proportional_counts(num_envs, p / p.sum())))
        self.env_origins = self.terrain_origins[self.terrain_levels, self.terrain_types].clone()

    def update_env_origins(self, env_ids, move_up, move_down) -> None:
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level, torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level), torch.clip(self.terrain_levels[env_ids], 0))
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def place(self, levels: torch.Tensor, types_: torch.Tensor) -> None:
        """Evaluation: put every env on a chosen tile."""
        self.terrain_levels[:] = levels
        self.terrain_types[:] = types_
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]


# ----------------------------------------------------------------------------- robot stand-in (the parts of mjlab's Entity the variant reads / writes)
class _RobotData:
    pass


class RobotStandIn:
    def __init__(self, env):
        self.env = env
        m = env.model
        ids = [b for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("robot/")]
        self.body_ids = np.array(ids)
        self.body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b).split("/", 1)[1] for b in ids]
        self.data = _RobotData()
        n = env.num_envs
        c = env.contract
        self.data.default_joint_pos = torch.tensor(c["default_joint_pos"], dtype=torch.float32)[None].repeat(n, 1)
        self.data.soft_joint_pos_limits = torch.tensor(c["soft_joint_pos_limits"], dtype=torch.float32)[None].repeat(n, 1, 1)
        self.data.gravity_vec_w = torch.tensor([0.0, 0.0, -1.0])[None].repeat(n, 1)

    def find_bodies(self, names, preserve_order: bool = True):
        return [self.body_names.index(nm) for nm in names], list(names)

    def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids=None):
        for k, e in enumerate(env_ids.tolist()):
            d = self.env.datas[e]
            d.qpos[7:] = joint_pos[k].double().numpy()
            d.qvel[6:] = joint_vel[k].double().numpy()

    def write_root_state_to_sim(self, root_state, env_ids=None):
        """[pos 3, quat 4, lin vel (world) 3, ang vel (world) 3]; the free joint stores the angular velocity in the body frame."""
        ang_b = quat_apply_inverse(root_state[:, 3:7], root_state[:, 10:13])
        for k, e in enumerate(env_ids.tolist()):
            d = self.env.datas[e]
            d.qpos[:7] = root_state[k, :7].double().numpy()
            d.qvel[:3] = root_state[k, 7:10].double().numpy()
            d.qvel[3:6] = ang_b[k].double().numpy()

    def reset(self, env_ids=None):
        for e in (range(self.env.num_envs) if env_ids is None else env_ids.tolist()):
            d = self.env.datas[e]
            d.qacc_warmstart[:] = 0.0
            d.ctrl[:] = 0.0
            d.time = 0.0

    def refresh(self) -> None:
        env, m, n, nb = self.env, self.env.model, self.env.num_envs, len(self.body_ids)
        pos, quat = np.zeros((n, nb, 3), np.float32), np.zeros((n, nb, 4), np.float32)
        lin, ang = np.zeros((n, nb, 3), np.float32), np.zeros((n, nb, 3), np.float32)
        qpos, qvel = np.zeros((n, m.nq), np.float32), np.zeros((n, m.nv), np.float32)
        v6 = np.zeros(6)
        for e, d in enumerate(env.datas):
            pos[e], quat[e] = d.xpos[self.body_ids], d.xquat[self.body_ids]
            for k, b in enumerate(self.body_ids):
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, int(b), v6, 0)  # [ang, lin] at the link origin (XBODY; BODY would be the COM), world axes
                ang[e, k], lin[e, k] = v6[:3], v6[3:]
            qpos[e], qvel[e] = d.qpos, d.qvel
        dd = self.data
        dd.body_link_pos_w, dd.body_link_quat_w = torch.from_numpy(pos), torch.from_numpy(quat)
        dd.body_link_lin_vel_w, dd.body_link_ang_vel_w = torch.from_numpy(lin), torch.from_numpy(ang)
        dd.joint_pos, dd.joint_vel = torch.from_numpy(qpos[:, 7:].copy()), torch.from_numpy(qvel[:, 6:].copy())
        env.sim.data.qpos, env.sim.data.qvel = torch.from_numpy(qpos), torch.from_numpy(qvel)


# ----------------------------------------------------------------------------- the environment
class LocalEnv:
    def __init__(self, contract_dir: Path, npz_dir: Path, num_envs: int, terrain: TerrainArgs | None = None, clips: list[str] | None = None, max_init_level: int | None = 0,
                 episode_length_s: float = 10.0, reset_noise: bool = True, sampling_mode: str = "return", record: bool = False, capacity: int = 600, seed: int = 0,
                 terrain_clip_regex: str = r"^(walk|run|sprint)", command_kwargs: dict | None = None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.num_envs, self.device = num_envs, torch.device("cpu")
        self.contract = c = json.loads((Path(contract_dir) / "contract.json").read_text())
        self.terrain_args = terrain or TerrainArgs()
        gen_cfg = make_terrain_generator_cfg(self.terrain_args)
        self.model, origins, self.scene_report = LS.build_scene(Path(contract_dir), gen_cfg)
        self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]
        act_joint = [j.split("/")[-1] for j in c["actuator_joint_names"]]
        self.ctrl_of = np.array([act_joint.index(j) for j in c["action_joint_names"]])
        self.act_offset, self.act_scale = np.asarray(c["action_offset"]), np.asarray(c["action_scale"])
        self.decimation = int(c["decimation"])
        self.step_dt = float(c["control_dt"])
        self.cfg = types.SimpleNamespace(episode_length_s=episode_length_s)
        self.max_episode_length = int(np.ceil(episode_length_s / self.step_dt))
        self.sim = types.SimpleNamespace(mj_model=self.model, data=types.SimpleNamespace(qpos=None, qvel=None))
        self.terrain = TerrainStandIn(gen_cfg, origins, num_envs, max_init_level)
        self.robot = RobotStandIn(self)
        env = self

        class _Scene:
            terrain = env.terrain

            @property
            def env_origins(self):
                return env.terrain.env_origins

            def __getitem__(self, key):
                assert key == "robot", key
                return env.robot

        self.scene = _Scene()
        self.robot.refresh()
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.common_step_counter = 0
        self.action = torch.zeros(num_envs, 29)
        self.prev_action = torch.zeros(num_envs, 29)
        self.executed = torch.zeros(num_envs, 29)
        self.action_manager = types.SimpleNamespace(action=self.action, prev_action=self.prev_action, get_term=lambda name: types.SimpleNamespace(executed_raw=env.executed))
        self.termination_manager = types.SimpleNamespace(terminated=torch.zeros(num_envs, dtype=torch.bool), time_outs=torch.zeros(num_envs, dtype=torch.bool))
        self.reset_terminated = self.termination_manager.terminated
        files = sorted(str(p) for p in Path(npz_dir).glob("*.npz"))
        if clips:
            files = [f for f in files if Path(f).stem in clips]
        pose_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)}
        vel_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2), "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)}
        self.command = MultiClipMotionCommandCfg(
            npz_files=files, anchor_body_name="pelvis", root_body_name="pelvis", goal_contract="official", body_names=TRACKED_BODIES, entity_name="robot",
            pose_range=pose_range if reset_noise else {}, velocity_range=vel_range if reset_noise else {}, joint_position_range=(-0.1, 0.1) if reset_noise else (0.0, 0.0),
            sampling_mode=sampling_mode, clip_end_norm=True, library_fp16=False, terrain_task=True, terrain_clip_regex=terrain_clip_regex, **(command_kwargs or {}),
        ).build(self)
        self.command_manager = types.SimpleNamespace(get_term=lambda name: env.command)
        self.recorder = None
        if record:
            _rec_mod.SimContext = _Ctx
            rc = ReplayRecorderCfg(params={"capacity": capacity, "foot_geom_regex": c["foot_geom_regex"], "sim2real_dr": True})
            self.recorder = rc.func(rc, self)
        self.q0 = torch.tensor(c["default_joint_pos"], dtype=torch.float32)
        self.xi = nominal_xi(num_envs)
        self.obs_hist = None
        self.reset_buf = torch.zeros(num_envs, dtype=torch.bool)
        self.reward_terms: dict[str, torch.Tensor] = {}

    # -------------------------------------------------------------- pieces of the step
    def forward(self, env_ids=None) -> None:
        for e in (range(self.num_envs) if env_ids is None else env_ids.tolist()):
            mujoco.mj_forward(self.model, self.datas[e])
        self.robot.refresh()

    def state(self) -> torch.Tensor:
        """(N, 71) [qpos | qvel] relative to the state origin (StateIO convention)."""
        from supermimic.envs.state_io import state_origin

        s = torch.cat([self.sim.data.qpos, self.sim.data.qvel], -1).clone()
        s[:, 0:3] -= state_origin(self)
        return s

    def _observe(self, reset_ids: torch.Tensor | None) -> dict:
        obs = analytic_observation(self.state(), self.action, self.xi, self.q0)
        if self.obs_hist is None:
            self.obs_hist = obs[None].repeat(10, 1, 1)
        else:
            self.obs_hist = torch.cat([self.obs_hist[1:], obs[None]], 0)
            if reset_ids is not None and reset_ids.numel() > 0:
                self.obs_hist[:, reset_ids] = obs[reset_ids][None]  # a reset fills the history with the first observation
        cmd = self.command
        goal = torch.cat([cmd.goal, cmd.anchor_ori_window], -1)
        return {"policy": flatten_history_term_major(self.obs_hist), "goal": goal}

    def _reward(self) -> torch.Tensor:
        dt = self.step_dt
        terms = {name: dt * w * TRACK[fn](self, "motion", std) for name, fn, w, std in REWARD_TERMS}
        terms["action_rate_l2"] = dt * -0.1 * torch.sum(torch.square(self.action - self.prev_action), dim=1)
        q, lim = self.robot.data.joint_pos, self.robot.data.soft_joint_pos_limits
        terms["joint_limit"] = dt * -2.0 * (-(q - lim[..., 0]).clip(max=0.0) + (q - lim[..., 1]).clip(min=0.0)).sum(1)
        self.reward_terms = terms
        return sum(terms.values())

    def reset(self) -> dict:
        ids = torch.arange(self.num_envs)
        self.command.reset(ids)
        self.episode_length_buf[:] = 0
        self.forward()
        self.command.compute(0.0, ids)
        self.action.zero_()
        self.prev_action.zero_()
        self.obs_hist = None
        obs = self._observe(None)
        if self.recorder is not None:
            self.recorder.record_post_reset(ids)
        return obs

    def step(self, action: torch.Tensor):
        self.prev_action[:] = self.action
        self.action[:] = action
        self.executed[:] = action
        a = action.double().numpy()
        for e, d in enumerate(self.datas):
            d.ctrl[self.ctrl_of] = self.act_offset + self.act_scale * a[e]
            for _ in range(self.decimation):
                mujoco.mj_step(self.model, d)
        self.forward()  # mjlab evaluates termination / reward on quantities that lag by one physics substep; here they are fresh
        self.episode_length_buf += 1
        self.common_step_counter += 1
        cmd = self.command
        height, tilt = sm_term.bad_root_height(self, "motion", 0.25), sm_term.bad_root_tilt(self, "motion", 0.8)
        terminated = height | tilt
        time_outs = sm_term.clip_end(self, "motion") | (self.episode_length_buf >= self.max_episode_length)
        self.termination_manager.terminated = terminated
        self.reset_terminated = terminated
        self.termination_manager.time_outs = time_outs
        self.reset_buf = terminated | time_outs
        self.term_flags = {"root_height": height, "root_tilt": tilt}
        self.reward_buf = self._reward()
        ids = self.reset_buf.nonzero().squeeze(-1)
        if ids.numel() > 0:
            if self.recorder is not None:
                self.recorder.record_pre_reset(ids)
            cmd.reset(ids)
            self.episode_length_buf[ids] = 0
            self.action[ids] = 0.0  # mjlab's action manager clears the action history of a reset env
            self.prev_action[ids] = 0.0
        self.forward()
        dt = torch.full((self.num_envs,), self.step_dt)
        dt[ids] = 0.0
        cmd.compute(dt)
        obs = self._observe(ids)
        if self.recorder is not None:
            if ids.numel() > 0:
                self.recorder.record_post_reset(ids)
            self.recorder.record_post_step()
        return obs, self.reward_buf, terminated, time_outs

    # -------------------------------------------------------------- checks
    def max_penetration(self, env_ids=None) -> torch.Tensor:
        """(n,) deepest robot-terrain penetration (m, positive = inside) in the current contacts."""
        m = self.model
        tb = m.body("terrain").id
        out = []
        for e in (range(self.num_envs) if env_ids is None else env_ids.tolist()):
            d, worst = self.datas[e], 0.0
            for i in range(d.ncon):
                con = d.contact[i]
                if m.geom_bodyid[con.geom1] == tb or m.geom_bodyid[con.geom2] == tb:
                    worst = max(worst, -con.dist)
            out.append(worst)
        return torch.tensor(out)
