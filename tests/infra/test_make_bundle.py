"""make_bundle.py must turn a frozen config into a run script whose training command carries every frozen flag."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _bundle(tmp_path, config, extra=()):
    code = tmp_path / "code.tar.gz"
    data = tmp_path / "data.tar.gz"
    code.write_bytes(b"code")
    data.write_bytes(b"data")
    sd = tmp_path / "sess"
    subprocess.run([sys.executable, "-I", str(ROOT / "infra/runpod/make_bundle.py"), str(sd), "--config", str(ROOT / "configs/cuboi-g1" / config),
                    "--code-tar", str(code), "--data-tar", str(data), *extra], check=True, capture_output=True, text=True)
    return sd, (sd / "bundle/run_remote.sh").read_text(), json.loads((sd / "bundle.json").read_text())


def _train_lines(script):
    return [ln for ln in script.splitlines() if "sm-train" in ln and "--run-name $RUN" in ln]


def test_anchor_flag_reaches_the_training_command(tmp_path):
    chk = tmp_path / "check_anchor_obs.py"
    chk.write_text("pass\n")
    sd, script, meta = _bundle(tmp_path, "paper-contract-h32-lafan40-anchor-seed0-v1.json", ["--extra-script", str(chk)])
    (line,) = _train_lines(script)
    assert "--env.anchor-obs" in line and "--seed 0" in line and "--env.num-envs 4096" in line
    assert meta["anchor_obs"] is True
    assert '"anchor_obs": true, "goal_dim": 689' in script  # runtime assertion against config.json
    assert "check_anchor_obs.py" in script and "bash -n" not in script
    assert subprocess.run(["bash", "-n", str(sd / "bundle/run_remote.sh")]).returncode == 0


def test_proprio_config_has_no_anchor_and_no_anchor_check(tmp_path):
    _, script, meta = _bundle(tmp_path, "paper-contract-h32-lafan40-seed0-v1.json")
    (line,) = _train_lines(script)
    assert "--env.anchor-obs" not in line
    assert meta["anchor_obs"] is False and "check-anchor-obs" not in script
    assert '"anchor_obs": false, "goal_dim": 680' in script


def test_warm_restart_adds_resume_and_session_suffix(tmp_path):
    parent = tmp_path / "ckpt_latest.pt"
    parent.write_bytes(b"ckpt")
    _, script, meta = _bundle(tmp_path, "paper-contract-h32-4clip-seed0-v1.json", ["--parent-ckpt", str(parent), "--session-index", "3"])
    (line,) = _train_lines(script)
    assert "--resume $B/parent_ckpt.pt" in line and "--data.clips walk1_subject1 dance1_subject1 jumps1_subject1 fight1_subject2" in line
    assert meta["run_name"].endswith("-s3") and meta["parent_ckpt_sha256"]
    assert "initial-ckpt" not in script
    assert meta["deviation_from_frozen_argv"] == ["max_hours 1.15 (frozen 1.2)"]


def test_sha256sums_cover_every_bundle_file(tmp_path):
    sd, _, meta = _bundle(tmp_path, "paper-contract-h32-lafan40-seed0-v1.json")
    listed = {ln.split()[1] for ln in (sd / "bundle/SHA256SUMS").read_text().splitlines()}
    present = {p.name for p in (sd / "bundle").iterdir()} - {"SHA256SUMS"}
    assert listed == present == set(meta["files"])
