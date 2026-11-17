#!/usr/bin/env python3
"""Stop one explicitly owned Pod at a fixed deadline; never delete resources.

Run independently on the local machine and inside the Pod. Pod mode uses only
RunPod's automatically injected self-scoped key. Credentials never enter output.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time


def graphql(key: str, query: str) -> dict:
    config = 'url = "https://api.runpod.io/graphql"\n'
    config += 'header = "Content-Type: application/json"\n'
    config += 'header = ' + json.dumps('Authorization: Bearer ' + key) + '\n'
    config += 'data = ' + json.dumps(json.dumps({'query': query})) + '\n'
    result = subprocess.run(['curl', '--disable', '--silent', '--show-error', '--fail',
                             '--max-time', '20', '--config', '-'], input=config,
                            text=True, capture_output=True, timeout=25)
    if result.returncode:
        raise RuntimeError(f'RunPod request failed: exit {result.returncode}')
    data = json.loads(result.stdout)
    if data.get('errors') or not data.get('data'):
        raise RuntimeError('RunPod rejected the scoped request')
    return data['data']


def _site(name: str, default: str = '') -> str:
    """Account-specific setting: environment variable RUNPOD_<NAME>, else .secrets/runpod-site.json in the repository."""
    v = os.environ.get('RUNPOD_' + name.upper())
    if v:
        return v
    try:
        p = Path(__file__).resolve().parents[2] / '.secrets/runpod-site.json'
        return str(json.loads(p.read_text()).get(name, default))
    except (OSError, ValueError, IndexError):
        return default


def credential(mode: str) -> str:
    if mode == 'local':
        key = Path(_site('api_key_file', '~/.runpod/api.key')).expanduser().read_text().strip()
    else:
        key = os.environ.get('RUNPOD_API_KEY', '')
        if not key:
            # SSH may not inherit the container's PID1 environment.
            environment = Path('/proc/1/environ').read_bytes().split(b'\0')
            key = next((x.split(b'=', 1)[1].decode() for x in environment
                        if x.startswith(b'RUNPOD_API_KEY=')), '')
    if not key or '\n' in key or '\r' in key:
        raise ValueError('Required RunPod credential is unavailable')
    return key


def validate_session(session: dict, mode: str) -> tuple[str, float]:
    pod = session.get('pod_id', '')
    if not re.fullmatch(r'[a-z0-9]{8,32}', pod) or session.get('created_for_this_task') is not True:
        raise ValueError('Only the recorded new task Pod may be stopped')
    deadline = datetime.fromisoformat(session['stop_deadline_utc']).timestamp()
    created = datetime.fromisoformat(session['created_utc']).timestamp()
    if deadline <= created or deadline - created > 7200:
        raise ValueError('Deadline must be within two hours of creation')
    if not 0 < session['authorized_total_budget_usd'] <= 15:
        raise ValueError('Invalid authorized budget')
    if mode == 'pod':
        own_id = os.environ.get('RUNPOD_POD_ID', '')
        if not own_id:
            own_id = next((x.split(b'=', 1)[1].decode() for x in Path('/proc/1/environ').read_bytes().split(b'\0')
                           if x.startswith(b'RUNPOD_POD_ID=')), '')
        if pod != own_id:
            raise ValueError('Pod-scoped watchdog may stop only itself')
    return pod, deadline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--mode', choices=['local', 'pod'], required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    session = json.loads(args.session.read_text())
    pod, deadline = validate_session(session, args.mode)
    key = credential(args.mode)
    query = 'query { pod(input:{podId:' + json.dumps(pod) + '}) { id desiredStatus } }'
    mutation = 'mutation { podStop(input:{podId:' + json.dumps(pod) + '}) { id desiredStatus } }'
    log_path = args.session.with_name('watchdog-' + args.mode + '.jsonl')

    def log(event, **fields):
        record = {'utc': datetime.now(timezone.utc).isoformat(), 'event': event, 'pod_id': pod, **fields}
        with log_path.open('a') as out:
            out.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)

    observed = graphql(key, query).get('pod')
    if not observed or observed.get('id') != pod:
        raise ValueError('Could not verify watchdog target')
    log('authentication_verified', mode=args.mode, deadline_utc=session['stop_deadline_utc'])
    if args.check:
        return
    while True:
        try:
            state = graphql(key, query).get('pod')
            if not state or state.get('desiredStatus') == 'EXITED':
                log('already_stopped_or_removed')
                return
            if time.time() >= deadline:
                stopped = graphql(key, mutation).get('podStop')
                log('deadline_stop_requested', desired_status=stopped.get('desiredStatus') if stopped else None)
            time.sleep(15 if time.time() >= deadline else min(60, max(1, deadline - time.time())))
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            log('watchdog_request_failed', error_type=type(exc).__name__)
            time.sleep(15)


if __name__ == '__main__':
    main()
