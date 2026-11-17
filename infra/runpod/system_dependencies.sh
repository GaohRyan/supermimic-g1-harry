#!/usr/bin/env bash
# Run on the Ubuntu training Pod before installing Python packages.
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo "Linux training Pod required" >&2; exit 1; }
if ! python3 - <<'PY'
import ctypes
for library in ('libEGL.so.1', 'libGL.so.1', 'libglib-2.0.so.0'):
    ctypes.CDLL(library)
PY
then
  [[ "$(id -u)" == 0 ]] || { echo "Missing graphics libraries; install libegl1 libgl1 libglib2.0-0 as root" >&2; exit 1; }
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends libegl1 libgl1 libglib2.0-0
fi
dpkg-query -W -f='${binary:Package}\t${Version}\n' libegl1 libgl1 libglib2.0-0t64
