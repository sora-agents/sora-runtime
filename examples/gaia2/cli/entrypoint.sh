#!/bin/bash
# Starts the S-ORA worker. The Gaia2 daemon and the HTTP adapter are already running as the gaia2
# user; this script owns only the worker process, which runs as `agent`.
set -o pipefail

LOG=/tmp/entrypoint.log
# Everything here is absolute-pathed on purpose: gaia2-init-entrypoint.sh drops us to the `agent`
# user with PATH=/home/agent/bin — the ten app symlinks and nothing else — so a bare `date` or
# `tee` does not resolve.
# Writes to the log only; the `tail -f` below is the single path to stdout, so teeing here
# as well would print every line twice.
log() { echo "[$(/usr/bin/date +%H:%M:%S)] $*" >> "$LOG"; }

log "=== entrypoint start (s-ora) ==="
log "user: $(/usr/bin/id -un), PATH: $PATH, state: ${GAIA2_STATE_DIR:-unset}"

# The worker deliberately runs WITHOUT libfaketime, unlike the app CLIs and the sibling
# harnesses' shell wrappers. Preloading it does make datetime.now() return scenario time — and it
# also makes OpenSSL validate the model provider's certificate against that clock, so every
# outbound call fails "certificate is not yet valid" for any scenario set in the past. The agent
# gets scenario time from the workspace's DomainClock instead, which reads the daemon's own
# /tmp/faketime.rc; see _FaketimeClock in sora/adapters/gaia2_cli.py.
[ -f /tmp/faketime.rc ] && log "scenario clock: $(/usr/bin/cat /tmp/faketime.rc) (read by the adapter, not preloaded)"

PYTHONUNBUFFERED=1 /usr/local/bin/python3 /opt/sora_worker.py >> $LOG 2>&1 &
WORKER_PID=$!
log "worker PID: $WORKER_PID"

# Mirror the log to stdout so the trajectory is visible from `docker run` — a container started
# with --rm takes /tmp/entrypoint.log with it when it exits. Piping python through tee instead
# would make $! the tee's PID and break both the liveness check and the exit code below.
/usr/bin/tail -n +1 -f $LOG &
TAIL_PID=$!

/usr/bin/sleep 3
if ! kill -0 $WORKER_PID 2>/dev/null; then
    log "ERROR: the S-ORA worker exited immediately"
    /usr/bin/tail -40 $LOG 2>/dev/null || true
    exit 1
fi
log "worker running"

# shellcheck disable=SC2317  # invoked via trap
cleanup() {
    log "shutting down..."
    kill "$WORKER_PID" 2>/dev/null || true
    wait "$WORKER_PID" 2>/dev/null || true
    kill "$TAIL_PID" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

wait $WORKER_PID 2>/dev/null
EXIT_CODE=$?
log "worker (PID $WORKER_PID) exited with code $EXIT_CODE"
exit $EXIT_CODE
