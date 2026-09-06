#!/bin/bash
# Runtime scenario init + Gaia2 infrastructure startup, then drop to the agent user.
#
# Adapted from the sibling harnesses' init script; the launcher overrides the image entrypoint with
# `bash /opt/gaia2-init-entrypoint.sh`, so this path is fixed by the harness. Two differences from
# theirs: there is no AGENTS.md rendering step (operations are typed, so there is no shell prompt
# to render into), and the environment handed to the worker carries the SORA_* keys instead of the
# MINI_* ones. The tool-pruning step is kept as-is — it is what makes the adapter's discovery
# scenario-scoped.
#
# Privilege separation:
#   gaia2 user: gaia2-eventd + the HTTP adapter + state files + events.jsonl
#   agent user: sora_worker.py, reaching the environment only through the setuid CLI wrappers

set -eo pipefail

export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

CUSTOM_SCENARIO="${GAIA2_SCENARIO:-/var/gaia2/custom_scenario.json}"
ADAPTER_PORT="${GAIA2_ADAPTER_PORT:-8090}"

# ── 1. Initialise Gaia2 state ───────────────────────────────────────────────────────────────────
if [ ! -f "$CUSTOM_SCENARIO" ]; then
    echo "[gaia2-init] no scenario at $CUSTOM_SCENARIO, skipping init" >&2
else
    echo "[gaia2-init] initialising from $CUSTOM_SCENARIO ..." >&2
    rm -rf /var/gaia2/state/*
    gaia2-init --scenario "$CUSTOM_SCENARIO" \
              --state-dir /var/gaia2/state \
              ${GAIA2_FS_BACKING_DIR:+--fs-backing-dir "$GAIA2_FS_BACKING_DIR"}
    chown -R gaia2:gaia2 /var/gaia2/state
    # The agent must not read the scenario (ground truth) or the state files directly.
    chmod 700 /var/gaia2
    chmod -R go-rwx /var/gaia2/state

    if [ -z "$FAKETIME" ]; then
        FAKETIME=$(python3 -c "
import json
from datetime import datetime, timezone
d = json.load(open('$CUSTOM_SCENARIO'))
st = d.get('metadata', {}).get('definition', {}).get('start_time', d.get('start_time', 0))
if st and float(st) > 0:
    print(datetime.fromtimestamp(float(st), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
" 2>/dev/null)
        if [ -n "$FAKETIME" ]; then
            export FAKETIME
            echo "[gaia2-init] FAKETIME from scenario: $FAKETIME" >&2
        fi
    fi

    # Remove the CLI symlinks for apps this scenario does not use. The adapter discovers whatever
    # survives, so this is what scopes the agent's tool set to the scenario.
    _REMOVE=$(python3 -c "
from gaia2_cli.app_registry import resolve_scenario_tools
r = resolve_scenario_tools('$CUSTOM_SCENARIO')
print(' '.join(r.get('remove', [])))
" 2>/dev/null)
    if [ -n "$_REMOVE" ]; then
        for cmd in $_REMOVE; do rm -f "/home/agent/bin/$cmd"; done
        echo "[gaia2-init] removed unused CLI symlinks: $_REMOVE" >&2
    fi
    echo "[gaia2-init] done" >&2
fi

# ── 2. gaia2-eventd, as the gaia2 user ──────────────────────────────────────────────────────────
if [ "${GAIA2_DAEMON_DISABLE:-0}" != "1" ] && [ -f "$CUSTOM_SCENARIO" ]; then
    touch /tmp/gaia2-eventd.log && chown gaia2:gaia2 /tmp/gaia2-eventd.log
    if [ -n "$FAKETIME" ]; then echo "$FAKETIME" > /tmp/faketime.rc; else touch /tmp/faketime.rc; fi
    chown gaia2:gaia2 /tmp/faketime.rc
    chmod 644 /tmp/faketime.rc

    echo "[gaia2-init] starting gaia2-eventd ..." >&2
    su -s /usr/bin/bash gaia2 -c "
        export PATH=/usr/local/bin:/usr/bin:/bin
        export GAIA2_STATE_DIR=/var/gaia2/state
        ${GAIA2_JUDGE_FINAL_TURN:+export GAIA2_JUDGE_FINAL_TURN='$GAIA2_JUDGE_FINAL_TURN'}
        /usr/local/bin/gaia2-eventd \
            --scenario '$CUSTOM_SCENARIO' \
            --state-dir /var/gaia2/state \
            --notify-url 'http://127.0.0.1:$ADAPTER_PORT' \
            --poll-interval 0.5 \
            ${FAKETIME:+--faketime-path /tmp/faketime.rc} \
            ${GAIA2_NOTIFICATION_MODE:+--notification-mode $GAIA2_NOTIFICATION_MODE} \
            ${GAIA2_JUDGE_MODEL:+--judge-model $GAIA2_JUDGE_MODEL} \
            ${GAIA2_JUDGE_PROVIDER:+--judge-provider $GAIA2_JUDGE_PROVIDER} \
            ${GAIA2_JUDGE_BASE_URL:+--judge-base-url $GAIA2_JUDGE_BASE_URL} \
            ${GAIA2_JUDGE_API_KEY:+--judge-api-key '$GAIA2_JUDGE_API_KEY'} \
            ${GAIA2_JUDGE_PROMPT_VERSION:+--judge-prompt-version $GAIA2_JUDGE_PROMPT_VERSION} \
            ${GAIA2_JUDGE_EXTRA_BODY:+--judge-extra-body '$GAIA2_JUDGE_EXTRA_BODY'} \
            ${GAIA2_TIME_SPEED:+--time-speed $GAIA2_TIME_SPEED} \
            ${GAIA2_IDLE_TIMEOUT:+--idle-timeout $GAIA2_IDLE_TIMEOUT} \
            >> /tmp/gaia2-eventd.log 2>&1 &
    "
    echo "[gaia2-init] daemon launched (log: /tmp/gaia2-eventd.log)" >&2
else
    echo "[gaia2-init] gaia2-eventd skipped (disable=${GAIA2_DAEMON_DISABLE:-0})" >&2
fi

# ── 2b. The HTTP adapter, as the gaia2 user ─────────────────────────────────────────────────────
if [ -f "$CUSTOM_SCENARIO" ]; then
    touch /tmp/gaia2-adapter.log && chown gaia2:gaia2 /tmp/gaia2-adapter.log
    echo "[gaia2-init] starting the HTTP adapter ..." >&2
    su -s /usr/bin/bash gaia2 -c "
        export PATH=/usr/local/bin:/usr/bin:/bin
        export GAIA2_STATE_DIR=/var/gaia2/state
        export GAIA2_ADAPTER_PORT='$ADAPTER_PORT'
        PYTHONUNBUFFERED=1 /usr/local/bin/python3 /opt/gaia2_adapter.py \
            >> /tmp/gaia2-adapter.log 2>&1 &
    "
    echo "[gaia2-init] adapter launched (log: /tmp/gaia2-adapter.log)" >&2
fi

# ── 3. Drop to the agent user ───────────────────────────────────────────────────────────────────
# Via a file rather than inline, so an API key containing shell metacharacters survives.
ENV_FILE=/tmp/sora-env.sh
cat > "$ENV_FILE" << 'STATIC'
export HOME=/home/agent
export PATH=/home/agent/bin
export GAIA2_STATE_DIR=/var/gaia2/state
export DONT_FAKE_MONOTONIC=1
STATIC

for var in \
    PROVIDER MODEL API_KEY BASE_URL THINKING MAX_TOKENS \
    ANTHROPIC_API_KEY OPENAI_API_KEY OPENROUTER_API_KEY \
    SORA_AGENT_CONFIG SORA_LOG_LEVEL SORA_WORKER_SOCK \
    no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY \
    FAKETIME GAIA2_TRACE_FILE \
; do
    val="${!var:-}"
    [ -z "$val" ] && continue
    printf 'export %s=%q\n' "$var" "$val" >> "$ENV_FILE"
done

chown agent:agent "$ENV_FILE"
chmod 600 "$ENV_FILE"
exec su -s /usr/bin/bash agent -c "source $ENV_FILE && exec /usr/bin/bash /opt/entrypoint.sh"
