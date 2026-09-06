# Gaia2 container harness (`gaia2-cli`)

Running S-ORA inside Meta's containerised Gaia2 harness — one container per scenario, holding the
event daemon, the ten app CLIs, the in-container judge, and the agent behind an HTTP adapter. This
is the path the *reported* numbers should come from: the [in-process harness](../README.md) diverges
from stock ARE in ways a published Gaia2 result cannot absorb, most consequentially on the clock.

It is a **walking skeleton**: the full contract runs end to end, and `emails`/`calendar` have
hand-picked observable properties. The other eight apps work on the mechanical fallback (see
*Observable properties* below), which is enough to run them and not enough to run them well.

## What the agent sees

Sibling harnesses (OpenClaw, Hermes, mini-swe-agent) hand their model a shell and let it type
`emails send-email --to ...`. S-ORA does not: at discovery the adapter runs `<binary> schema` — the
JSON description every app answers with — and turns each subcommand into a typed
`OperationSpecification`. The agent then invokes `send_email(recipients=[...], subject=...)` as a
named operation with real parameters, and `_Gaia2CliTool.invoke` builds the argv and runs the binary
with no shell in between.

Grading is unaffected: `log_action()` runs *inside* each app CLI, so an invocation is recorded in
`events.jsonl` however it was issued. Operations are named after the schema's `oracle_function` —
the callback name the judge matches the oracle on, and also the name the in-process adapter uses, so
a plan reads the same on both harnesses.

## Build and verify

The upstream base image must exist first:

```bash
cd <gaia2-cli>
docker build --platform linux/amd64 -f cli/Dockerfile -t localhost/gaia2-cli:local .
```

Then, from this repository:

```bash
make -C examples/gaia2/cli build GAIA2_CLI_DIR=<gaia2-cli>
make -C examples/gaia2/cli verify
```

`verify` checks that the ten CLI symlinks resolve, that `emails schema` / `calendar schema` parse
into manuals, and that `sora` imports inside the image.

> The images are **x86_64 only** — the base hardcodes an `x86_64-linux-gnu` libfaketime path — so
> `--platform linux/amd64` is pinned in the Makefile and an arm64 host runs them under emulation.
> That works (it is how this was developed); it is slower, and the scenario clock does not care.

## Run one scenario

```bash
docker run --rm -p 8090:8090 --platform linux/amd64 \
  -v "$PWD/examples/gaia2/scenarios/execution/<scenario>.json:/var/gaia2/custom_scenario.json:ro" \
  -e PROVIDER=openai -e MODEL=gpt-5.4-2026-03-05 -e API_KEY="$OPENAI_API_KEY" \
  -e BASE_URL=https://api.openai.com/v1 \
  localhost/gaia2-sora:latest
```

Add `-e GAIA2_JUDGE_MODEL=... -e GAIA2_JUDGE_PROVIDER=... -e GAIA2_JUDGE_API_KEY=...` to score it.
Under upstream's runner, pass `--runtime docker` and the image name; the image deliberately avoids
every substring the runner's profile detection matches (`openclaw`, `gaia2-oc`, `hermes`,
`gaia2-mini`, `mini-swe-agent`, `oracle`), so it falls through to the default profile — the one that
passes `PROVIDER`/`MODEL`/`API_KEY`/`BASE_URL`.

Useful inside a running container (its `PATH` is the agent sandbox, so pass a real one):

| Path | What |
|---|---|
| `/tmp/entrypoint.log` | The S-ORA worker's own log — the whole trajectory |
| `/tmp/gaia2-adapter.log` | HTTP adapter: turns in, responses out |
| `/tmp/gaia2-eventd.log` | The daemon: turn gating, environment events, the judge |
| `/var/gaia2/state/events.jsonl` | What the judge reads |

## Model selection comes from the environment

This one harness inverts the project's usual "`agent.yaml` is the only model selector" rule. The
runner sets `PROVIDER`/`MODEL`/`API_KEY`/`BASE_URL` per invocation so its own `--model` flag works
across a sweep; [`sora_worker.py`](sora_worker.py) reads them, writes a derived config to `/tmp`, and
calls `build_agent()` on that. Everything else still comes from [`agent.yaml`](agent.yaml).

## Observable properties

State is deliberately unreachable except through the binaries — the setuid wrapper exists so the
agent user cannot read the state files — so **this environment has no observable state to offer**.
The only way to see anything is to run a command, which is an operation, not an observation. The
manuals say so: no app declares an observable property, and nothing is polled.

That is a deliberate reversal. An earlier revision nominated one zero-argument read per app and
published its output *as* observed state on a one-second timer, which manufactures an affordance
the environment does not have, degenerates where no read represents the state (`calendar` yields
its tag list), and costs a subprocess per tool per second to obtain what the sibling harnesses get
by deciding to look. `polls:` is still supported for an ecosystem that genuinely offers an
observable — a WoT poll form is exactly that — and this one does not. See the design note,
[Perception tiers and judge-free replan](../../../docs/architecture/notes/perception-tiers-and-judge-free-replan.md).

The consequence for planning is that world change arrives only as an announcement, so the agent
reads the detail as a plan step. That is also why `agent.yaml` sets
`context_adaptation: replan_on_change`: shown an announcement and nothing else, a revalidation
judge can rarely do better than "maybe", so it would spend a model call to learn what the
change-gate already established.

## The clock: read, not preloaded

The harness fakes time with libfaketime — its daemon writes the scenario's simulated time to
`/tmp/faketime.rc`, and the app CLIs are preloaded so their timestamps come from there. The agent
process is **not** preloaded. It reads the same file through a `DomainClock`
(`_FaketimeClock` in [`gaia2_cli.py`](../../../src/sora/adapters/gaia2_cli.py)), so the workspace
reports scenario time while the process keeps a real wall clock.

That split is not tidiness. Preloading the worker does give `datetime.now()` scenario time, and it
also hands that clock to OpenSSL, which then rejects the model provider's certificate as *not yet
valid* on any scenario set in the past — every model call fails, and the run looks like a planning
failure. It would also measure the HTTP timeouts, the poll period and the inference watchdog on a
clock that is not wall time. Reading the file keeps domain time simulated and infrastructure time
real, which is the distinction `DomainClock` exists to draw.

Note what this means for the numbers: simulated time advances ~1:1 with wall time, *including*
while the agent is generating. Nothing freezes the clock while it thinks.

## Two deliberate differences from the sibling harnesses

**Environment notifications arrive as signals, not as user messages.** The harness posts
`/notify`, `/send_user_message` and `/send_notifications` with identical bodies, so the route is the
only thing separating "the user said something" from "the world changed".
[`gaia2_adapter.py`](gaia2_adapter.py) claims `/send_notifications` and forwards the text to the
worker, which pushes it into the signal sink of the tool it names as an `env_notification` signal.
Delivering it as a turn instead — what the sibling harnesses do — would attribute it to someone who
did not speak.

This is the agent's *only* channel for world change, and it announces rather than describes: it is
rendered prose with no ids, exists only for environment-initiated actions that have a formatter
upstream (`contacts` and `cloud-drive` have none), and never fires for the agent's own writes. An
earlier revision dropped it and kept only a timer poll, which left the agent perceiving strictly
less than the sibling harnesses on exactly the scenarios that turn on environment-initiated change.

The manual declares `env_notification` only for an app that has a formatter, so a plan can author a
`watch` against a signal that can actually fire and cannot wait forever on one that cannot. Its
payload schema states the tier explicitly — `{app, text}`, and no `changes` — so a consumer cannot
read structure off prose that carries none.

Upstream's own agent prompt tells the sibling harnesses the same thing ("**Do NOT poll or actively
check** … the system will deliver them to you"), and they respond to a notification by running a
read command inside the turn. The tier-2 position here is not a handicap relative to them; it is
the same position.

**`send_message_to_user` is the turn boundary, and it is emitted even when empty.** The adapter
turns each final response into the synthetic `AgentUserInterface` event the daemon watches for.
Without one the run ends `error: no turn boundary detected`; more than `nb_turns * 3` of them aborts
the scenario.

## Known gaps

- No `completion_signal` on any operation, so the `blocked` machinery is unexercised. Not a
  regression — the in-process adapter sets none either — and not closable from the native
  description: every notification formatter upstream is registered on a `hidden=True` environment
  function, and `schema` omits hidden commands.
- Per-capability settings (`context_adaptation` differs by capability — `search`/`execution` need
  no reconsideration at all, `adaptability`/`time` do) have no host→
  container channel: the launcher forwards a fixed key list, and the split is encoded in the export
  *directory* name rather than in the scenario JSON, so the worker cannot recover it. The natural
  seam is one thin image tag per capability, each shipping its own `agent.yaml` — matching the
  runner's own one-split-per-invocation unit. Not needed to run a single scenario.
- `contacts` and `cloud-drive` register no notification formatter upstream, so a change to either
  is imperceptible until something reads them. The sibling harnesses share that blind spot exactly;
  closing it would need an upstream formatter, not an adapter change.
