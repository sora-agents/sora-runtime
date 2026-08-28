"""Point ARE's file-system fallback at a local snapshot instead of the Hugging Face Hub.

ARE's ``Files`` app restores a tree of *empty placeholders* and backs them with
``hf://datasets/meta-agents-research-environments/gaia2_filesystem/demo_filesystem`` (294 files,
~247 MiB), copying real bytes down only when a file is read. But ``get_state()`` lists that tree
with sizes, and the wrapper answers each placeholder's size by asking the fallback — one
``paths-info`` request per file. Nothing is downloaded; it is ~294 round-trips of pure metadata,
about 66s, and it happens twice per run (~592 calls, ~135s): once while the oracle graph is built,
then again on the agent's first Observe, because ``set_fallback_root`` clears the stat registry and
re-marks every entry lazy.

The second one is not merely slow, it is a correctness bug. The ARE clock starts before the agent
takes its focus baseline, so those ~66s of scenario timeline elapse *during* the baseline read: any
event the scenario injects in that window is already in the baseline and can never diff as `added`,
so it never becomes a signal. The agent is simply blind to the head of its own timeline. It is
visible in a run log as a first plan prompt whose ``Calendar.state`` holds more events than the
scenario's initial state, next to "(none observed yet)".

Downloading all 247 MiB takes ~30s — less than half the time spent merely stat-ing it — so a local
mirror is strictly cheaper than the status quo, and after the first run it is free. Same bytes, same
tree, so a score is unaffected; only the timing changes, which is the point.

The root cause is upstream and small: ``FallbackFileSystem._get_fallback_stats_for_item`` calls
``HfFileSystem.info(path)`` bare, which defaults ``expand_info=True`` and so rejects the dircache
the cheap listing already populated (its ``last_commit`` is None). It reads only ``size``/``mode``,
both already cached. Until that is fixed upstream, redirecting the fallback is the whole fix.

Not in ``sora.adapters.are_sim``: a benchmark's data-staging is not runtime behaviour, and the core
CLI stays benchmark-agnostic. An agent pointed at ARE through ``sora run`` still gets stock ARE.
"""

from __future__ import annotations

import os
import sys

_REPO_ID = "meta-agents-research-environments/gaia2_filesystem"
_SUBDIR = "demo_filesystem"

# Pinned, not ``main``: a benchmark's inputs should not move under it between runs, and a resolved
# commit also lets a warm cache serve the snapshot with zero network calls. Override with
# GAIA2_FS_REVISION to track a newer upstream.
_REVISION = "132e26376f5e963bb59f64bcccdd02188cb08dee"

# Read at import time by ARE, and *bound as a default argument* on SandboxLocalFileSystem.__init__,
# so setting it after ARE is imported has no effect at all — silently, which is why the import order
# is checked rather than assumed.
_ARE_CONFIG_MODULE = "are.simulation.config"


def ensure_local_fallback_fs(announce: bool = True) -> str | None:
    """Download the fallback file system once and hand ARE the local path via ``DEMO_FS_PATH``.

    Returns the path in use, or None if the remote default was left in place. Never raises: a
    missing snapshot is a slow run, not a failed one, so any error degrades to stock ARE.

    Must be called **before** anything imports ARE.
    """
    if os.environ.get("SORA_GAIA2_LOCAL_FS") == "0":
        return None
    if os.environ.get("DEMO_FS_PATH"):
        # An explicit choice outranks ours, including someone pointing it at their own tree.
        if announce:
            print(f"file-system fallback: {os.environ['DEMO_FS_PATH']} (DEMO_FS_PATH, preset)")
        return os.environ["DEMO_FS_PATH"]

    if _ARE_CONFIG_MODULE in sys.modules:
        # Setting the variable now would be a no-op that looks like a fix, and the run would quietly
        # pay the ~135s and lose the head of its timeline. Say so instead.
        print(
            f"warning: {_ARE_CONFIG_MODULE} was imported before the file-system fallback was "
            "staged; using the remote default. Call ensure_local_fallback_fs() earlier.",
            file=sys.stderr,
        )
        return None

    try:
        from huggingface_hub import snapshot_download

        root = snapshot_download(
            repo_id=_REPO_ID,
            repo_type="dataset",
            revision=os.environ.get("GAIA2_FS_REVISION", _REVISION),
            allow_patterns=f"{_SUBDIR}/**",
        )
    except Exception as exc:  # noqa: BLE001 - any failure means "use the remote", never "stop"
        print(
            f"warning: could not stage the file-system fallback locally ({exc}); falling back to "
            "the Hub. Expect ~135s of extra startup and a blind window at the head of the "
            "timeline.",
            file=sys.stderr,
        )
        return None

    path = os.path.join(root, _SUBDIR)
    if not os.path.isdir(path):
        print(
            f"warning: staged snapshot has no {_SUBDIR}/ directory ({path}); falling back to "
            "the Hub.",
            file=sys.stderr,
        )
        return None

    os.environ["DEMO_FS_PATH"] = path
    if announce:
        # Disclosed in the run's own output like --scenario-duration and the verdict-parse note:
        # it changes the run's timing profile, and that has to survive the output being pasted
        # somewhere without the command line.
        print(f"file-system fallback: local snapshot @ {_REVISION[:12]} (no Hub round-trips)")
    return path
