"""Depth-counted ARE clock used by the charged Gaia2 harness.

ARE's stock clock has a single pause bit.  That is insufficient for S-ORA, where two activities
may have model calls in flight together: the first completion resumes the world underneath the
second one, and the later completion can then subtract the wrong wall interval.  The stock
``time_passed`` also tests the saved value by truthiness, so a pause begun at exactly ``t == 0`` is
not frozen at all.

These subclasses are deliberately harness-local.  The runtime does not prescribe simulated time;
the experiment does.  Generation pauses are tokenized so completions may arrive in any order, and
their non-negative charges are summed individually.  Ordinary ARE pauses remain a separate depth
used by the online judge, which lets the runner's short watchdog stay judge-specific.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from are.simulation.environment import Environment
from are.simulation.time_manager import TimeManager
from are.simulation.types import EnvironmentState


class ChargedTimeManager(TimeManager):  # type: ignore[misc]  # ARE is untyped
    """ARE ``TimeManager`` with a thread-safe, depth-counted frozen interval."""

    def __init__(self, *, wall_time: Callable[[], float] = time.time) -> None:
        self._wall_time = wall_time
        self._pause_lock = threading.RLock()
        self._pause_depth = 0
        super().__init__()
        self.pause_real_start_time: float | None
        self.pause_passed_time: float | None
        # ``TimeManager.__init__`` samples ``time.time`` directly. Reinitialize through our
        # override so an injected wall clock is coherent even before a caller explicitly resets.
        self.reset()

    @property
    def pause_depth(self) -> int:
        with self._pause_lock:
            return self._pause_depth

    def reset(self, start_time: float | None = None) -> None:
        current_time = self._wall_time()
        self.start_time = current_time if start_time is None else start_time
        self.real_start_time = current_time
        self.offset = 0.0
        self.is_paused = False
        self.pause_real_start_time = None
        self.pause_passed_time = None
        self.pause_offset = 0.0
        self._pause_depth = 0

    def real_time_passed(self) -> float:
        if self.real_start_time is None:
            raise Exception("real_start_time cannot be null")
        return self._wall_time() - self.real_start_time

    def time_passed(self) -> float:
        if self.real_start_time is None:
            raise Exception("real_start_time cannot be null")
        with self._pause_lock:
            # Explicit None check: 0.0 is the most important valid saved value here.
            if self._pause_depth and self.pause_passed_time is not None:
                return self.pause_passed_time + self.pause_offset
            return self.real_time_passed() + self.offset

    def pause(self) -> None:
        with self._pause_lock:
            if self._pause_depth == 0:
                self.pause_real_start_time = self._wall_time()
                self.pause_passed_time = self.real_time_passed() + self.offset
                self.pause_offset = 0.0
                self.is_paused = True
            self._pause_depth += 1

    def resume(self) -> None:
        with self._pause_lock:
            if self._pause_depth == 0:
                return
            self._pause_depth -= 1
            if self._pause_depth:
                return
            assert self.pause_real_start_time is not None
            pause_duration = max(0.0, self._wall_time() - self.pause_real_start_time)
            self.offset -= pause_duration
            self.offset += self.pause_offset
            self.pause_offset = 0.0
            self.is_paused = False
            self.pause_passed_time = None
            self.pause_real_start_time = None

    def add_offset(self, offset: float) -> None:
        # A malformed duration must never rewind the scenario.  Zero is intentionally retained:
        # the event loop's ordinary one-second tick adds exactly zero for the shipped scenarios.
        value = max(0.0, float(offset))
        with self._pause_lock:
            if self._pause_depth:
                self.pause_offset += value
            else:
                self.offset += value


class ChargedEnvironment(Environment):  # type: ignore[misc]  # ARE is untyped
    """ARE ``Environment`` with independent judge and tokenized generation pauses."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Environment constructs its clock internally. Replace it before apps are registered and
        # rebind the already-created notification system, the only object initialized with it.
        clock = ChargedTimeManager()
        clock.reset(start_time=self.start_time)
        self.time_manager = clock
        self.current_time = clock.time()
        self.notification_system.initialize(clock)
        self._charged_pause_lock = threading.RLock()
        self._judge_pause_depth = 0
        self._generation_tokens: set[int] = set()
        self._generation_charge_starts: dict[int, float] = {}
        self._generation_parallel_frontier: float | None = None
        self._next_generation_token = 0

    @property
    def judge_paused(self) -> bool:
        with self._charged_pause_lock:
            return self._judge_pause_depth > 0

    @property
    def generation_pause_depth(self) -> int:
        with self._charged_pause_lock:
            return len(self._generation_tokens)

    def _pause_once(self) -> None:
        self.pause_event.set()
        self.time_manager.pause()
        self.state = EnvironmentState.PAUSED

    def _resume_once(self, offset: float) -> None:
        self.time_manager.add_offset(max(0.0, float(offset)))
        self.time_manager.resume()
        if self.time_manager.pause_depth:
            return
        self.pause_event.clear()
        self.state = EnvironmentState.RUNNING
        self._catch_up_tick()

    def _catch_up_tick(self) -> None:
        """Deliver everything the frozen interval skipped, synchronously, before returning.

        ARE's loop only reaches ``tick`` after its paused wait *and* its one-second sleep, so
        without this the next call can re-freeze the world first and the skipped events wait for
        the freeze after that.  Lateness then accumulates across a call chain and depends on host
        thread scheduling rather than on the charge.  Stock ``resume_with_offset`` ticks here for
        the same reason; the charged clock needs it more, because a reaction is a chain of calls
        with only a decision cycle between them.

        A dispatch failure is logged rather than raised: this runs inside the model client's
        ``finally``, where propagating would turn a successful call into an inference error, and
        the event loop's own next tick re-processes whatever this one did not.
        """
        try:
            self.tick()
        except Exception as exc:  # pragma: no cover - defensive; loop retries within ~1s
            self.log_debug(f"catch-up tick after resume failed: {exc!r}")

    def pause(self) -> None:
        """Depth-counted ordinary pause, used by ARE's online judge."""
        with self._charged_pause_lock:
            if self.state not in (EnvironmentState.RUNNING, EnvironmentState.PAUSED):
                self.log_debug("Attempt to pause when environment is not running.")
                return
            self._judge_pause_depth += 1
            self._pause_once()

    def resume(self) -> None:
        with self._charged_pause_lock:
            if self._judge_pause_depth == 0:
                return
            self._judge_pause_depth -= 1
            self._resume_once(0.0)

    def resume_with_offset(self, offset: float) -> None:
        """Compatibility path for non-tokenized callers; still depth-counted and non-negative."""
        with self._charged_pause_lock:
            if self._judge_pause_depth == 0:
                return
            self._judge_pause_depth -= 1
            self._resume_once(offset)

    def pause_generation(self) -> int:
        """Freeze for one physical model round-trip and return its completion token."""
        with self._charged_pause_lock:
            if self.state not in (EnvironmentState.RUNNING, EnvironmentState.PAUSED):
                # A judge rejection stops ARE directly while the agent can still have one cycle
                # in flight. Token zero is a no-op sentinel: teardown must not manufacture a new
                # inference error unrelated to the rejection that actually stopped the world.
                return 0
            if not self._generation_tokens:
                # Keep the sensitivity on its own axis. ``pause_offset`` is the trajectory's
                # serialized sum, so using it here would make the parallel counterfactual inherit
                # the policy it is meant to compare against.
                self._generation_parallel_frontier = float(self.time_manager.time_passed())
            self._next_generation_token += 1
            token = self._next_generation_token
            self._generation_tokens.add(token)
            assert self._generation_parallel_frontier is not None
            self._generation_charge_starts[token] = self._generation_parallel_frontier
            self._pause_once()
            return token

    def resume_generation(self, token: int, offset: float) -> None:
        """Apply one call's charge; only the final outstanding completion resumes the world."""
        with self._charged_pause_lock:
            if token not in self._generation_tokens:
                return
            self._generation_tokens.remove(token)
            value = max(0.0, float(offset))
            charged_start = self._generation_charge_starts.pop(token)
            assert self._generation_parallel_frontier is not None
            self._generation_parallel_frontier = max(
                self._generation_parallel_frontier, charged_start + value
            )
            self._resume_once(value)
            if not self._generation_tokens:
                self._generation_parallel_frontier = None

    def generation_charge_time(self, token: int) -> float:
        """Return one crossing's start on the online parallel-charge sensitivity axis."""
        with self._charged_pause_lock:
            charged_start = self._generation_charge_starts.get(token)
            if charged_start is not None:
                return charged_start
            return float(self.time_manager.time_passed())

    def stop(self, final_state: EnvironmentState = EnvironmentState.STOPPED) -> None:
        """Stop even while a judge or generation holds the clock frozen.

        ARE's event loop waits only on ``pause_event`` inside its paused loop; it does not also
        inspect ``stop_event`` there. A watchdog that merely called the stock ``stop`` could
        therefore leave the daemon thread parked forever.
        """
        with self._charged_pause_lock:
            self._judge_pause_depth = 0
            self._generation_tokens.clear()
            self._generation_charge_starts.clear()
            self._generation_parallel_frontier = None
            while self.time_manager.pause_depth:
                self.time_manager.resume()
            self.pause_event.clear()
            # Keep stop and pause admission atomic. ARE's paused loop does not inspect stop_event,
            # so admitting a generation token after the drain but before stock stop would park it.
            super().stop(final_state=final_state)
