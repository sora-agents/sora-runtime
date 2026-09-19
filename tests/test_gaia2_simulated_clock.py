from __future__ import annotations

import pytest

pytest.importorskip("are.simulation.environment")

from are.simulation.environment import EnvironmentConfig  # noqa: E402
from are.simulation.types import EnvironmentState, Event  # noqa: E402
from examples.gaia2.simulated_clock import (  # noqa: E402
    ChargedEnvironment,
    ChargedTimeManager,
)


class _Wall:
    now = 0.0

    def __call__(self) -> float:
        return self.now


def _environment(wall: _Wall) -> ChargedEnvironment:
    env = ChargedEnvironment(config=EnvironmentConfig(start_time=0, verbose=False))
    clock = ChargedTimeManager(wall_time=wall)
    clock.reset(start_time=0)
    env.time_manager = clock
    env.state = EnvironmentState.RUNNING
    return env


def test_pause_at_zero_is_frozen() -> None:
    wall = _Wall()
    clock = ChargedTimeManager(wall_time=wall)

    clock.pause()
    wall.now = 9.0

    assert clock.time_passed() == 0.0


def test_nested_generation_pauses_sum_charges_in_reverse_completion_order() -> None:
    wall = _Wall()
    env = _environment(wall)

    first = env.pause_generation()
    assert env.generation_charge_time(first) == 0.0
    wall.now = 1.0
    second = env.pause_generation()
    assert env.generation_charge_time(second) == 0.0
    wall.now = 4.0
    env.resume_generation(first, 2.5)

    assert env.state == EnvironmentState.PAUSED
    assert env.generation_pause_depth == 1
    assert env.time_manager.time_passed() == pytest.approx(2.5)
    # A causal follow-up admitted after the first completion starts at the settled parallel
    # frontier, not at the uninterrupted freeze's origin and not at the serialized sum.
    third = env.pause_generation()
    assert env.generation_charge_time(third) == pytest.approx(2.5)
    env.resume_generation(third, 1.0)

    wall.now = 7.0
    env.resume_generation(second, 3.5)

    assert env.state == EnvironmentState.RUNNING
    assert env.generation_pause_depth == 0
    assert env.time_manager.time_passed() == pytest.approx(7.0)


def test_parallel_sensitivity_frontier_does_not_sum_completed_siblings() -> None:
    wall = _Wall()
    env = _environment(wall)

    first = env.pause_generation()
    second = env.pause_generation()
    env.resume_generation(first, 4.0)
    after_first = env.pause_generation()
    assert env.generation_charge_time(after_first) == pytest.approx(4.0)

    env.resume_generation(second, 3.0)
    after_both = env.pause_generation()
    # The trajectory has banked 7s, but the parallel frontier is max(4s, 3s), not their sum.
    assert env.generation_charge_time(after_both) == pytest.approx(4.0)

    env.resume_generation(after_first, 2.0)
    env.resume_generation(after_both, 1.0)
    assert env.generation_pause_depth == 0
    assert env.time_manager.time_passed() == pytest.approx(10.0)


def test_judge_and_generation_pauses_cannot_resume_each_other() -> None:
    wall = _Wall()
    env = _environment(wall)

    env.pause()
    token = env.pause_generation()
    env.resume()

    assert env.judge_paused is False
    assert env.state == EnvironmentState.PAUSED
    assert env.time_manager.pause_depth == 1

    env.resume_generation(token, 1.0)
    assert env.state == EnvironmentState.RUNNING


def test_negative_offsets_never_rewind_and_idle_time_is_not_fast_forwarded() -> None:
    wall = _Wall()
    clock = ChargedTimeManager(wall_time=wall)
    clock.reset(start_time=0)

    wall.now = 2.0
    assert clock.time_passed() == 2.0
    clock.add_offset(-100.0)
    wall.now = 3.0

    # Outside an explicit charged pause, the harness adds neither a jump nor a multiplier.
    assert clock.offset == 0.0
    assert clock.time_passed() == 3.0


def test_duplicate_or_unknown_generation_completion_is_ignored() -> None:
    wall = _Wall()
    env = _environment(wall)
    token = env.pause_generation()

    env.resume_generation(token + 1, 20.0)
    assert env.state == EnvironmentState.PAUSED

    env.resume_generation(token, 1.0)
    before = env.time_manager.time_passed()
    env.resume_generation(token, 20.0)
    assert env.time_manager.time_passed() == before


def test_generation_started_after_environment_stop_is_an_unpaused_no_op() -> None:
    wall = _Wall()
    env = _environment(wall)
    env.state = EnvironmentState.STOPPED

    token = env.pause_generation()
    env.resume_generation(token, 20.0)

    assert token == 0
    assert env.generation_pause_depth == 0
    assert env.time_manager.pause_depth == 0
    assert env.time_manager.offset == 0.0


def test_environment_factory_keeps_the_are_surface() -> None:
    env = ChargedEnvironment(config=EnvironmentConfig(start_time=10, verbose=False))
    assert isinstance(env.time_manager, ChargedTimeManager)
    assert env.time_manager.time() >= 10.0
    assert isinstance(env.get_state(), dict)


def _recording_event(index: int, log: list[tuple[int, float]], env: ChargedEnvironment) -> Event:
    def deliver() -> None:
        log.append((index, float(env.time_manager.time_passed())))

    return Event.from_function(deliver).depends_on(None, delay_seconds=index)


def test_a_long_provider_wait_is_replaced_by_its_charge_not_by_wall_time() -> None:
    """Events authored inside a frozen interval are delivered on its resume, at the charge.

    This pins the semantics rather than claiming timeliness: the six events are authored at
    simulated seconds 0..5 and all six arrive together at simulated second 5, so the five earlier
    ones are late by up to 5s.  What the charge buys is that the delay is the *charge* (5s) and
    not the provider's wall latency (600s) -- and that delivery happens on the resume itself,
    without anyone driving the loop by hand.
    """
    wall = _Wall()
    env = _environment(wall)
    delivered: list[tuple[int, float]] = []

    env.schedule([_recording_event(index, delivered, env) for index in range(6)])
    env.prepare_events_for_start()

    token = env.pause_generation()
    wall.now = 600.0
    env.resume_generation(token, 5.0)

    assert [index for index, _ in delivered] == list(range(6))
    assert [at for _, at in delivered] == pytest.approx([5.0] * 6)
    assert env.time_manager.time_passed() == pytest.approx(5.0)


def test_sequential_generations_do_not_accumulate_event_lateness() -> None:
    """The second call must not re-freeze the world before the first call's events are delivered.

    Without a synchronous resume-time tick this is a race against ARE's one-second loop: the
    agent's next call wins it, the skipped events wait for the freeze after that, and lateness
    grows with the length of the call chain instead of staying bounded by one charge.
    """
    wall = _Wall()
    env = _environment(wall)
    delivered: list[tuple[int, float]] = []

    env.schedule([_recording_event(3, delivered, env)])
    env.prepare_events_for_start()

    first = env.pause_generation()
    env.resume_generation(first, 5.0)
    # Delivered by the first resume -- before the next call can freeze the world again.
    assert [index for index, _ in delivered] == [3]
    assert [at for _, at in delivered] == pytest.approx([5.0])

    second = env.pause_generation()
    env.resume_generation(second, 5.0)
    assert [at for _, at in delivered] == pytest.approx([5.0])
    assert env.time_manager.time_passed() == pytest.approx(10.0)


def test_overlapping_generations_deliver_once_at_the_end_of_the_epoch() -> None:
    """With calls in flight together the bound is one freeze epoch, not one call.

    The world resumes on the last outstanding completion, so an event authored during the first
    call waits for the second one too -- and on the primary trajectory the two charges are
    *serialized* (4s + 6s), so the epoch costs their sum rather than their overlap.  Charging the
    overlap instead is the separate sensitivity axis carried by the parallel frontier.  This is
    the residual delay the resume-time tick does not remove, and it is deliberately pinned here.
    """
    wall = _Wall()
    env = _environment(wall)
    delivered: list[tuple[int, float]] = []

    env.schedule([_recording_event(3, delivered, env)])
    env.prepare_events_for_start()

    first = env.pause_generation()
    second = env.pause_generation()

    env.resume_generation(first, 4.0)
    assert delivered == []  # still frozen: the epoch is the union, not the first call

    env.resume_generation(second, 6.0)
    assert [index for index, _ in delivered] == [3]
    assert [at for _, at in delivered] == pytest.approx([10.0])


def test_judge_pause_nested_in_a_generation_delivers_on_the_final_resume() -> None:
    """The two pause axes share one world; whichever releases last is the one that delivers."""
    wall = _Wall()
    env = _environment(wall)
    delivered: list[tuple[int, float]] = []

    env.schedule([_recording_event(2, delivered, env)])
    env.prepare_events_for_start()

    token = env.pause_generation()
    env.pause()

    env.resume_generation(token, 4.0)
    assert delivered == []
    assert env.state == EnvironmentState.PAUSED

    env.resume()
    assert env.state == EnvironmentState.RUNNING
    assert [index for index, _ in delivered] == [2]
    assert [at for _, at in delivered] == pytest.approx([4.0])


def test_a_failing_catch_up_tick_does_not_surface_as_an_inference_error() -> None:
    """``resume_generation`` runs in the model client's ``finally``; it must not raise there.

    A dispatch failure would otherwise turn a successful model call into an inference error at
    teardown.  The clock state still settles, and ARE's own loop re-ticks within a second.
    """
    wall = _Wall()
    env = _environment(wall)

    def failing_tick() -> None:
        raise RuntimeError("event dispatch exploded")

    env.tick = failing_tick

    token = env.pause_generation()
    env.resume_generation(token, 3.0)

    assert env.state == EnvironmentState.RUNNING
    assert env.generation_pause_depth == 0
    assert env.time_manager.time_passed() == pytest.approx(3.0)
