"""
Tests for the asynchronous LLM path.

Covers:
  - AsyncLLMBridge (unit): submit/poll lifecycle, request ids, error and
    latency propagation, abandonment of late results, shutdown behavior
  - Non-blocking guarantee: while the worker executes a slow LLM call, the
    calling loop keeps running (regression test for the blocking bug where
    requests.post could stall the main loop for up to 120 seconds)
  - CollaborativeReasoningDemo (integration, dependency-injected fakes):
    * EEG pushes keep flowing while the LLM is slow
    * LLM timeout degrades gracefully (fallback candidates / abort to idle)
    * Response expansion runs asynchronously; expansion timeout keeps the
      short response instead of hanging
    * An abandoned late result never corrupts the following turn
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.config import load_config
from src.intent.context_manager import BCIState
from src.llm_bridge.async_bridge import AsyncLLMBridge, LLMResult
from collaborative_reasoning_demo import CollaborativeReasoningDemo


# =============================================================================
# Fakes
# =============================================================================

class ScriptedLLMClient:
    """Configurable fake LLM client: latency, failures, call recording."""

    def __init__(self, delay: float = 0.0, fail: bool = False):
        self.delay = delay
        self.fail = fail
        self.calls = []
        self._lock = threading.Lock()  # calls are recorded on the worker thread

    def is_available(self) -> bool:
        return True

    def generate_candidates(self, intent_mode, context, n_candidates=3,
                            topic_hint=""):
        with self._lock:
            self.calls.append(
                ("generate_candidates", intent_mode, list(context))
            )
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("simulated LLM crash")
        return [f"candidate {i}" for i in range(n_candidates)]

    def expand_response(self, selected_response, intent_mode, context):
        with self._lock:
            self.calls.append(("expand_response", intent_mode))
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("simulated LLM crash")
        return selected_response + " [expanded]"


class FakeAcquisition:
    """Deterministic acquisition stub exposing the BrainFlowAcquisition API."""

    def __init__(self, sample_rate: int = 250, n_channels: int = 8):
        self.available = True
        self.sample_rate = sample_rate
        self.n_eeg_channels = n_channels
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def get_recent_data(self, n_samples: int):
        n = int(n_samples)
        now = time.time()
        t = np.linspace(now - n / self.sample_rate, now, n)
        row = np.sin(2.0 * np.pi * 10.0 * t)
        return np.tile(row, (self.n_eeg_channels, 1))


class FakeFeedback:
    """Records UI events instead of serving them over HTTP."""

    def __init__(self):
        self.port = 0
        self.states = []
        self.eeg_pushes = 0
        self.selections = []
        self.history = None

    def start(self):
        pass

    def stop(self):
        pass

    def update_state(self, state, intent=None, candidates=None):
        self.states.append(state)

    def update_eeg(self, eeg_batch):
        self.eeg_pushes += 1

    def update_selection(self, index, auto=False):
        self.selections.append((index, auto))

    def update_history(self, turns):
        self.history = turns


# =============================================================================
# Helpers
# =============================================================================

def poll_until(bridge: AsyncLLMBridge, timeout: float = 5.0) -> LLMResult:
    """Poll the bridge until a result shows up (or fail the test)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = bridge.poll()
        if result is not None:
            return result
        time.sleep(0.005)
    pytest.fail(f"No LLM result within {timeout}s")


def run_until(demo, target: BCIState, timeout: float = 5.0) -> bool:
    """Drive the demo frame by frame until the state machine reaches target."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        demo._tick(time.time())
        if demo.context_manager.state == target:
            return True
        time.sleep(0.005)
    return False


def make_demo(client, *, llm_wait_timeout=5.0, llm_timeout_policy="fallback",
              push_interval=0.02, selection_timeout=0.2, completed_pause=0.05):
    """Build a demo with fake acquisition/feedback for integration tests."""
    cfg = load_config()
    cfg['eeg_stream']['push_interval'] = push_interval
    cfg['bci']['selection_timeout'] = selection_timeout
    return CollaborativeReasoningDemo(
        config=cfg,
        acquisition=FakeAcquisition(),
        feedback=FakeFeedback(),
        llm_client=client,
        llm_wait_timeout=llm_wait_timeout,
        llm_timeout_policy=llm_timeout_policy,
        completed_pause=completed_pause,
    )


@pytest.fixture
def bridge():
    """Fresh AsyncLLMBridge around a fast fake client, shut down after test."""
    b = AsyncLLMBridge(ScriptedLLMClient())
    yield b
    b.shutdown(timeout=1.0)


@pytest.fixture
def demo_factory():
    """Factory that builds demos and shuts their bridges down afterwards."""
    demos = []

    def factory(client, **kwargs):
        demo = make_demo(client, **kwargs)
        demos.append(demo)
        return demo

    yield factory
    for d in demos:
        d.llm_bridge.shutdown(timeout=1.0)


# =============================================================================
# Unit tests: AsyncLLMBridge
# =============================================================================

class TestAsyncLLMBridgeBasics:
    """Core submit/poll contract of the bridge."""

    def test_poll_returns_none_when_no_work(self, bridge):
        assert bridge.poll() is None

    def test_generate_candidates_roundtrip(self, bridge):
        request_id = bridge.submit_generate_candidates(
            "query", [{"role": "user", "content": "hi"}], n_candidates=3
        )
        result = poll_until(bridge)

        assert result.request_id == request_id
        assert result.kind == AsyncLLMBridge.GENERATE_CANDIDATES
        assert result.ok is True
        assert result.value == ["candidate 0", "candidate 1", "candidate 2"]
        assert result.error is None
        # Result is consumed exactly once.
        assert bridge.poll() is None

    def test_expand_response_roundtrip(self, bridge):
        request_id = bridge.submit_expand_response("answer", "reason", [])
        result = poll_until(bridge)

        assert result.request_id == request_id
        assert result.kind == AsyncLLMBridge.EXPAND_RESPONSE
        assert result.ok is True
        assert result.value == "answer [expanded]"

    def test_request_ids_increase_monotonically(self, bridge):
        ids = [
            bridge.submit_generate_candidates("query", []),
            bridge.submit_generate_candidates("reason", []),
            bridge.submit_expand_response("x", "query", []),
        ]
        assert ids == sorted(ids)
        assert len(set(ids)) == 3

    def test_result_records_elapsed_time(self):
        b = AsyncLLMBridge(ScriptedLLMClient(delay=0.2))
        try:
            b.submit_generate_candidates("query", [])
            result = poll_until(b, timeout=3.0)
            assert result.elapsed >= 0.15
        finally:
            b.shutdown(timeout=1.0)

    def test_is_available_delegates_to_client(self):
        class UnavailableClient(ScriptedLLMClient):
            def is_available(self):
                return False

        b = AsyncLLMBridge(UnavailableClient())
        try:
            assert b.is_available is False
        finally:
            b.shutdown(timeout=1.0)

    def test_context_is_snapshotted_at_submit_time(self, bridge):
        """Mutating the caller's context after submit must not leak in."""
        context = [{"role": "user", "content": "original"}]
        bridge.submit_generate_candidates("query", context)
        context.append({"role": "user", "content": "mutated"})

        poll_until(bridge)
        recorded = bridge._client.calls[-1]
        assert len(recorded[2]) == 1
        assert recorded[2][0]["content"] == "original"


class TestAsyncLLMBridgeErrors:
    """Failure and cancellation behavior."""

    def test_client_exception_yields_error_result(self):
        b = AsyncLLMBridge(ScriptedLLMClient(fail=True))
        try:
            request_id = b.submit_generate_candidates("query", [])
            result = poll_until(b)

            assert result.request_id == request_id
            assert result.ok is False
            assert result.value is None
            assert "simulated LLM crash" in result.error
        finally:
            b.shutdown(timeout=1.0)

    def test_abandoned_result_is_dropped(self):
        """A late result of an abandoned request never reaches the queue."""
        b = AsyncLLMBridge(ScriptedLLMClient(delay=0.4))
        try:
            request_id = b.submit_generate_candidates("query", [])
            b.abandon(request_id)

            time.sleep(0.8)  # Let the worker finish the abandoned call.
            assert b.poll() is None
            assert b.poll() is None
        finally:
            b.shutdown(timeout=1.0)

    def test_abandon_unknown_id_is_harmless(self, bridge):
        bridge.abandon(9999)  # Must not raise.
        bridge.abandon(9999)


class TestAsyncLLMBridgeNonBlocking:
    """The regression core: a slow LLM must never block the calling loop."""

    def test_slow_llm_does_not_block_caller(self):
        """While the worker sleeps 0.6s, the caller loop must keep ticking.

        Before the async refactor, generate_candidates() ran inline and the
        caller (main loop -> EEG updates) froze for the whole call.
        """
        b = AsyncLLMBridge(ScriptedLLMClient(delay=0.6))
        try:
            start = time.perf_counter()
            request_id = b.submit_generate_candidates("query", [])
            submit_elapsed = time.perf_counter() - start
            assert submit_elapsed < 0.05  # Submission itself is instant.

            iterations = 0
            saw_result = False
            loop_start = time.perf_counter()
            while time.perf_counter() - loop_start < 0.3:
                if b.poll() is not None:
                    saw_result = True
                    break
                iterations += 1
                time.sleep(0.002)

            # The worker is still inside its 0.6s call, yet the caller
            # loop completed many iterations without observing a result.
            assert saw_result is False
            assert iterations >= 20

            result = poll_until(b, timeout=3.0)
            assert result.ok is True
            assert result.request_id == request_id
        finally:
            b.shutdown(timeout=1.0)


class TestAsyncLLMBridgeShutdown:
    """Lifecycle: clean exit even with an in-flight request."""

    def test_shutdown_joins_idle_worker(self, bridge):
        assert bridge.shutdown(timeout=1.0) is True
        assert bridge._worker.is_alive() is False

    def test_submit_after_shutdown_raises(self, bridge):
        bridge.shutdown(timeout=1.0)
        with pytest.raises(RuntimeError, match="shut down"):
            bridge.submit_generate_candidates("query", [])

    def test_shutdown_is_idempotent(self, bridge):
        assert bridge.shutdown(timeout=1.0) is True
        assert bridge.shutdown(timeout=1.0) is True

    def test_shutdown_with_stuck_worker_returns_promptly(self):
        """A worker stuck inside a long HTTP-ish call must not stall exit."""
        b = AsyncLLMBridge(ScriptedLLMClient(delay=3.0))
        b.submit_expand_response("x", "query", [])

        start = time.perf_counter()
        exited = b.shutdown(timeout=0.2)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0       # Did not wait for the 3s fake call.
        assert exited is False     # Worker is deliberately left behind (daemon).

    def test_context_manager_shuts_down(self):
        with AsyncLLMBridge(ScriptedLLMClient()) as b:
            b.submit_generate_candidates("query", [])
        assert b._worker.is_alive() is False


# =============================================================================
# Integration tests: demo loop with async LLM
# =============================================================================

class TestDemoAsyncCandidateGeneration:
    """INTENT_LOCKED -> AWAITING_LLM -> PRESENTING_CANDIDATES, EEG alive."""

    def test_slow_llm_keeps_eeg_streaming(self, demo_factory):
        """EEG pushes must continue while the worker waits on the LLM."""
        demo = demo_factory(
            ScriptedLLMClient(delay=0.8), llm_wait_timeout=10.0
        )
        feedback = demo.feedback
        cm = demo.context_manager

        cm.start_turn("query", 0.9, "Query")

        awaiting_baseline = None
        deadline = time.time() + 5.0
        while time.time() < deadline:
            demo._tick(time.time())
            if cm.state == BCIState.AWAITING_LLM and awaiting_baseline is None:
                awaiting_baseline = feedback.eeg_pushes
            if cm.state == BCIState.PRESENTING_CANDIDATES:
                break
            time.sleep(0.005)

        assert cm.state == BCIState.PRESENTING_CANDIDATES
        # LLM took 0.8s; with push_interval=0.02s expect ~40 pushes.
        # A conservative floor proves the loop never froze.
        pushes_during_await = feedback.eeg_pushes - awaiting_baseline
        assert pushes_during_await >= 15
        # Real candidates (not fallback) were presented.
        assert cm.current_turn.llm_candidates[0] == "candidate 0"
        assert "awaiting_llm" in feedback.states
        assert feedback.states[-1] == "presenting_candidates"

    def test_llm_timeout_falls_back_gracefully(self, demo_factory):
        """Main-loop timeout short-circuits a 3s LLM call and degrades."""
        demo = demo_factory(
            ScriptedLLMClient(delay=3.0), llm_wait_timeout=0.3
        )
        feedback = demo.feedback
        cm = demo.context_manager

        cm.start_turn("query", 0.9, "Query")

        awaiting_baseline = None
        start = time.time()
        deadline = start + 5.0
        while time.time() < deadline:
            demo._tick(time.time())
            if cm.state == BCIState.AWAITING_LLM and awaiting_baseline is None:
                awaiting_baseline = feedback.eeg_pushes
            if cm.state == BCIState.PRESENTING_CANDIDATES:
                break
            time.sleep(0.005)

        elapsed = time.time() - start

        assert cm.state == BCIState.PRESENTING_CANDIDATES
        # Degraded after ~0.3s instead of waiting for the 3s LLM call.
        assert elapsed < 2.0
        # Clearly-labelled fallback candidates, not an empty screen.
        candidates = cm.current_turn.llm_candidates
        assert len(candidates) == 3
        assert all(c.startswith("[LLM offline]") for c in candidates)
        # EEG kept flowing while waiting for the dead LLM.
        assert feedback.eeg_pushes - awaiting_baseline >= 5
        # The turn still completes end to end.
        assert run_until(demo, BCIState.IDLE, timeout=5.0)
        assert feedback.history is not None
        assert feedback.states[-1] == "idle"

    def test_llm_timeout_abort_policy_returns_to_idle(self, demo_factory):
        """'abort' policy discards the turn and goes straight back to IDLE."""
        demo = demo_factory(
            ScriptedLLMClient(delay=3.0),
            llm_wait_timeout=0.3,
            llm_timeout_policy="abort",
        )
        feedback = demo.feedback
        cm = demo.context_manager

        cm.start_turn("query", 0.9, "Query")
        assert run_until(demo, BCIState.IDLE, timeout=5.0)

        assert cm.state == BCIState.IDLE
        assert cm.current_turn is None
        assert cm.turns == []              # Turn discarded, nothing archived.
        assert "llm_timeout" in feedback.states
        assert "presenting_candidates" not in feedback.states
        assert feedback.eeg_pushes > 0     # Loop kept ticking throughout.

    def test_invalid_timeout_policy_rejected(self):
        with pytest.raises(ValueError, match="llm_timeout_policy"):
            make_demo(ScriptedLLMClient(), llm_timeout_policy="explode")


class TestDemoAsyncExpansion:
    """Selection -> background expansion -> COMPLETED, no blocking."""

    def test_manual_selection_expands_asynchronously(self, demo_factory):
        client = ScriptedLLMClient(delay=0.15)
        demo = demo_factory(client, llm_wait_timeout=5.0)
        feedback = demo.feedback
        cm = demo.context_manager

        cm.start_turn("reason", 0.85, "Reason")
        assert run_until(demo, BCIState.PRESENTING_CANDIDATES, timeout=5.0)

        # Select immediately (deterministic: no encoder tick in between).
        demo._select_candidate(1, auto=False)

        assert cm.state == BCIState.COMPLETED
        assert run_until(demo, BCIState.IDLE, timeout=5.0)

        # The expansion really ran on the worker.
        kinds = [c[0] for c in client.calls]
        assert "expand_response" in kinds
        # History keeps the short candidate (pre-existing semantics).
        assert feedback.selections == [(1, False)]
        assert feedback.history is not None
        assert len(cm.turns) == 1
        assert cm.turns[0].final_response == "candidate 1"
        assert feedback.states[-1] == "idle"

    def test_expansion_timeout_keeps_short_response(self, demo_factory):
        """Dead LLM during expansion: keep the short answer, finish anyway."""
        demo = demo_factory(
            ScriptedLLMClient(delay=3.0), llm_wait_timeout=0.3
        )
        feedback = demo.feedback
        cm = demo.context_manager

        cm.start_turn("query", 0.9, "Query")
        assert run_until(demo, BCIState.PRESENTING_CANDIDATES, timeout=5.0)
        fallback = cm.current_turn.llm_candidates

        demo._select_candidate(0, auto=False)

        start = time.time()
        assert run_until(demo, BCIState.IDLE, timeout=5.0)
        elapsed = time.time() - start

        # Expansion degraded after ~0.3s, not after the 3s LLM call.
        assert elapsed < 2.0
        assert cm.turns[0].final_response == fallback[0]
        assert feedback.history is not None
        assert feedback.states[-1] == "idle"

    def test_auto_selection_skips_expansion(self, demo_factory):
        client = ScriptedLLMClient()
        demo = demo_factory(client, llm_wait_timeout=5.0)
        cm = demo.context_manager

        cm.start_turn("query", 0.9, "Query")
        assert run_until(demo, BCIState.PRESENTING_CANDIDATES, timeout=5.0)

        demo._select_candidate(0, auto=True)

        kinds = [c[0] for c in client.calls]
        assert "expand_response" not in kinds
        assert run_until(demo, BCIState.IDLE, timeout=5.0)
        assert cm.turns[0].final_response == "candidate 0"


class TestDemoStaleResultIsolation:
    """Abandoned requests must never corrupt the following turn."""

    def test_late_result_of_abandoned_turn_does_not_affect_next_turn(
        self, demo_factory
    ):
        client = ScriptedLLMClient(delay=1.0)
        demo = demo_factory(client, llm_wait_timeout=0.25)
        cm = demo.context_manager

        # Turn 1: LLM too slow -> fallback candidates.
        cm.start_turn("query", 0.9, "Query")
        assert run_until(demo, BCIState.PRESENTING_CANDIDATES, timeout=5.0)
        assert cm.current_turn.llm_candidates[0].startswith("[LLM offline]")

        demo._select_candidate(0, auto=True)
        assert run_until(demo, BCIState.IDLE, timeout=5.0)

        # Let the abandoned turn-1 call finish on the worker: its result
        # must be dropped instead of being delivered.
        time.sleep(1.2)
        assert demo.llm_bridge.poll() is None

        # Turn 2 with a healthy client returns real candidates.
        client.delay = 0.0
        cm.start_turn("reason", 0.8, "Reason")
        assert run_until(demo, BCIState.PRESENTING_CANDIDATES, timeout=5.0)
        assert cm.current_turn.llm_candidates[0] == "candidate 0"

        demo._select_candidate(2, auto=True)
        assert run_until(demo, BCIState.IDLE, timeout=5.0)
        assert len(cm.turns) == 2
        assert cm.turns[1].final_response == "candidate 2"
