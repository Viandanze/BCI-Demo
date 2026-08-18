"""
Async LLM Bridge - Non-blocking wrapper around synchronous LLM clients.

Problem
-------
The concrete LLM clients (OllamaClient / APIClient) issue synchronous HTTP
requests (requests.post) with long timeouts (up to 120s for first-time model
loading). Calling them directly from the main processing loop stalls the
whole pipeline: EEG acquisition, decoding, and SSE waveform pushes all stop
for the duration of the request.

Solution
--------
AsyncLLMBridge moves every LLM call onto a single dedicated daemon worker
thread and hands the outcome back through a thread-safe result queue that
the main loop polls once per frame (non-blocking). See the class docstring
for the full threading rationale.

Thread model:
    main thread                     llm-bridge-worker (daemon)
    -----------                     --------------------------
    submit_generate_candidates -->  task queue -> run client call
    ... keeps ticking EEG ...       (may block up to client timeout)
    poll() <-- result queue <------- LLMResult (or drop if abandoned)
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Sentinel pushed onto the task queue to ask the worker thread to exit.
_SHUTDOWN = object()


@dataclass(frozen=True)
class LLMResult:
    """Outcome of a single background LLM call, consumed by the main loop.

    Attributes:
        request_id: Identifier returned by the matching submit_* call.
        kind: Request kind, one of AsyncLLMBridge.GENERATE_CANDIDATES or
            AsyncLLMBridge.EXPAND_RESPONSE.
        ok: True when the wrapped call returned normally.
        value: Return value of the call (list[str] for candidate
            generation, str for expansion), None when ok is False.
        error: Repr of the raised exception when ok is False, else None.
        elapsed: Wall-clock duration of the call in seconds.
    """

    request_id: int
    kind: str
    ok: bool
    value: Any = None
    error: Optional[str] = None
    elapsed: float = 0.0


class AsyncLLMBridge:
    """Runs synchronous LLM calls on a background worker thread.

    Why a dedicated worker thread + result queue (instead of a
    ThreadPoolExecutor with Future callbacks):

    1. Deterministic single-threaded orchestration. The main loop polls the
       result queue once per frame and applies every state-machine mutation
       itself. A done-callback approach would fire on the worker thread and
       push cross-thread writes into the UI / state layer, spreading lock
       requirements across the codebase. With the queue, the only data
       crossing the thread boundary is an immutable LLMResult.
    2. Clean interpreter exit. ThreadPoolExecutor uses non-daemon threads
       that are joined by an atexit hook, so a request stuck inside a 120s
       requests.post would delay process shutdown even after shutdown().
       A daemon worker joined with a timeout guarantees prompt exit; a
       hung HTTP call simply dies with the process.
    3. Late-result disposal. The caller can abandon a request (for example
       after its own, shorter timeout). The worker checks the abandoned set
       before enqueueing, so stale responses never reach the caller and can
       never be mistaken for the result of a newer request. As an extra
       guard, the caller matches LLMResult.request_id against the id it
       expects.
    4. One LLM call at a time is the intended operating mode anyway (one
       in-flight request per interaction turn), so a single worker both
       matches the workload and serializes requests naturally.

    Locking discipline:
    - ``_lock`` protects only ``_abandoned``, ``_next_request_id`` and the
      ``_closed`` flag. Both queues are queue.Queue instances and are
      thread-safe by themselves.
    - Submit payloads (context lists) are snapshots owned by the worker
      once submitted; the caller must not mutate them afterwards.

    Usage:
        bridge = AsyncLLMBridge(llm_client)
        rid = bridge.submit_generate_candidates("query", context, 3)
        result = bridge.poll()          # call each frame; None if not ready
        ...
        bridge.shutdown(timeout=2.0)    # on application exit
    """

    GENERATE_CANDIDATES = "generate_candidates"
    EXPAND_RESPONSE = "expand_response"

    def __init__(self, client, worker_name: str = "llm-bridge-worker"):
        """
        Args:
            client: Any LLMClient implementation (Ollama/APIClient, mocks).
                Its public interface is used unchanged; the client itself
                remains unaware of threading.
            worker_name: Name for the worker thread (shows up in logs).
        """
        self._client = client
        self._tasks: "queue.Queue" = queue.Queue()
        self._results: "queue.Queue" = queue.Queue()
        self._lock = threading.Lock()
        self._abandoned: set = set()
        self._next_request_id = 0
        self._closed = False

        self._worker = threading.Thread(
            target=self._worker_loop, name=worker_name, daemon=True
        )
        self._worker.start()

    # ------------------------------------------------------------------
    # Submission API (main thread)
    # ------------------------------------------------------------------

    def submit_generate_candidates(
        self,
        intent_mode: str,
        context: list,
        n_candidates: int = 3,
        topic_hint: str = "",
    ) -> int:
        """Schedule generate_candidates() on the worker thread.

        Args:
            intent_mode: Cognitive mode (query/reason/create/review).
            context: Conversation context snapshot (must not be mutated
                by the caller after submission).
            n_candidates: Number of candidates to request.
            topic_hint: Optional topic hint.

        Returns:
            Monotonically increasing request id for result matching.
        """
        context_snapshot = list(context)

        def task() -> list:
            return self._client.generate_candidates(
                intent_mode=intent_mode,
                context=context_snapshot,
                n_candidates=n_candidates,
                topic_hint=topic_hint,
            )

        return self._submit(self.GENERATE_CANDIDATES, task)

    def submit_expand_response(
        self, selected_response: str, intent_mode: str, context: list
    ) -> int:
        """Schedule expand_response() on the worker thread.

        Args:
            selected_response: The short candidate the user selected.
            intent_mode: Original cognitive mode of the turn.
            context: Conversation context snapshot.

        Returns:
            Monotonically increasing request id for result matching.
        """
        context_snapshot = list(context)

        def task() -> str:
            return self._client.expand_response(
                selected_response, intent_mode, context_snapshot
            )

        return self._submit(self.EXPAND_RESPONSE, task)

    def _submit(self, kind: str, fn: Callable) -> int:
        """Register a task and enqueue it for the worker."""
        with self._lock:
            if self._closed:
                raise RuntimeError("AsyncLLMBridge has been shut down")
            self._next_request_id += 1
            request_id = self._next_request_id
        self._tasks.put((request_id, kind, fn))
        return request_id

    # ------------------------------------------------------------------
    # Result API (main thread, one call per frame)
    # ------------------------------------------------------------------

    def poll(self) -> Optional[LLMResult]:
        """Return one completed result, or None if none is ready.

        Non-blocking: safe to call every frame of the main loop.
        """
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Cancellation / lifecycle
    # ------------------------------------------------------------------

    def abandon(self, request_id: int) -> None:
        """Mark a request as abandoned; its result will be discarded.

        The worker thread cannot be interrupted mid-request (the underlying
        HTTP call owns it until it returns); abandonment simply guarantees
        the late result is dropped instead of being delivered.
        """
        with self._lock:
            self._abandoned.add(request_id)

    @property
    def is_available(self) -> bool:
        """Delegate availability check to the wrapped client."""
        return bool(self._client.is_available())

    def shutdown(self, timeout: float = 2.0) -> bool:
        """Stop the worker thread. Idempotent and safe to call repeatedly.

        Sends the shutdown sentinel, cancels the idea of pending work (the
        worker drains silently), and joins the worker with a timeout. If
        the worker is stuck inside a long HTTP call it is left running as a
        daemon thread so it cannot delay process exit.

        Args:
            timeout: Seconds to wait for the worker to finish its current
                task and exit.

        Returns:
            True if the worker exited within the timeout, False otherwise.
        """
        with self._lock:
            if self._closed:
                return not self._worker.is_alive()
            self._closed = True

        self._tasks.put(_SHUTDOWN)

        if threading.current_thread() is self._worker:
            return True

        self._worker.join(timeout)
        if self._worker.is_alive():
            logger.warning(
                "LLM worker still busy after %.1fs (stuck HTTP call?); "
                "leaving daemon thread behind on exit.",
                timeout,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "AsyncLLMBridge":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.shutdown()
        return False

    # ------------------------------------------------------------------
    # Worker (background thread)
    # ------------------------------------------------------------------

    def _worker_loop(self):
        """Execute queued LLM tasks sequentially until shutdown."""
        while True:
            task = self._tasks.get()
            if task is _SHUTDOWN:
                break

            request_id, kind, fn = task
            start = time.time()
            try:
                value = fn()
                result = LLMResult(
                    request_id=request_id,
                    kind=kind,
                    ok=True,
                    value=value,
                    elapsed=time.time() - start,
                )
            except Exception as exc:  # Worker must never die on client errors.
                logger.error("LLM task %s [%d] raised: %r", kind, request_id, exc)
                result = LLMResult(
                    request_id=request_id,
                    kind=kind,
                    ok=False,
                    error=repr(exc),
                    elapsed=time.time() - start,
                )

            with self._lock:
                abandoned = request_id in self._abandoned
                if abandoned:
                    self._abandoned.discard(request_id)

            if abandoned:
                logger.debug(
                    "Dropping late result of abandoned LLM request %d.", request_id
                )
                continue

            self._results.put(result)
