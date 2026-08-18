"""
Collaborative Reasoning Demo - BCI x LLM

Main entry point for Phase 1 of the NeuroDecode collaborative reasoning system.

Pipeline:
  BrainFlow Synthetic Board
    -> EEGStreamManager (rolling window)
    -> Decoder (EEGNet or Mock)
    -> IntentEncoder (MI class -> cognitive mode)
    -> AsyncLLMBridge (LLM calls run on a background worker thread)
    -> Visual Feedback (web display)
    -> Second BCI Round (candidate selection)
    -> Response expansion -> Context update -> loop

Threading model (LLM calls never block the real-time loop):
  The main loop ticks at ~100 Hz, running EEG acquisition / decoding / SSE
  waveform pushes. All LLM requests (candidate generation, response
  expansion) are dispatched to AsyncLLMBridge and executed on a dedicated
  background worker thread. The main loop polls the bridge result queue
  once per frame, so EEG updates keep flowing while the LLM is thinking,
  and LLM failures degrade the interaction gracefully instead of freezing
  the UI for up to 120 seconds.

Usage:
  # Mock mode (no Ollama needed):
  python scripts/collaborative_reasoning_demo.py --backend mock --real-decoder

  # Ollama mode (full LLM experience):
  python scripts/collaborative_reasoning_demo.py --backend ollama --real-decoder

  # API mode (OpenAI-compatible):
  python scripts/collaborative_reasoning_demo.py --backend api \\
    --api-url https://api.deepseek.com/v1 \\
    --api-key YOUR_KEY \\
    --model deepseek-chat

  # Coze agent mode (bring your own bot, credentials via COZE_* env vars):
  python scripts/collaborative_reasoning_demo.py --backend coze --real-decoder

  # Custom config:
  python scripts/collaborative_reasoning_demo.py --config configs/phase1.yaml

Prerequisites:
  pip install brainflow flask numpy scipy pyyaml
"""

import argparse
import logging
import os
import sys
import time
from typing import Optional

# --- Add project root to path ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.acquisition import BrainFlowAcquisition, EEGStreamConfig, EEGStreamManager
from src.decoders import MockDecoder, RealDecoder
from src.intent import IntentEncoder, ContextManager, BCIState
from src.llm_bridge import create_llm_client, AsyncLLMBridge
from src.feedback import VisualFeedback

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CollaborativeReasoning")


# =============================================================================
# Main Collaborative Reasoning Loop
# =============================================================================

class CollaborativeReasoningDemo:
    """
    Main demo orchestrator.

    Connects BrainFlow -> Decoder -> Intent Encoder -> LLM -> Visual Feedback
    into a complete collaborative reasoning loop.

    Concurrency design:
    - The main thread owns the state machine, the UI and the EEG stream.
      It never performs a blocking LLM call.
    - AsyncLLMBridge owns a single daemon worker thread that executes LLM
      requests; results come back through a queue polled once per frame.
    - On LLM timeout the loop degrades gracefully (configurable policy):
      * "fallback": present clearly-labelled placeholder candidates so the
        BCI selection round still works.
      * "abort": notify the UI and return to IDLE for a fresh attempt.
      There is no automatic retry, so a dead LLM backend cannot cause a
      request storm.
    """

    # Placeholder candidates shown when the LLM fails or times out. They are
    # deliberately self-describing so the degradation is visible to the user.
    LLM_FALLBACK_CANDIDATES = [
        "[LLM offline] The language model did not answer in time; this is a local placeholder.",
        "[LLM offline] Select this to finish the turn and try again on the next intent.",
        "[LLM offline] Check that the LLM backend (e.g. 'ollama serve') is running.",
    ]

    def __init__(
        self,
        llm_backend: str = "mock",
        llm_kwargs: Optional[dict] = None,
        use_real_decoder: bool = False,
        model_path: str = None,
        config: dict = None,
        acquisition=None,
        feedback=None,
        llm_client=None,
        llm_wait_timeout: float = 15.0,
        llm_timeout_policy: str = "fallback",
        completed_pause: float = 2.0,
    ):
        """
        Args:
            llm_backend: LLM backend name for create_llm_client (ignored if
                llm_client is provided; kept for backward compatibility).
            llm_kwargs: Backend-specific kwargs for create_llm_client.
            use_real_decoder: Use the trained EEGNet decoder if available.
            model_path: Optional path to the EEGNet checkpoint.
            config: Loaded configuration dict (defaults to load_config()).
            acquisition: Optional acquisition override (dependency
                injection, used by tests; defaults to BrainFlowAcquisition).
            feedback: Optional visual feedback override (dependency
                injection, used by tests; defaults to VisualFeedback).
            llm_client: Optional LLMClient override (dependency injection,
                used by tests; defaults to create_llm_client(...)).
            llm_wait_timeout: Seconds the main loop waits for an LLM result
                before degrading. Intentionally shorter than the raw HTTP
                timeout (120s) so the UI recovers quickly.
            llm_timeout_policy: "fallback" (placeholder candidates) or
                "abort" (return to idle) on LLM timeout/failure.
            completed_pause: Seconds to linger in COMPLETED before resetting
                to IDLE (non-blocking; EEG keeps flowing).
        """
        if llm_timeout_policy not in ("fallback", "abort"):
            raise ValueError(
                f"llm_timeout_policy must be 'fallback' or 'abort', "
                f"got {llm_timeout_policy!r}"
            )

        config = config or load_config()
        self.config = config
        self.sample_rate = config['acquisition']['sample_rate']
        self.window_size = config['acquisition']['window_size']

        self.acquisition = acquisition if acquisition is not None else BrainFlowAcquisition()
        self.decoder = self._create_decoder(use_real_decoder, model_path)
        self.intent_encoder = IntentEncoder(
            confidence_threshold=config['bci']['confidence_threshold'],
            debounce_frames=config['bci']['debounce_frames'],
        )
        self.context_manager = ContextManager(
            selection_timeout=config['bci']['selection_timeout']
        )
        self.llm_client = (
            llm_client if llm_client is not None
            else create_llm_client(llm_backend, **(llm_kwargs or {}))
        )
        self.llm_bridge = AsyncLLMBridge(self.llm_client)
        self.feedback = feedback if feedback is not None else VisualFeedback()
        self.feedback.port = config['feedback']['port']

        # Asynchronous LLM bookkeeping (main thread only).
        self._llm_wait_timeout = llm_wait_timeout
        self._llm_timeout_policy = llm_timeout_policy
        self._completed_pause = completed_pause
        self._pending_llm_id: Optional[int] = None      # Expected request id
        self._pending_llm_deadline: float = 0.0         # Main-loop deadline
        self._expansion_pending: bool = False           # Expansion in flight
        self._completed_pause_until: float = 0.0        # Non-blocking pause

        self._running = False

        eeg_config = EEGStreamConfig(
            window_seconds=config['eeg_stream']['window_seconds'],
            push_interval=config['eeg_stream']['push_interval'],
            display_channels=config['eeg_stream']['display_channels'],
            sample_rate=config['eeg_stream']['sample_rate'],
        )
        self._eeg_stream = EEGStreamManager(self.acquisition, eeg_config)

    def _create_decoder(self, use_real: bool, model_path: str = None):
        """Create decoder instance."""
        if use_real:
            try:
                return RealDecoder(
                    model_path=model_path,
                    source_sample_rate=float(self.sample_rate),
                )
            except Exception as e:
                logger.warning(f"Failed to load real decoder: {e}. Falling back to mock.")
                return MockDecoder(sample_rate=self.sample_rate)
        return MockDecoder(sample_rate=self.sample_rate)

    def run(self):
        """Run the main collaborative reasoning loop."""
        logger.info("=" * 60)
        logger.info("NeuroDecode x LLM - Collaborative Reasoning Demo")
        logger.info("=" * 60)

        if self.llm_bridge.is_available:
            logger.info(f"LLM backend: {self.llm_client.__class__.__name__} OK")
        else:
            logger.warning(f"LLM backend not available! Will use mock responses.")

        if not self.acquisition.available:
            logger.error("BrainFlow not available. Install with: pip install brainflow")
            self.llm_bridge.shutdown()
            return
        self.acquisition.start()

        self.feedback.start()
        logger.info(f"Open your browser to: http://127.0.0.1:{self.feedback.port}")
        logger.info("LLM calls run on a background worker; EEG streaming never blocks.")
        logger.info("Press Ctrl+C to stop.\n")

        self._running = True
        self.feedback.update_state("idle")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("\nStopping demo...")
        finally:
            self._running = False
            self.acquisition.stop()
            self.feedback.stop()
            self.llm_bridge.shutdown(timeout=2.0)
            logger.info("Demo stopped. Goodbye!")

    def _main_loop(self):
        """Main processing loop: one tick per frame, EEG never stalls."""
        while self._running:
            self._tick(time.time())
            time.sleep(0.01)

    def _tick(self, current_time: float):
        """Process exactly one frame: state dispatch + EEG display push.

        Kept as a separate method so tests can drive the loop frame by
        frame without spinning up threads or a real board.
        """
        state = self.context_manager.state

        if state == BCIState.IDLE:
            self._handle_idle(current_time)

        elif state == BCIState.INTENT_LOCKED:
            self._handle_intent_locked()

        elif state == BCIState.AWAITING_LLM:
            self._handle_awaiting_llm(current_time)

        elif state == BCIState.PRESENTING_CANDIDATES:
            self._handle_presenting_candidates(current_time)

        elif state == BCIState.COMPLETED:
            self._handle_completed(current_time)

        self._push_eeg_display(current_time)

    def _handle_idle(self, current_time: float):
        """IDLE state: acquire EEG, decode, check for intent."""
        eeg_data = self.acquisition.get_recent_data(self.window_size)
        if eeg_data is None or eeg_data.shape[1] < self.window_size:
            return

        label, probabilities = self.decoder.predict(eeg_data)
        intent = self.intent_encoder.encode(label, probabilities, current_time)

        if intent is not None and intent.is_confident:
            logger.info(f"Intent locked: {intent.mode_label} "
                        f"(confidence: {intent.confidence:.2%})")

            turn = self.context_manager.start_turn(
                intent_mode=intent.mode.value,
                confidence=intent.confidence,
                label=intent.mode_label,
            )

            self.feedback.update_state(
                "intent_locked",
                intent=intent.to_dict(),
            )

    def _handle_intent_locked(self):
        """INTENT_LOCKED state: dispatch candidate generation to the worker.

        The LLM call itself happens on the AsyncLLMBridge worker thread;
        this handler only submits the request and moves the state machine
        to AWAITING_LLM, so the next frames stay free for EEG updates.
        """
        current_turn = self.context_manager.current_turn
        if not current_turn:
            self.context_manager.reset_to_idle()
            return

        context = self.context_manager.get_context_for_llm()
        logger.info(f"Querying LLM in background (mode={current_turn.intent_mode})...")

        try:
            self._pending_llm_id = self.llm_bridge.submit_generate_candidates(
                intent_mode=current_turn.intent_mode,
                context=context,
                n_candidates=3,
            )
        except RuntimeError as exc:
            logger.error(f"Cannot dispatch LLM request: {exc}")
            self._degrade_after_llm_failure("bridge closed")
            return

        self._pending_llm_deadline = time.time() + self._llm_wait_timeout
        self.context_manager.set_awaiting_llm()
        self.feedback.update_state("awaiting_llm")

    def _handle_awaiting_llm(self, current_time: float):
        """AWAITING_LLM state: poll for candidates without blocking.

        Each frame: non-blocking poll of the bridge result queue. EEG
        pushes continue in _tick() while the worker talks to the LLM.
        """
        result = self.llm_bridge.poll()

        if result is not None:
            if result.request_id != self._pending_llm_id:
                # Defensive: stale result of an already-abandoned request.
                logger.debug("Discarding stale LLM result (request %d).",
                             result.request_id)
                return
            if result.ok and result.value:
                self._present_candidates(result.value)
            else:
                logger.error(f"LLM generation failed: {result.error or 'empty response'}")
                self._degrade_after_llm_failure("LLM error")
            return

        if current_time >= self._pending_llm_deadline:
            logger.warning(
                f"LLM did not answer within {self._llm_wait_timeout:.1f}s. "
                "Degrading gracefully (EEG pipeline unaffected)."
            )
            self.llm_bridge.abandon(self._pending_llm_id)
            self._degrade_after_llm_failure("timeout")

    def _present_candidates(self, candidates: list):
        """Publish candidates and move on to the selection round."""
        logger.info(f"LLM returned {len(candidates)} candidates")
        for i, c in enumerate(candidates):
            logger.info(f"  [{i}] {c[:80]}...")

        self._pending_llm_id = None
        self.context_manager.set_candidates(candidates)
        self.feedback.update_state(
            "presenting_candidates",
            candidates=candidates,
        )
        self.intent_encoder.reset()

    def _degrade_after_llm_failure(self, reason: str):
        """Graceful degradation when the LLM fails or times out.

        "fallback": present clearly-labelled placeholder candidates so the
            interaction loop (and the second BCI selection round) survives.
        "abort": notify the UI and return to IDLE for a fresh attempt.

        Either way there is exactly one request per turn and no automatic
        retry, so an unreachable LLM backend cannot trigger a retry storm.
        """
        self._pending_llm_id = None

        if self._llm_timeout_policy == "abort":
            logger.error(f"LLM unavailable ({reason}). Aborting turn, back to idle.")
            self.feedback.update_state("llm_timeout")
            self.context_manager.reset_to_idle()
            self.intent_encoder.reset()
            return

        logger.warning(f"LLM unavailable ({reason}). Presenting fallback candidates.")
        self._present_candidates(list(self.LLM_FALLBACK_CANDIDATES))

    def _handle_presenting_candidates(self, current_time: float):
        """PRESENTING_CANDIDATES state: wait for BCI selection."""
        if self.context_manager.check_timeout():
            logger.warning("Selection timeout! Auto-selecting best candidate.")
            self._select_candidate(0, auto=True)
            return

        eeg_data = self.acquisition.get_recent_data(self.window_size)
        if eeg_data is None or eeg_data.shape[1] < self.window_size:
            return

        label, probabilities = self.decoder.predict(eeg_data)

        current_turn = self.context_manager.current_turn
        if not current_turn or not current_turn.llm_candidates:
            return

        n_candidates = len(current_turn.llm_candidates)
        selected = self.intent_encoder.encode_selection(
            label, probabilities, n_candidates, current_time
        )

        if selected is not None:
            logger.info(f"User selected candidate [{selected}]")
            self._select_candidate(selected, auto=False)

    def _select_candidate(self, index: int, auto: bool):
        """Process candidate selection; expansion runs on the worker.

        Selection itself is instant; the optional expand_response() call is
        dispatched to AsyncLLMBridge and resolved later in
        _handle_completed(), so the main loop returns immediately.
        """
        response = self.context_manager.select_candidate(index)
        if response is None:
            return

        self.feedback.update_selection(index, auto=auto)

        current_turn = self.context_manager.current_turn
        self._completed_pause_until = time.time() + self._completed_pause

        if not auto:
            logger.info("Dispatching response expansion to background worker...")
            context = self.context_manager.get_context_for_llm()
            mode = current_turn.intent_mode if current_turn else "query"
            try:
                self._pending_llm_id = self.llm_bridge.submit_expand_response(
                    response, mode, context
                )
                self._pending_llm_deadline = time.time() + self._llm_wait_timeout
                self._expansion_pending = True
                return
            except RuntimeError as exc:
                logger.error(f"Cannot dispatch expansion: {exc}. Keeping short response.")

        self._finish_turn(response)

    def _handle_completed(self, current_time: float):
        """COMPLETED state: resolve pending expansion, then pause and reset.

        Replaces the previous blocking time.sleep(2.0) with an
        elapsed-time check, so EEG pushes keep flowing while the final
        response is being expanded and during the completion pause.
        """
        if self._expansion_pending:
            result = self.llm_bridge.poll()

            if result is not None:
                if result.request_id != self._pending_llm_id:
                    logger.debug("Discarding stale LLM result (request %d).",
                                 result.request_id)
                    return
                self._expansion_pending = False
                self._pending_llm_id = None
                if result.ok:
                    logger.info(f"Final response: {result.value[:200]}")
                    self._finish_turn(result.value)
                else:
                    logger.error(
                        f"Response expansion failed: {result.error}. "
                        "Keeping short response."
                    )
                    self._finish_turn(self._short_response())
                return

            if current_time >= self._pending_llm_deadline:
                logger.warning("Response expansion timed out. Keeping short response.")
                self.llm_bridge.abandon(self._pending_llm_id)
                self._expansion_pending = False
                self._pending_llm_id = None
                self._finish_turn(self._short_response())
            return

        if current_time < self._completed_pause_until:
            return

        self.context_manager.reset_to_idle()
        self.intent_encoder.reset()
        self.feedback.update_state("idle")

    def _short_response(self) -> str:
        """Return the selected (un-expanded) candidate of the current turn."""
        turn = self.context_manager.current_turn
        return turn.final_response if turn else ""

    def _finish_turn(self, final_response: str):
        """Push history to the UI and close the turn in the logs."""
        self.feedback.update_history(
            [t.to_dict() for t in self.context_manager.turns]
        )

        current_turn = self.context_manager.current_turn
        if current_turn:
            logger.info(f"Turn completed. Duration: {current_turn.duration:.1f}s")
        logger.info("-" * 40)

    def _push_eeg_display(self, current_time: float):
        """Push rolling-window EEG data to visual feedback."""
        self._eeg_stream.update(current_time, self.feedback)


# =============================================================================
# Entry Point
# =============================================================================

def _load_env_file(path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (no deps).

    Real environment variables always win: .env only fills gaps, so it is
    safe on machines without the file. Used for the coze backend
    credentials (COZE_AGENT_DOMAIN / COZE_PROJECT_ID / COZE_API_TOKEN).
    """
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def main():
    _load_env_file()
    parser = argparse.ArgumentParser(
        description="NeuroDecode x LLM Collaborative Reasoning Demo"
    )
    parser.add_argument(
        "--backend", choices=["ollama", "api", "coze", "mock"], default=None,
        help="LLM backend (default: from config, usually mock)",
    )
    parser.add_argument("--model", default=None, help="Ollama model name")
    parser.add_argument("--host", default=None, help="Ollama host")
    parser.add_argument("--api-url", default="", help="API URL (for api backend)")
    parser.add_argument("--api-key", default="", help="API key (for api backend)")
    parser.add_argument("--api-model", default="", help="API model name (for api backend)")
    parser.add_argument(
        "--llm-wait-timeout", type=float, default=15.0,
        help="Seconds to wait for an LLM result before graceful degradation "
             "(default: 15)",
    )
    parser.add_argument(
        "--llm-timeout-policy", choices=["fallback", "abort"], default="fallback",
        help="Behavior on LLM timeout: present placeholder candidates or "
             "abort the turn back to idle (default: fallback)",
    )
    parser.add_argument(
        "--real-decoder", action="store_true",
        help="Use trained EEGNet decoder instead of mock",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Path to trained EEGNet checkpoint (auto-detected if not specified)",
    )
    parser.add_argument("--port", type=int, default=None, help="Web UI port (default: from config)")
    parser.add_argument(
        "--config", default=None,
        help="Path to YAML config file (default: configs/phase1.yaml)",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    backend = args.backend or config['llm']['default_backend']

    llm_kwargs = {}
    if backend == "ollama":
        llm_kwargs = {
            "model": args.model or config['llm']['ollama']['model'],
            "host": args.host or config['llm']['ollama']['host'],
        }
    elif backend == "api":
        api_url = args.api_url or config['llm']['api']['url']
        api_key = args.api_key or config['llm']['api']['key']
        if not api_url or not api_key:
            parser.error("--api-url and --api-key required for api backend")
        llm_kwargs = {
            "api_url": api_url,
            "api_key": api_key,
            "model": args.api_model or config['llm']['api']['model'],
        }

    demo = CollaborativeReasoningDemo(
        llm_backend=backend,
        llm_kwargs=llm_kwargs,
        use_real_decoder=args.real_decoder,
        model_path=args.model_path,
        config=config,
        llm_wait_timeout=args.llm_wait_timeout,
        llm_timeout_policy=args.llm_timeout_policy,
    )

    if args.port is not None:
        demo.feedback.port = args.port

    demo.run()


if __name__ == "__main__":
    main()
