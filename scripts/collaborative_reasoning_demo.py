"""
Collaborative Reasoning Demo - BCI x LLM

Main entry point for Phase 1 of the NeuroDecode collaborative reasoning system.

Pipeline:
  BrainFlow Synthetic Board
    -> EEGStreamManager (rolling window)
    -> Decoder (EEGNet or Mock)
    -> IntentEncoder (MI class -> cognitive mode)
    -> LLM Bridge (generate candidates)
    -> Visual Feedback (web display)
    -> Second BCI Round (candidate selection)
    -> Response expansion -> Context update -> loop

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
from src.llm_bridge import create_llm_client
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
    """

    def __init__(
        self,
        llm_backend: str = "mock",
        llm_kwargs: Optional[dict] = None,
        use_real_decoder: bool = False,
        model_path: str = None,
        config: dict = None,
    ):
        config = config or load_config()
        self.config = config
        self.sample_rate = config['acquisition']['sample_rate']
        self.window_size = config['acquisition']['window_size']

        self.acquisition = BrainFlowAcquisition()
        self.decoder = self._create_decoder(use_real_decoder, model_path)
        self.intent_encoder = IntentEncoder(
            confidence_threshold=config['bci']['confidence_threshold'],
            debounce_frames=config['bci']['debounce_frames'],
        )
        self.context_manager = ContextManager(
            selection_timeout=config['bci']['selection_timeout']
        )
        self.llm_client = create_llm_client(llm_backend, **(llm_kwargs or {}))
        self.feedback = VisualFeedback()
        self.feedback.port = config['feedback']['port']

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

        if self.llm_client.is_available():
            logger.info(f"LLM backend: {self.llm_client.__class__.__name__} OK")
        else:
            logger.warning(f"LLM backend not available! Will use mock responses.")

        if not self.acquisition.available:
            logger.error("BrainFlow not available. Install with: pip install brainflow")
            return
        self.acquisition.start()

        self.feedback.start()
        logger.info(f"Open your browser to: http://127.0.0.1:{self.feedback.port}")
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
            logger.info("Demo stopped. Goodbye!")

    def _main_loop(self):
        """Main processing loop."""
        while self._running:
            current_time = time.time()

            state = self.context_manager.state

            if state == BCIState.IDLE:
                self._handle_idle(current_time)

            elif state == BCIState.INTENT_LOCKED:
                self._handle_intent_locked()

            elif state == BCIState.PRESENTING_CANDIDATES:
                self._handle_presenting_candidates(current_time)

            elif state == BCIState.COMPLETED:
                self._handle_completed()

            self._push_eeg_display(current_time)
            time.sleep(0.01)

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
        """INTENT_LOCKED state: query LLM for candidates."""
        self.context_manager.set_awaiting_llm()
        self.feedback.update_state("awaiting_llm")

        current_turn = self.context_manager.current_turn
        if not current_turn:
            self.context_manager.reset_to_idle()
            return

        context = self.context_manager.get_context_for_llm()

        logger.info(f"Querying LLM (mode={current_turn.intent_mode})...")
        candidates = self.llm_client.generate_candidates(
            intent_mode=current_turn.intent_mode,
            context=context,
            n_candidates=3,
        )

        logger.info(f"LLM returned {len(candidates)} candidates")
        for i, c in enumerate(candidates):
            logger.info(f"  [{i}] {c[:80]}...")

        self.context_manager.set_candidates(candidates)
        self.feedback.update_state(
            "presenting_candidates",
            candidates=candidates,
        )
        self.intent_encoder.reset()

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
        """Process candidate selection."""
        response = self.context_manager.select_candidate(index)
        if response is None:
            return

        self.feedback.update_selection(index, auto=auto)

        current_turn = self.context_manager.current_turn

        if not auto:
            logger.info("Expanding selected response...")
            context = self.context_manager.get_context_for_llm()
            expanded = self.llm_client.expand_response(
                response,
                current_turn.intent_mode if current_turn else "query",
                context,
            )
            logger.info(f"Final response: {expanded[:200]}")
        else:
            expanded = response

        self.feedback.update_history(
            [t.to_dict() for t in self.context_manager.turns]
        )

        logger.info(f"Turn completed. Duration: "
                    f"{current_turn.duration:.1f}s" if current_turn else "")
        logger.info("-" * 40)

    def _handle_completed(self):
        """COMPLETED state: brief pause then reset to idle."""
        time.sleep(2.0)
        self.context_manager.reset_to_idle()
        self.intent_encoder.reset()
        self.feedback.update_state("idle")

    def _push_eeg_display(self, current_time: float):
        """Push rolling-window EEG data to visual feedback."""
        self._eeg_stream.update(current_time, self.feedback)


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NeuroDecode x LLM Collaborative Reasoning Demo"
    )
    parser.add_argument(
        "--backend", choices=["ollama", "api", "mock"], default=None,
        help="LLM backend (default: from config, usually mock)",
    )
    parser.add_argument("--model", default=None, help="Ollama model name")
    parser.add_argument("--host", default=None, help="Ollama host")
    parser.add_argument("--api-url", default="", help="API URL (for api backend)")
    parser.add_argument("--api-key", default="", help="API key (for api backend)")
    parser.add_argument("--api-model", default="", help="API model name (for api backend)")
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
    )

    if args.port is not None:
        demo.feedback.port = args.port

    demo.run()


if __name__ == "__main__":
    main()
