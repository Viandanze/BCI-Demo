#!/usr/bin/env python3
"""
Real-Time BCI Inference Demo Script
Demonstrates online EEG classification with simulated streaming data

Features:
- Mock data stream for testing without hardware
- Real-time prediction with latency tracking
- Visualization of prediction confidence
- Configurable window size and preprocessing

Examples:
    # Run with mock data stream
    python scripts/realtime_demo.py --mock --duration 60
    
    # Run with trained model
    python scripts/realtime_demo.py --model_path models/eegnet.pt --duration 120
    
    # Custom configuration
    python scripts/realtime_demo.py --mock --window_size 3.0 --smoothing_window 10
"""

import argparse
import logging
import os
import sys
import time
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.pipeline import (
    PredictionResult, StreamConfig,
    RealTimePipeline, MockDataStream,
    create_pipeline_from_checkpoint, print_prediction
)
from src.models.eegnet import EEGNetClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Real-time BCI inference demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo with mock data stream
  python scripts/realtime_demo.py --mock
  
  # With trained model
  python scripts/realtime_demo.py --model_path ./models/eegnet.pt
  
  # Custom window and duration
  python scripts/realtime_demo.py --mock --window_size 3.0 --duration 30
        """
    )
    
    # Model arguments
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument(
        '--model_path', type=str, default=None,
        help='Path to trained model checkpoint (.pt file)'
    )
    model_group.add_argument(
        '--n_channels', type=int, default=48,
        help='Number of EEG channels'
    )
    model_group.add_argument(
        '--n_times', type=int, default=512,
        help='Number of time points in window'
    )
    model_group.add_argument(
        '--n_classes', type=int, default=4,
        help='Number of classes'
    )
    model_group.add_argument(
        '--class_labels', type=str, nargs='+',
        default=['left_hand', 'right_hand', 'feet', 'rest'],
        help='Class label names'
    )
    
    # Stream arguments
    stream_group = parser.add_argument_group('Stream Options')
    stream_group.add_argument(
        '--mock', action='store_true',
        help='Use mock data stream for testing'
    )
    stream_group.add_argument(
        '--sampling_rate', type=float, default=128.0,
        help='EEG sampling rate (Hz)'
    )
    stream_group.add_argument(
        '--window_size', type=float, default=4.0,
        help='Sliding window duration (seconds)'
    )
    stream_group.add_argument(
        '--step_size', type=float, default=0.5,
        help='Step between predictions (seconds)'
    )
    stream_group.add_argument(
        '--bandpass_low', type=float, default=4.0,
        help='Bandpass filter low cutoff (Hz)'
    )
    stream_group.add_argument(
        '--bandpass_high', type=float, default=38.0,
        help='Bandpass filter high cutoff (Hz)'
    )
    
    # Post-processing arguments
    post_group = parser.add_argument_group('Post-processing Options')
    post_group.add_argument(
        '--smoothing_window', type=int, default=5,
        help='Number of predictions to smooth'
    )
    post_group.add_argument(
        '--confidence_threshold', type=float, default=0.6,
        help='Minimum confidence threshold'
    )
    post_group.add_argument(
        '--use_voting', action='store_true', default=True,
        help='Use majority voting instead of probability averaging'
    )
    post_group.add_argument(
        '--no_voting', action='store_false', dest='use_voting',
        help='Disable majority voting'
    )
    
    # Demo arguments
    demo_group = parser.add_argument_group('Demo Options')
    demo_group.add_argument(
        '--duration', type=float, default=60.0,
        help='Demo duration in seconds'
    )
    demo_group.add_argument(
        '--event_interval', type=float, default=8.0,
        help='Average interval between motor imagery events (seconds)'
    )
    demo_group.add_argument(
        '--show_proba', action='store_true',
        help='Show full probability vector'
    )
    demo_group.add_argument(
        '--verbose', action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


# ============================================================================
# Model Creation
# ============================================================================

def create_demo_model(
    n_channels: int,
    n_times: int,
    n_classes: int,
) -> torch.nn.Module:
    """
    Create a demo model for testing.
    
    If a trained model path is provided, loads from checkpoint.
    Otherwise, creates a simple EEGNet-like model.
    
    Args:
        n_channels: Number of EEG channels
        n_times: Number of time points
        n_classes: Number of classes
        
    Returns:
        Model instance
    """
    logger.info("Creating demo model...")
    
    # Create EEGNet-like model
    model = EEGNetClassifier(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        F1=8,
        D=2,
        kernel_length=64,
        dropout_rate=0.5,
    )
    
    # Initialize with random weights (for demo)
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)
    
    logger.info(f"Demo model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    return model


def load_trained_model(
    model_path: str,
    n_channels: int,
    n_times: int,
    n_classes: int,
) -> torch.nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        n_channels: Number of EEG channels
        n_times: Number of time points
        n_classes: Number of classes
        
    Returns:
        Loaded model
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info(f"Loading trained model from: {model_path}")
    
    # Create model
    model = EEGNetClassifier(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    logger.info("Model loaded successfully")
    
    return model


# ============================================================================
# Prediction Handler
# ============================================================================

class PredictionHandler:
    """
    Handles prediction callbacks for the demo.
    
    Maintains prediction history and prints formatted results.
    """
    
    def __init__(
        self,
        class_labels: List[str],
        show_proba: bool = False,
        log_interval: float = 1.0,
    ):
        self.class_labels = class_labels
        self.show_proba = show_proba
        self.log_interval = log_interval
        
        self.predictions: List[PredictionResult] = []
        self.last_print_time = 0.0
        self._lock = threading.Lock()
        
        # Statistics
        self.prediction_counts = {label: 0 for label in class_labels}
        self.total_predictions = 0
        self.avg_confidence = 0.0
        self.avg_latency = 0.0
    
    def __call__(self, result: PredictionResult) -> None:
        """Handle prediction callback."""
        with self._lock:
            self.predictions.append(result)
            
            # Update statistics
            self.total_predictions += 1
            label_name = self.class_labels[result.label] if result.label < len(self.class_labels) else str(result.label)
            self.prediction_counts[label_name] += 1
            
            # Running average
            self.avg_confidence = (self.avg_confidence * (self.total_predictions - 1) + result.confidence) / self.total_predictions
            self.avg_latency = (self.avg_latency * (self.total_predictions - 1) + result.latency_ms) / self.total_predictions
            
            # Print at interval
            current_time = time.time()
            if current_time - self.last_print_time >= self.log_interval:
                self._print_summary()
                self.last_print_time = current_time
    
    def _print_summary(self) -> None:
        """Print prediction summary."""
        if not self.predictions:
            return
        
        # Get last prediction
        last = self.predictions[-1]
        label_name = self.class_labels[last.label] if last.label < len(self.class_labels) else str(last.label)
        
        # Format output
        print(f"\n[{last.timestamp:.2f}] {label_name:12s} | "
              f"conf: {last.confidence:.3f} | "
              f"latency: {last.latency_ms:.1f}ms")
        
        if self.show_proba:
            proba_str = " | ".join([
                f"{self.class_labels[i]}: {p:.2f}"
                for i, p in enumerate(last.probability)
                if i < len(self.class_labels)
            ])
            print(f"  Probabilities: {proba_str}")
        
        # Print running statistics
        print(f"  [Stats] predictions: {self.total_predictions} | "
              f"avg_conf: {self.avg_confidence:.3f} | "
              f"avg_latency: {self.avg_latency:.1f}ms")
    
    def print_final_summary(self) -> None:
        """Print final summary at end of demo."""
        print("\n" + "=" * 60)
        print("DEMO COMPLETE - FINAL SUMMARY")
        print("=" * 60)
        
        print(f"\nTotal Predictions: {self.total_predictions}")
        print(f"Average Confidence: {self.avg_confidence:.4f}")
        print(f"Average Latency: {self.avg_latency:.2f}ms")
        
        print("\nPrediction Distribution:")
        for label, count in self.prediction_counts.items():
            pct = (count / self.total_predictions * 100) if self.total_predictions > 0 else 0
            bar = "#" * int(pct / 5)
            print(f"  {label:12s}: {count:5d} ({pct:5.1f}%) {bar}")


# ============================================================================
# Main Demo
# ============================================================================

def run_demo(args: argparse.Namespace) -> int:
    """
    Run the real-time BCI demo.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success)
    """
    logger.info("=" * 60)
    logger.info("Real-Time BCI Inference Demo")
    logger.info("=" * 60)
    
    # Create stream config
    stream_config = StreamConfig(
        window_size=args.window_size,
        step_size=args.step_size,
        sampling_rate=args.sampling_rate,
        n_channels=args.n_channels,
        bandpass_low=args.bandpass_low,
        bandpass_high=args.bandpass_high,
        smoothing_window=args.smoothing_window,
        confidence_threshold=args.confidence_threshold,
        use_voting=args.use_voting,
        batch_size=32,
    )
    
    logger.info(f"Stream Configuration:")
    logger.info(f"  Window size: {stream_config.window_size}s")
    logger.info(f"  Step size: {stream_config.step_size}s")
    logger.info(f"  Sampling rate: {stream_config.sampling_rate}Hz")
    logger.info(f"  Channels: {stream_config.n_channels}")
    logger.info(f"  Smoothing window: {stream_config.smoothing_window}")
    logger.info(f"  Confidence threshold: {stream_config.confidence_threshold}")
    logger.info(f"  Use voting: {stream_config.use_voting}")
    
    # Create or load model
    if args.model_path:
        model = load_trained_model(
            args.model_path,
            args.n_channels,
            args.n_times,
            args.n_classes,
        )
    else:
        model = create_demo_model(
            args.n_channels,
            args.n_times,
            args.n_classes,
        )
    
    # Create prediction handler
    handler = PredictionHandler(
        class_labels=args.class_labels,
        show_proba=args.show_proba,
        log_interval=1.0,
    )
    
    # Create pipeline
    pipeline = RealTimePipeline(
        model=model,
        config=stream_config,
        class_labels=args.class_labels,
    )
    
    # Create mock data stream if requested
    if args.mock:
        logger.info("\nUsing mock data stream for testing...")
        data_stream = MockDataStream(
            config=stream_config,
            n_classes=args.n_classes,
            event_interval=args.event_interval,
        )
    else:
        data_stream = None
        logger.warning("No data stream available. Use --mock for demo.")
    
    # Run demo
    try:
        # Start pipeline
        pipeline.start(callback=handler)
        
        if data_stream:
            # Start data stream
            data_stream.start(pipeline.push_data)
            
            # Run for specified duration
            logger.info(f"\nRunning demo for {args.duration} seconds...")
            logger.info("Press Ctrl+C to stop early\n")
            
            start_time = time.time()
            try:
                while time.time() - start_time < args.duration:
                    time.sleep(0.5)
                    
                    # Log buffer status
                    if args.verbose:
                        buffer_fill = pipeline.buffer.buffer_duration
                        logger.debug(f"Buffer: {buffer_fill:.2f}s / {args.window_size}s")
                        
            except KeyboardInterrupt:
                logger.info("\nInterrupted by user")
        
        else:
            # No data stream - just show pipeline setup
            logger.info("\nPipeline ready. Connect a data source to begin.")
            time.sleep(args.duration)
    
    except Exception as e:
        logger.error(f"Demo error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Stop everything
        logger.info("\nStopping demo...")
        
        if data_stream:
            data_stream.stop()
        
        pipeline.stop()
        
        # Print final summary
        handler.print_final_summary()
        
        # Print pipeline stats
        stats = pipeline.get_stats()
        logger.info("\nPipeline Statistics:")
        logger.info(f"  Total predictions: {stats['total_predictions']}")
        logger.info(f"  Average latency: {stats['avg_latency_ms']:.2f}ms")
        
        inference_stats = stats.get('inference_stats', {})
        if inference_stats:
            logger.info(f"  Total inferences: {inference_stats.get('total_inferences', 0)}")
            logger.info(f"  Avg inference time: {inference_stats.get('avg_inference_time_ms', 0):.2f}ms")
    
    return 0


def interactive_demo(args: argparse.Namespace) -> int:
    """
    Run an interactive demo with user control.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    logger.info("=" * 60)
    logger.info("Interactive BCI Demo Mode")
    logger.info("=" * 60)
    
    print("\nCommands:")
    print("  start - Start the pipeline")
    print("  stop  - Stop the pipeline")
    print("  stats - Show statistics")
    print("  reset - Reset statistics")
    print("  quit  - Exit demo")
    print()
    
    # Create pipeline (same as before)
    stream_config = StreamConfig(
        window_size=args.window_size,
        step_size=args.step_size,
        sampling_rate=args.sampling_rate,
        n_channels=args.n_channels,
        bandpass_low=args.bandpass_low,
        bandpass_high=args.bandpass_high,
        smoothing_window=args.smoothing_window,
        confidence_threshold=args.confidence_threshold,
        use_voting=args.use_voting,
    )
    
    model = create_demo_model(args.n_channels, args.n_times, args.n_classes)
    
    handler = PredictionHandler(
        class_labels=args.class_labels,
        show_proba=args.show_proba,
    )
    
    pipeline = RealTimePipeline(
        model=model,
        config=stream_config,
        class_labels=args.class_labels,
    )
    
    data_stream = None
    if args.mock:
        data_stream = MockDataStream(
            config=stream_config,
            n_classes=args.n_classes,
            event_interval=args.event_interval,
        )
    
    running = False
    
    while True:
        try:
            cmd = input("\nbci_demo> ").strip().lower()
            
            if cmd == 'start':
                if not running:
                    pipeline.start(callback=handler)
                    if data_stream:
                        data_stream.start(pipeline.push_data)
                    running = True
                    print("Pipeline started")
                else:
                    print("Pipeline already running")
            
            elif cmd == 'stop':
                if running:
                    if data_stream:
                        data_stream.stop()
                    pipeline.stop()
                    running = False
                    print("Pipeline stopped")
                else:
                    print("Pipeline not running")
            
            elif cmd == 'stats':
                handler._print_summary()
                stats = pipeline.get_stats()
                print(f"\nPipeline stats: {stats}")
            
            elif cmd == 'reset':
                handler.predictions.clear()
                handler.total_predictions = 0
                handler.prediction_counts = {label: 0 for label in args.class_labels}
                handler.avg_confidence = 0.0
                handler.avg_latency = 0.0
                print("Statistics reset")
            
            elif cmd == 'quit' or cmd == 'exit' or cmd == 'q':
                if running:
                    if data_stream:
                        data_stream.stop()
                    pipeline.stop()
                print("Goodbye!")
                break
            
            else:
                print(f"Unknown command: {cmd}")
        
        except KeyboardInterrupt:
            print("\nInterrupted. Use 'quit' to exit.")
        except EOFError:
            break
    
    return 0


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run demo
    return run_demo(args)


if __name__ == '__main__':
    sys.exit(main())
