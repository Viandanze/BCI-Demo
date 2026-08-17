"""
Real-Time Inference Pipeline for Online BCI Classification
Implements streaming data processing, preprocessing, and model inference

This module provides a complete real-time inference framework for online BCI systems:
- StreamingBuffer: Sliding window buffer for continuous EEG data
- PreprocessWorker: Asynchronous preprocessing pipeline
- InferenceEngine: Model inference with batch/single sample support
- PostProcessor: Post-processing (smoothing, voting, threshold filtering)
- RealTimePipeline: Complete pipeline coordinating all components
- MockDataStream: Simulated EEG data stream for testing

Author: BCI_Projects Team
"""

import copy
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Dict, List, Optional, Tuple, Union, Any, Callable
)
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.data.preprocessing import PreprocessingConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PredictionResult:
    """
    Result of a single prediction with timing and confidence information.
    
    Attributes:
        label: Predicted class label (int or str)
        confidence: Prediction confidence (0-1)
        probability: Full probability vector over classes
        timestamp: Unix timestamp when prediction was made
        latency_ms: Processing latency in milliseconds
        features: Optional extracted features for debugging
    """
    label: Union[int, str]
    confidence: float
    probability: np.ndarray
    timestamp: float
    latency_ms: float
    features: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        """String representation for logging/display."""
        return (
            f"Prediction(label={self.label}, "
            f"confidence={self.confidence:.3f}, "
            f"latency={self.latency_ms:.1f}ms, "
            f"time={self.timestamp:.3f})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'label': self.label,
            'confidence': float(self.confidence),
            'probability': self.probability.tolist(),
            'timestamp': self.timestamp,
            'latency_ms': self.latency_ms,
            'features': self.features,
        }


@dataclass
class StreamConfig:
    """Configuration for streaming data pipeline."""
    # Buffer settings
    window_size: float = 4.0  # Window duration in seconds
    step_size: float = 0.5  # Step between predictions in seconds
    overlap_ratio: float = 0.875  # 1 - step_size/window_size
    
    # Sampling rate
    sampling_rate: float = 128.0  # Hz
    
    # Channel configuration
    n_channels: int = 48
    channel_names: Optional[List[str]] = None
    
    # Preprocessing
    bandpass_low: float = 4.0
    bandpass_high: float = 38.0
    apply_normalization: bool = True
    
    # Inference
    batch_size: int = 32
    max_inference_time: float = 0.1  # Max time to wait for inference (seconds)
    
    # Post-processing
    smoothing_window: int = 5  # Number of predictions to average
    confidence_threshold: float = 0.6  # Minimum confidence to output
    use_voting: bool = True  # Use majority voting across smoothed predictions
    
    # Performance
    num_workers: int = 2  # Number of preprocessing workers
    queue_size: int = 100  # Max size of processing queues


# ============================================================================
# Streaming Buffer
# ============================================================================

class StreamingBuffer:
    """
    Thread-safe sliding window buffer for continuous EEG data.
    
    Maintains a rolling buffer of EEG data and provides windows
    for processing. Uses locks for thread safety.
    
    Args:
        window_size: Window duration in seconds
        sampling_rate: Sampling frequency in Hz
        n_channels: Number of EEG channels
        step_size: Step size for window extraction (seconds)
    """
    
    def __init__(
        self,
        window_size: float = 4.0,
        sampling_rate: float = 128.0,
        n_channels: int = 48,
        step_size: float = 0.5,
    ):
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.step_size = step_size
        
        # Calculate buffer sizes
        self.window_samples = int(window_size * sampling_rate)
        self.step_samples = int(step_size * sampling_rate)
        
        # Initialize buffer
        self.buffer = np.zeros((n_channels, self.window_samples * 2))
        self.buffer_fill = 0  # Number of filled samples
        self.buffer_lock = threading.Lock()
        
        # State tracking
        self.last_window_time = None
        self.total_samples_received = 0
        
        logger.info(
            f"StreamingBuffer initialized: "
            f"window={window_size}s ({self.window_samples} samples), "
            f"step={step_size}s ({self.step_samples} samples), "
            f"channels={n_channels}"
        )
    
    def push(self, data: np.ndarray) -> None:
        """
        Add new data to the buffer.
        
        Args:
            data: New EEG data of shape (n_channels, n_new_samples)
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_new_samples = data.shape[1]
        
        with self.buffer_lock:
            # Shift buffer to make room for new data
            if n_new_samples >= len(self.buffer[0]):
                # Data larger than buffer, keep only latest
                self.buffer = data[:, -len(self.buffer[0]):]
                self.buffer_fill = len(self.buffer[0])
            else:
                # Shift and append
                shift = min(n_new_samples, len(self.buffer[0]) - self.window_samples // 2)
                self.buffer[:, :-shift] = self.buffer[:, shift:]
                self.buffer[:, -n_new_samples:] = data
                self.buffer_fill = min(
                    self.buffer_fill + n_new_samples,
                    len(self.buffer[0])
                )
            
            self.total_samples_received += n_new_samples
    
    def get_window(self) -> Optional[np.ndarray]:
        """
        Get the current window for processing.
        
        Returns:
            Window data of shape (n_channels, window_samples) or None if not ready
        """
        with self.buffer_lock:
            if self.buffer_fill < self.window_samples:
                return None
            
            # Return the most recent window
            return self.buffer[:, -self.window_samples:].copy()
    
    def get_windows_ready(self) -> int:
        """Get number of windows ready for processing."""
        with self.buffer_lock:
            if self.buffer_fill < self.window_samples:
                return 0
            
            extra_samples = self.buffer_fill - self.window_samples
            return 1 + extra_samples // self.step_samples
    
    def is_ready(self) -> bool:
        """Check if buffer has enough data for processing."""
        with self.buffer_lock:
            return self.buffer_fill >= self.window_samples
    
    def reset(self) -> None:
        """Clear the buffer."""
        with self.buffer_lock:
            self.buffer = np.zeros((self.n_channels, self.window_samples * 2))
            self.buffer_fill = 0
            self.last_window_time = None
    
    @property
    def buffer_duration(self) -> float:
        """Get current buffer duration in seconds."""
        with self.buffer_lock:
            return self.buffer_fill / self.sampling_rate


# ============================================================================
# Preprocessing Worker
# ============================================================================

class PreprocessWorker:
    """
    Asynchronous EEG preprocessing worker.
    
    Runs preprocessing in a separate thread to avoid blocking inference.
    Supports filtering, normalization, and artifact rejection.
    
    Args:
        config: StreamConfig object with preprocessing parameters
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.input_queue: queue.Queue = queue.Queue(maxsize=config.queue_size)
        self.output_queue: queue.Queue = queue.Queue(maxsize=config.queue_size)
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Preprocessing state
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._filter_state: Dict[str, Any] = {}
        
        # Initialize filters
        self._init_filters()
    
    def _init_filters(self) -> None:
        """Initialize preprocessing filters."""
        try:
            from scipy.signal import butter, lfilter
            
            # Bandpass filter
            nyquist = self.config.sampling_rate / 2
            low = self.config.bandpass_low / nyquist
            high = self.config.bandpass_high / nyquist
            b, a = butter(4, [low, high], btype='band')
            self._filter_b = b
            self._filter_a = a
            self._filter_initialized = True
            
            logger.info(
                f"Preprocessing filters initialized: "
                f"{self.config.bandpass_low}-{self.config.bandpass_high} Hz"
            )
        except ImportError:
            logger.warning("scipy not available, skipping filtering")
            self._filter_initialized = False
    
    def start(self) -> None:
        """Start the preprocessing worker thread."""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._run, daemon=True)
        self.worker_thread.start()
        logger.info("PreprocessWorker started")
    
    def stop(self) -> None:
        """Stop the preprocessing worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        logger.info("PreprocessWorker stopped")
    
    def _run(self) -> None:
        """Worker thread main loop."""
        from scipy.signal import lfilter
        
        while self.running:
            try:
                # Get data from input queue with timeout
                try:
                    item = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                data, metadata = item
                start_time = time.time()
                
                # Apply preprocessing
                processed = data.copy()
                
                # 1. Filtering
                if self._filter_initialized:
                    for ch in range(processed.shape[0]):
                        processed[ch] = lfilter(
                            self._filter_b, 
                            self._filter_a, 
                            processed[ch]
                        )
                
                # 2. Normalization (running statistics)
                if self.config.apply_normalization:
                    if self._mean is None:
                        self._mean = processed.mean(axis=1, keepdims=True)
                        self._std = processed.std(axis=1, keepdims=True) + 1e-8
                    else:
                        # Update running statistics (EMA)
                        alpha = 0.1
                        self._mean = alpha * processed.mean(axis=1, keepdims=True) + \
                                    (1 - alpha) * self._mean
                        self._std = alpha * processed.std(axis=1, keepdims=True) + \
                                   (1 - alpha) * self._std
                    
                    processed = (processed - self._mean) / self._std
                
                # 3. Add metadata
                preprocess_time = (time.time() - start_time) * 1000
                
                result = {
                    'data': processed,
                    'original_shape': data.shape,
                    'preprocess_time_ms': preprocess_time,
                    'timestamp': metadata.get('timestamp', time.time()),
                }
                
                self.output_queue.put(result)
                self.input_queue.task_done()
                
            except Exception as e:
                logger.error(f"Preprocessing error: {e}")
                continue
    
    def submit(self, data: np.ndarray, metadata: Optional[Dict] = None) -> None:
        """
        Submit data for preprocessing.
        
        Args:
            data: EEG data of shape (n_channels, n_samples)
            metadata: Optional metadata dictionary
        """
        metadata = metadata or {}
        metadata.setdefault('timestamp', time.time())
        
        try:
            self.input_queue.put_nowait((data, metadata))
        except queue.Full:
            logger.warning("PreprocessWorker input queue full, dropping data")
    
    def get_result(self, timeout: float = 0.1) -> Optional[Dict]:
        """
        Get preprocessed result.
        
        Args:
            timeout: Maximum time to wait for result
            
        Returns:
            Preprocessed data dictionary or None if timeout
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def calibrate(self, calibration_data: np.ndarray) -> None:
        """
        Calibrate normalization parameters using calibration data.
        
        Args:
            calibration_data: Calibration EEG data
        """
        self._mean = calibration_data.mean(axis=1, keepdims=True)
        self._std = calibration_data.std(axis=1, keepdims=True) + 1e-8
        logger.info("Preprocessing calibration completed")


# ============================================================================
# Inference Engine
# ============================================================================

class InferenceEngine:
    """
    Model inference engine for real-time prediction.
    
    Supports loading and running PyTorch models for EEG classification.
    Provides both batch and single-sample inference modes.
    
    Args:
        model: PyTorch model or path to model checkpoint
        device: Device to run inference on ('cpu', 'cuda', 'auto')
        batch_size: Batch size for inference
    """
    
    def __init__(
        self,
        model: Union[nn.Module, str, Path],
        device: str = 'auto',
        batch_size: int = 32,
    ):
        self.batch_size = batch_size
        
        # Determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        if isinstance(model, (str, Path)):
            self.model = self._load_model(model)
        else:
            self.model = model
        
        self.model.to(self.device)
        self.model.eval()
        
        # Inference statistics
        self._inference_count = 0
        self._total_inference_time = 0.0
        
        # Get model info
        self.n_classes = self._get_n_classes()
        
        logger.info(f"InferenceEngine initialized on {self.device}")
        logger.info(f"Model: {type(self.model).__name__}, {self.n_classes} classes")
    
    def _load_model(self, model_path: Union[str, Path]) -> nn.Module:
        """Load model from checkpoint."""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Try loading full checkpoint first
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                # Extract state dict from checkpoint
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Try to create model from state dict keys
            # This is a simplified version - in practice, you'd need to 
            # know the model architecture
            raise ValueError(
                "Cannot infer model architecture from checkpoint. "
                "Please provide a configured model instance."
            )
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_n_classes(self) -> int:
        """Infer number of classes from model."""
        # Try to find classifier layer
        if hasattr(self.model, 'n_classes'):
            return self.model.n_classes
        elif hasattr(self.model, 'n_classes_'):
            return self.model.n_classes_
        elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'classifier'):
            # EEGNetClassifier structure
            classifier = self.model.backbone.classifier[-1]
            if hasattr(classifier, 'out_features'):
                return classifier.out_features
        
        # Default to 4-class MI
        return 4
    
    def predict(
        self,
        data: np.ndarray,
        return_proba: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on data.
        
        Args:
            data: Input data of shape (batch, channels, time) or (channels, time)
            return_proba: Whether to return probabilities
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        start_time = time.time()
        
        # Ensure 3D input
        if data.ndim == 2:
            data = data.unsqueeze(0)
        
        # Convert to tensor (ensure float32 - numpy defaults to float64)
        if isinstance(data, np.ndarray):
            data = data.astype(np.float32)
        elif isinstance(data, torch.Tensor) and data.dtype != torch.float32:
            data = data.float()
        X_tensor = torch.FloatTensor(data).to(self.device)
        
        # Inference
        with torch.no_grad():
            # Use model's forward directly (handles EEGNetClassifier correctly)
            outputs = self.model(X_tensor)
            
            # Get probabilities
            if outputs.shape[-1] > 1:
                proba = torch.softmax(outputs, dim=-1)
            else:
                proba = torch.sigmoid(outputs)
            
            preds = torch.argmax(proba, dim=-1)
        
        # Statistics
        inference_time = (time.time() - start_time) * 1000
        self._inference_count += len(data)
        self._total_inference_time += inference_time
        
        return preds.cpu().numpy(), proba.cpu().numpy()
    
    def predict_single(
        self,
        data: np.ndarray,
    ) -> Tuple[int, float, np.ndarray]:
        """
        Run inference on a single sample.
        
        Args:
            data: Input data of shape (channels, time)
            
        Returns:
            Tuple of (predicted_label, confidence, probability_vector)
        """
        preds, proba = self.predict(data.unsqueeze(0), return_proba=True)
        
        pred = int(preds[0])
        confidence = float(proba[0, pred])
        proba_vector = proba[0]
        
        return pred, confidence, proba_vector
    
    def get_stats(self) -> Dict[str, float]:
        """Get inference statistics."""
        avg_time = (
            self._total_inference_time / self._inference_count 
            if self._inference_count > 0 else 0
        )
        
        return {
            'total_inferences': self._inference_count,
            'avg_inference_time_ms': avg_time,
            'total_time_ms': self._total_inference_time,
        }
    
    def reset_stats(self) -> None:
        """Reset inference statistics."""
        self._inference_count = 0
        self._total_inference_time = 0.0


# ============================================================================
# Post Processor
# ============================================================================

class PostProcessor:
    """
    Post-processing for prediction results.
    
    Applies smoothing, voting, and confidence threshold filtering
    to improve prediction stability.
    
    Args:
        smoothing_window: Number of predictions to average
        confidence_threshold: Minimum confidence to output prediction
        use_voting: Use majority voting instead of probability averaging
    """
    
    def __init__(
        self,
        smoothing_window: int = 5,
        confidence_threshold: float = 0.6,
        use_voting: bool = True,
        n_classes: int = 4,
    ):
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        self.use_voting = use_voting
        self.n_classes = n_classes
        
        # Sliding window for predictions
        self.prediction_buffer: deque = deque(maxlen=smoothing_window)
        self.probability_buffer: deque = deque(maxlen=smoothing_window)
        
        # Last stable prediction (used when below threshold)
        self.last_stable_prediction: Optional[int] = None
        self.last_stable_confidence: float = 0.0
        
        logger.info(
            f"PostProcessor initialized: "
            f"window={smoothing_window}, "
            f"threshold={confidence_threshold}, "
            f"voting={use_voting}"
        )
    
    def add_prediction(
        self,
        label: int,
        confidence: float,
        probability: np.ndarray,
    ) -> Optional[PredictionResult]:
        """
        Add a prediction and get processed result.
        
        Args:
            label: Raw predicted label
            confidence: Prediction confidence
            probability: Full probability vector
            
        Returns:
            Processed PredictionResult or None if below threshold
        """
        timestamp = time.time()
        
        # Add to buffers
        self.prediction_buffer.append(label)
        self.probability_buffer.append(probability)
        
        # Process if buffer is full
        if len(self.prediction_buffer) < self.smoothing_window:
            return None
        
        # Apply smoothing/voting
        if self.use_voting:
            # Majority voting
            processed_label, processed_confidence = self._majority_vote()
        else:
            # Probability averaging
            processed_label, processed_confidence = self._average_probabilities()
        
        # Check confidence threshold
        if processed_confidence < self.confidence_threshold:
            # Return last stable prediction if available
            if self.last_stable_prediction is not None:
                return PredictionResult(
                    label=self.last_stable_prediction,
                    confidence=self.last_stable_confidence,
                    probability=probability,
                    timestamp=timestamp,
                    latency_ms=0.0,
                )
            return None
        
        # Update last stable
        self.last_stable_prediction = processed_label
        self.last_stable_confidence = processed_confidence
        
        return PredictionResult(
            label=int(processed_label),
            confidence=processed_confidence,
            probability=probability,
            timestamp=timestamp,
            latency_ms=0.0,
        )
    
    def _majority_vote(self) -> Tuple[int, float]:
        """Apply majority voting across buffered predictions."""
        predictions = list(self.prediction_buffer)
        
        # Count votes
        vote_counts = np.zeros(self.n_classes)
        for pred in predictions:
            vote_counts[pred] += 1
        
        # Get winner
        winner = np.argmax(vote_counts)
        confidence = vote_counts[winner] / len(predictions)
        
        return int(winner), float(confidence)
    
    def _average_probabilities(self) -> Tuple[int, float]:
        """Average probabilities across buffered predictions."""
        probas = np.array(list(self.probability_buffer))
        avg_proba = probas.mean(axis=0)
        
        winner = np.argmax(avg_proba)
        confidence = avg_proba[winner]
        
        return int(winner), float(confidence)
    
    def reset(self) -> None:
        """Reset buffers."""
        self.prediction_buffer.clear()
        self.probability_buffer.clear()
        self.last_stable_prediction = None
        self.last_stable_confidence = 0.0


# ============================================================================
# Real-Time Pipeline
# ============================================================================

class RealTimePipeline:
    """
    Complete real-time BCI inference pipeline.
    
    Coordinates buffer, preprocessing, inference, and post-processing
    for continuous online classification.
    
    Args:
        model: PyTorch model or path to model file
        config: StreamConfig object
        class_labels: Optional list of class label names
    """
    
    def __init__(
        self,
        model: Union[nn.Module, str, Path],
        config: Optional[StreamConfig] = None,
        class_labels: Optional[List[str]] = None,
    ):
        self.config = config or StreamConfig()
        self.class_labels = class_labels or ['left_hand', 'right_hand', 'feet', 'rest']
        
        # Initialize components
        self.buffer = StreamingBuffer(
            window_size=self.config.window_size,
            sampling_rate=self.config.sampling_rate,
            n_channels=self.config.n_channels,
            step_size=self.config.step_size,
        )
        
        self.preprocessor = PreprocessWorker(self.config)
        
        self.inference_engine = InferenceEngine(
            model=model,
            device='auto',
            batch_size=self.config.batch_size,
        )
        
        self.postprocessor = PostProcessor(
            smoothing_window=self.config.smoothing_window,
            confidence_threshold=self.config.confidence_threshold,
            use_voting=self.config.use_voting,
            n_classes=self.inference_engine.n_classes,
        )
        
        # State
        self.running = False
        self._process_thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable] = None
        
        # Statistics
        self.total_predictions = 0
        self.total_latency_ms = 0.0
        
        logger.info("RealTimePipeline initialized")
    
    def start(self, callback: Optional[Callable] = None) -> None:
        """
        Start the inference pipeline.
        
        Args:
            callback: Optional callback function for receiving predictions
        """
        if self.running:
            return
        
        self.running = True
        self._callback = callback
        
        # Start preprocessor
        self.preprocessor.start()
        
        # Start processing thread
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()
        
        logger.info("RealTimePipeline started")
    
    def stop(self) -> None:
        """Stop the inference pipeline."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop preprocessor
        self.preprocessor.stop()
        
        # Wait for process thread
        if self._process_thread:
            self._process_thread.join(timeout=2.0)
        
        logger.info("RealTimePipeline stopped")
    
    def _process_loop(self) -> None:
        """Main processing loop."""
        last_process_time = 0.0
        step_interval = self.config.step_size
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time to process
                if current_time - last_process_time < step_interval:
                    time.sleep(0.01)
                    continue
                
                # Check if window is ready
                window = self.buffer.get_window()
                if window is None:
                    time.sleep(0.01)
                    continue
                
                # Submit for preprocessing
                self.preprocessor.submit(window, {'timestamp': current_time})
                
                # Wait for preprocessing to complete
                result = self.preprocessor.get_result(timeout=0.1)
                
                if result is None:
                    continue
                
                # Run inference
                preprocessed = result['data']
                preprocess_time = result['preprocess_time_ms']
                
                inference_start = time.time()
                label, confidence, proba = self.inference_engine.predict_single(
                    torch.from_numpy(preprocessed.astype(np.float32))
                )
                inference_time = (time.time() - inference_start) * 1000
                
                total_latency = preprocess_time + inference_time
                
                # Post-process
                pred_result = self.postprocessor.add_prediction(
                    label=label,
                    confidence=confidence,
                    probability=proba,
                )
                
                if pred_result is not None:
                    pred_result.latency_ms = total_latency
                    self.total_predictions += 1
                    self.total_latency_ms += total_latency
                    
                    # Call callback if provided
                    if self._callback:
                        self._callback(pred_result)
                
                last_process_time = current_time
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
                continue
    
    def push_data(self, data: np.ndarray) -> None:
        """
        Push new EEG data into the pipeline.
        
        Args:
            data: EEG data of shape (n_channels, n_samples)
        """
        self.buffer.push(data)
    
    def get_prediction(self) -> Optional[PredictionResult]:
        """
        Get the latest prediction result.
        
        Returns:
            Latest PredictionResult or None if no new prediction
        """
        # This is a blocking call - in practice, use callback instead
        return None
    
    def calibrate(self, calibration_data: np.ndarray) -> None:
        """
        Calibrate the pipeline using calibration data.
        
        Args:
            calibration_data: Calibration EEG data
        """
        self.preprocessor.calibrate(calibration_data)
        logger.info("Pipeline calibration completed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        avg_latency = (
            self.total_latency_ms / self.total_predictions 
            if self.total_predictions > 0 else 0
        )
        
        return {
            'total_predictions': self.total_predictions,
            'avg_latency_ms': avg_latency,
            'buffer_fill_s': self.buffer.buffer_duration,
            'inference_stats': self.inference_engine.get_stats(),
        }
    
    def __enter__(self) -> 'RealTimePipeline':
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


# ============================================================================
# Mock Data Stream
# ============================================================================

class MockDataStream:
    """
    Mock EEG data stream for testing and demonstration.
    
    Generates simulated motor imagery EEG signals with realistic
    alpha/beta rhythms and event-related changes.
    
    Args:
        config: StreamConfig object
        n_classes: Number of motor imagery classes (default: 4)
        event_interval: Average interval between events (seconds)
    """
    
    def __init__(
        self,
        config: Optional[StreamConfig] = None,
        n_classes: int = 4,
        event_interval: float = 10.0,
    ):
        self.config = config or StreamConfig()
        self.n_classes = n_classes
        self.event_interval = event_interval
        
        # Simulation state
        self.running = False
        self._stream_thread: Optional[threading.Thread] = None
        self._data_callback: Optional[Callable] = None
        
        # Event simulation
        self._current_class = 0
        self._event_start_time = 0.0
        self._event_duration = 4.0  # seconds
        self._last_event_time = 0.0
        
        # Signal generation state
        np.random.seed(int(time.time()))
        self._phase = np.random.rand(config.n_channels) * 2 * np.pi
        
        logger.info(f"MockDataStream initialized: {n_classes} classes")
    
    def start(
        self,
        data_callback: Callable[[np.ndarray], None],
    ) -> None:
        """
        Start the mock data stream.
        
        Args:
            data_callback: Callback function to receive data chunks
        """
        if self.running:
            return
        
        self.running = True
        self._data_callback = data_callback
        
        self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._stream_thread.start()
        
        logger.info("MockDataStream started")
    
    def stop(self) -> None:
        """Stop the mock data stream."""
        if not self.running:
            return
        
        self.running = False
        
        if self._stream_thread:
            self._stream_thread.join(timeout=2.0)
        
        logger.info("MockDataStream stopped")
    
    def _stream_loop(self) -> None:
        """Main streaming loop."""
        samples_per_chunk = int(self.config.sampling_rate * self.config.step_size)
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check for new event
                if current_time - self._last_event_time > self.event_interval:
                    self._trigger_new_event(current_time)
                
                # Generate data chunk
                data = self._generate_chunk(samples_per_chunk)
                
                # Send to callback
                if self._data_callback:
                    self._data_callback(data)
                
                # Sleep to simulate real-time data
                time.sleep(self.config.step_size)
                
            except Exception as e:
                logger.error(f"Stream error: {e}")
                continue
    
    def _trigger_new_event(self, current_time: float) -> None:
        """Trigger a new motor imagery event."""
        self._current_class = np.random.randint(0, self.n_classes)
        self._event_start_time = current_time
        self._last_event_time = current_time
        
        logger.debug(f"New event: class {self._current_class}")
    
    def _generate_chunk(self, n_samples: int) -> np.ndarray:
        """
        Generate a chunk of simulated EEG data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Generated data of shape (n_channels, n_samples)
        """
        current_time = time.time()
        t = np.arange(n_samples) / self.config.sampling_rate
        
        # Base signal parameters
        n_channels = self.config.n_channels
        sfreq = self.config.sampling_rate
        
        # Initialize with background noise
        data = np.random.randn(n_channels, n_samples).astype(np.float32) * 5
        
        # Add alpha rhythm (8-12 Hz) - strongest at occipital/parietal
        alpha_freq = 10.0
        for ch in range(n_channels):
            # Vary alpha power by channel (stronger at back)
            alpha_power = 10 * (1 - ch / n_channels * 0.5)
            
            # Event-related modulation
            if self._is_during_event(current_time):
                # ERD/ERS during motor imagery
                if self._current_class == 0 and ch in range(8, 16):  # Right motor
                    alpha_power *= 0.5  # Desynchronization
                elif self._current_class == 1 and ch in range(0, 8):  # Left motor
                    alpha_power *= 0.5
            
            alpha = alpha_power * np.sin(
                2 * np.pi * alpha_freq * t + self._phase[ch]
            )
            
            # Add harmonics
            beta_freq = 20.0
            beta = 5 * np.sin(2 * np.pi * beta_freq * t + self._phase[ch] * 1.5)
            
            data[ch] += alpha + beta
        
        # Update phase for next chunk
        self._phase = (self._phase + 2 * np.pi * alpha_freq * self.config.step_size) % (2 * np.pi)
        
        return data
    
    def _is_during_event(self, current_time: float) -> bool:
        """Check if currently during an event."""
        return (
            current_time >= self._event_start_time and
            current_time <= self._event_start_time + self._event_duration
        )
    
    def get_current_class(self) -> int:
        """Get the currently simulated motor imagery class."""
        return self._current_class
    
    def __enter__(self) -> 'MockDataStream':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


# ============================================================================
# Utility Functions
# ============================================================================

def create_pipeline_from_checkpoint(
    checkpoint_path: Union[str, Path],
    config: Optional[StreamConfig] = None,
    model_class: Optional[type] = None,
    model_kwargs: Optional[Dict] = None,
    **kwargs,
) -> RealTimePipeline:
    """
    Create a RealTimePipeline from a model checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Optional StreamConfig
        model_class: Model class (e.g., EEGNetClassifier)
        model_kwargs: Arguments for model constructor
        **kwargs: Additional arguments
        
    Returns:
        Configured RealTimePipeline
    """
    from src.models.eegnet import EEGNetClassifier
    
    # Default model kwargs
    model_kwargs = model_kwargs or {}
    model_kwargs.setdefault('n_channels', config.n_channels if config else 48)
    model_kwargs.setdefault('n_times', int(4 * 128))  # 4 seconds at 128 Hz
    model_kwargs.setdefault('n_classes', 4)
    
    # Create model
    model = model_class(**model_kwargs) if model_class else EEGNetClassifier(**model_kwargs)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return RealTimePipeline(model=model, config=config, **kwargs)


def print_prediction(
    result: PredictionResult,
    class_labels: Optional[List[str]] = None,
    show_proba: bool = False,
) -> None:
    """
    Print a formatted prediction result.
    
    Args:
        result: PredictionResult object
        class_labels: Optional class label names
        show_proba: Whether to show full probability vector
    """
    labels = class_labels or ['Left', 'Right', 'Feet', 'Rest']
    
    label_name = labels[result.label] if result.label < len(labels) else str(result.label)
    
    print(
        f"[{result.timestamp:.3f}] "
        f"Prediction: {label_name} "
        f"(conf: {result.confidence:.3f}, "
        f"latency: {result.latency_ms:.1f}ms)"
    )
    
    if show_proba:
        proba_str = ", ".join([f"{p:.2f}" for p in result.probability])
        print(f"  Probabilities: [{proba_str}]")


# Export
__all__ = [
    "PredictionResult",
    "StreamConfig",
    "StreamingBuffer",
    "PreprocessWorker",
    "InferenceEngine",
    "PostProcessor",
    "RealTimePipeline",
    "MockDataStream",
    "create_pipeline_from_checkpoint",
    "print_prediction",
]
