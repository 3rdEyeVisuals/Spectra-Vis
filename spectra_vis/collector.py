"""
Spectra Vis - Tensor Collector Module

This module provides the core functionality for capturing tensor observations
during LLM inference. It uses the cb_eval callback mechanism provided by
llama.cpp to count tensor evaluations as they occur.

How it works:
1. A callback function is registered with the llama.cpp context
2. During inference, llama.cpp calls this callback for each tensor
3. The callback records the tensor address and observation order
4. The frontend uses model profiles to map addresses to visualization

Requirements:
- llama-cpp-python with cb_eval patch (see README for instructions)

Copyright (c) 2025 3rdEyeVisuals
Licensed under the Spectra Vis License - see LICENSE file
"""

import ctypes
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field


# -----------------------------------------------------------------------------
# Trial Data Structure
# -----------------------------------------------------------------------------

@dataclass
class TrialData:
    """
    Data collected during a single inference run (trial).

    Attributes:
        trial_id: Sequential identifier for this trial
        prompt: The input text that was processed
        response: The generated output text
        inference_time_ms: Time taken for inference in milliseconds
        tensor_counts: Dict mapping tensor address to observation count
        tensor_order: List of tensor addresses in order of first observation
    """
    trial_id: int
    prompt: str = ""
    response: str = ""
    inference_time_ms: float = 0.0
    tensor_counts: Dict[str, int] = field(default_factory=dict)
    tensor_order: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Tensor Collector Class
# -----------------------------------------------------------------------------

class TensorCollector:
    """
    Collects tensor observation data during LLM inference.

    This class provides the core functionality for:
    - Counting tensor evaluations during inference via cb_eval callback
    - Recording observation order (useful for layer mapping)
    - Organizing data by trials for comparison
    - Exporting results to JSON for visualization

    The collector does NOT interpret tensor names - it records memory
    addresses and observation counts. The frontend visualization uses
    model profiles to map this data to meaningful tensor names.

    Usage:
        collector = TensorCollector(model_family="llama")
        callback = create_callback(collector)

        # Pass callback to Llama model (requires patched llama-cpp-python)
        model = Llama(model_path="...", cb_eval=callback)

        # Run inference
        collector.start_trial("Hello")
        response = model("Hello", max_tokens=50)
        collector.end_trial(response)

        # Export results
        collector.save_to_json("results.json")
    """

    def __init__(self, model_family: str = "llama", verbose: bool = False):
        """
        Initialize the tensor collector.

        Args:
            model_family: Model architecture family (for metadata only)
            verbose: Enable debug output
        """
        self.model_family = model_family.lower()
        self.verbose = verbose
        self.enabled = False

        # Global tensor tracking (across all trials)
        self.tensor_counts: Dict[str, int] = {}
        self.tensor_order: List[str] = []

        # Trial management
        self.trials: List[TrialData] = []
        self.current_trial: Optional[TrialData] = None
        self._trial_start_time: float = 0.0

        # Statistics
        self.total_callbacks = 0

        # Thread safety
        self._lock = threading.Lock()

    def enable(self) -> None:
        """Enable tensor collection."""
        self.enabled = True
        if self.verbose:
            print("[Spectra Vis] Collection enabled")

    def disable(self) -> None:
        """Disable tensor collection."""
        self.enabled = False
        if self.verbose:
            print("[Spectra Vis] Collection disabled")

    def clear(self) -> None:
        """Clear all collected data."""
        with self._lock:
            self.tensor_counts.clear()
            self.tensor_order.clear()
            self.trials.clear()
            self.current_trial = None
            self.total_callbacks = 0

    def start_trial(self, prompt: str = "") -> None:
        """
        Start a new trial for data collection.

        Args:
            prompt: The input prompt for this trial
        """
        import time

        with self._lock:
            trial_id = len(self.trials)
            self.current_trial = TrialData(
                trial_id=trial_id,
                prompt=prompt
            )
            self._trial_start_time = time.time()
            self.enable()

            if self.verbose:
                print(f"[Spectra Vis] Started trial {trial_id}")

    def end_trial(self, response: str = "") -> TrialData:
        """
        End the current trial and store results.

        Args:
            response: The generated response text

        Returns:
            The completed TrialData object
        """
        import time

        with self._lock:
            self.disable()

            if self.current_trial is None:
                raise RuntimeError("No active trial to end")

            # Calculate inference time
            elapsed = (time.time() - self._trial_start_time) * 1000
            self.current_trial.inference_time_ms = elapsed
            self.current_trial.response = response

            # Store trial
            self.trials.append(self.current_trial)
            completed_trial = self.current_trial
            self.current_trial = None

            if self.verbose:
                unique = len(completed_trial.tensor_counts)
                total = sum(completed_trial.tensor_counts.values())
                print(f"[Spectra Vis] Ended trial {completed_trial.trial_id}: "
                      f"{unique} unique tensors, {total} total observations, "
                      f"{elapsed:.1f}ms")

            return completed_trial

    def record_tensor(self, tensor_ptr: int, is_ask: bool) -> bool:
        """
        Record a tensor observation from the cb_eval callback.

        This method is called by the callback function for each tensor
        during graph evaluation.

        Args:
            tensor_ptr: Pointer address of the tensor
            is_ask: If True, llama.cpp is asking if we want this tensor.
                   If False, the tensor has been computed.

        Returns:
            True to continue (and request data if is_ask), False to skip
        """
        if not self.enabled:
            return True

        with self._lock:
            self.total_callbacks += 1

            if is_ask:
                # Always request to observe tensors
                return True

            # Data phase: record tensor address
            addr = f"tensor_{tensor_ptr:016x}"

            # Track observation order (first time we see this address)
            if addr not in self.tensor_counts:
                self.tensor_order.append(addr)
                self.tensor_counts[addr] = 0
            self.tensor_counts[addr] += 1

            # Update current trial
            if self.current_trial is not None:
                if addr not in self.current_trial.tensor_counts:
                    self.current_trial.tensor_order.append(addr)
                    self.current_trial.tensor_counts[addr] = 0
                self.current_trial.tensor_counts[addr] += 1

            return True

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for collected data.

        Returns:
            Dict with collection statistics
        """
        with self._lock:
            return {
                "total_callbacks": self.total_callbacks,
                "unique_tensors": len(self.tensor_counts),
                "total_trials": len(self.trials),
                "model_family": self.model_family,
            }

    def save_to_json(self, filepath: str) -> str:
        """
        Save collected data to a JSON file.

        Args:
            filepath: Path to save the JSON file

        Returns:
            The filepath that was saved to
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "tool": "Spectra Vis (Tensor Edition)",
            "author": "3rdEyeVisuals",
            "created": datetime.now().isoformat(),
            "model_family": self.model_family,
            "statistics": {
                "total_trials": len(self.trials),
                "unique_tensors": len(self.tensor_counts),
                "total_callbacks": self.total_callbacks
            },
            "tensor_order": self.tensor_order,
            "trials": [
                {
                    "trial_id": t.trial_id,
                    "prompt": t.prompt,
                    "response": t.response,
                    "inference_time_ms": t.inference_time_ms,
                    "tensor_counts": t.tensor_counts,
                    "tensor_order": t.tensor_order
                }
                for t in self.trials
            ]
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        if self.verbose:
            print(f"[Spectra Vis] Saved to {path}")

        return str(path)


# -----------------------------------------------------------------------------
# Callback Factory
# -----------------------------------------------------------------------------

# Global collector reference (required for ctypes callback)
_global_collector: Optional[TensorCollector] = None


def create_callback(collector: TensorCollector) -> Callable:
    """
    Create a cb_eval callback function for the given collector.

    The callback is a ctypes function that can be passed to llama.cpp
    via the cb_eval parameter. It will call the collector's record_tensor
    method for each tensor during inference.

    Args:
        collector: The TensorCollector to use for recording

    Returns:
        A ctypes callback function suitable for cb_eval

    Example:
        collector = TensorCollector()
        callback = create_callback(collector)
        model = Llama(model_path="...", cb_eval=callback)
    """
    global _global_collector
    _global_collector = collector

    # Import llama_cpp for the callback type
    try:
        from llama_cpp import llama_cpp
        callback_type = llama_cpp.ggml_backend_sched_eval_callback
    except ImportError:
        # Fallback: define the type ourselves
        callback_type = ctypes.CFUNCTYPE(
            ctypes.c_bool,      # return type
            ctypes.c_void_p,    # tensor pointer
            ctypes.c_bool,      # is_ask flag
            ctypes.c_void_p     # user_data
        )

    @callback_type
    def tensor_callback(tensor_ptr, is_ask, user_data):
        """
        Callback function invoked by llama.cpp during tensor evaluation.

        Args:
            tensor_ptr: Pointer to tensor being evaluated
            is_ask: True if asking whether to observe, False when computed
            user_data: Custom user data (not used)

        Returns:
            True to continue and observe tensor, False to skip
        """
        global _global_collector

        try:
            if _global_collector is None:
                return True
            return _global_collector.record_tensor(tensor_ptr, is_ask)
        except Exception:
            # Never crash inference due to callback errors
            return True

    return tensor_callback
