"""
Spectra Vis - Basic Capture Example

This example demonstrates how to capture tensor observations
during LLM inference and save them to a JSON file.

Requirements:
- llama-cpp-python with cb_eval patch (see README)
- A GGUF model file

Copyright (c) 2025 3rdEyeVisuals
Licensed under the Spectra Vis License - see LICENSE file
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectra_vis import TensorCollector, create_callback

# Try to import llama_cpp
try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python is not installed.")
    print("Install it with: pip install llama-cpp-python")
    sys.exit(1)


def main():
    """Run a basic tensor capture session."""

    # =========================================================================
    # Configuration - Update these for your setup
    # =========================================================================

    MODEL_PATH = "path/to/your/model.gguf"  # <-- UPDATE THIS
    MODEL_FAMILY = "llama"  # Options: llama, granite, qwen, phi, mistral
    OUTPUT_FILE = "capture_data.json"

    # Test prompts
    PROMPTS = [
        "What is the capital of France?",
        "Explain photosynthesis in simple terms.",
        "Write a haiku about programming.",
    ]

    # =========================================================================
    # Setup
    # =========================================================================

    print("=" * 60)
    print("Spectra Vis - Basic Capture Example")
    print("=" * 60)
    print()

    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please update MODEL_PATH in this script.")
        sys.exit(1)

    # Create the collector
    print(f"Creating TensorCollector for {MODEL_FAMILY}...")
    collector = TensorCollector(model_family=MODEL_FAMILY, verbose=True)

    # Create the callback function
    callback = create_callback(collector)

    # Load the model with the callback
    print(f"Loading model from {MODEL_PATH}...")
    print("(This may take a moment)")

    try:
        model = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_gpu_layers=-1,  # Use all GPU layers if available
            cb_eval=callback,
            verbose=False,
        )
    except TypeError as e:
        if "cb_eval" in str(e):
            print()
            print("Error: llama-cpp-python does not have the cb_eval patch.")
            print("Please apply the patch as described in the README.")
            sys.exit(1)
        raise

    print("Model loaded successfully!")
    print()

    # =========================================================================
    # Run Inference with Collection
    # =========================================================================

    print("Starting tensor capture...")
    print("-" * 60)

    for i, prompt in enumerate(PROMPTS):
        print(f"\nTrial {i + 1}: {prompt[:50]}...")

        # Start a new trial
        collector.start_trial(prompt)

        # Run inference
        response = model(
            prompt,
            max_tokens=50,
            temperature=0.7,
            stop=["\n\n"],
        )

        # Extract the generated text
        generated_text = response["choices"][0]["text"].strip()

        # End the trial
        trial = collector.end_trial(generated_text)

        # Print results
        print(f"  Response: {generated_text[:60]}...")
        print(f"  Unique tensors: {len(trial.tensor_counts)}")
        print(f"  Total observations: {sum(trial.tensor_counts.values())}")
        print(f"  Inference time: {trial.inference_time_ms:.1f}ms")

    print()
    print("-" * 60)

    # =========================================================================
    # Save Results
    # =========================================================================

    # Get statistics
    stats = collector.get_statistics()
    print("\nCollection Summary:")
    print(f"  Total trials: {stats['total_trials']}")
    print(f"  Total callbacks: {stats['total_callbacks']}")
    print(f"  Unique tensors: {stats['unique_tensors']}")

    # Save to JSON
    output_path = collector.save_to_json(OUTPUT_FILE)
    print(f"\nResults saved to: {output_path}")

    print()
    print("=" * 60)
    print("Capture complete!")
    print()
    print("Next steps:")
    print("1. Start the API server: python backend/server.py")
    print(f"2. Load this file via the API: POST /api/load")
    print(f"3. Analyze with: POST /api/analyze")
    print("=" * 60)


if __name__ == "__main__":
    main()
