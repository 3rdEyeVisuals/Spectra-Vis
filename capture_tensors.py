"""
Spectra Vis - Interactive Tensor Capture Tool

Drag & drop simplicity: Just run this script and follow the prompts.
No editing required!

Copyright (c) 2025 3rdEyeVisuals
Licensed under the Spectra Vis License
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from spectra_vis import TensorCollector, create_callback

# Try to import llama_cpp
try:
    from llama_cpp import Llama
except ImportError:
    print("\n[ERROR] llama-cpp-python is not installed or not patched.")
    print("You need the cb_eval patched version. See README for instructions.")
    input("\nPress Enter to exit...")
    sys.exit(1)


def get_model_path():
    """Get model path from user."""
    print("\n" + "=" * 60)
    print("  SPECTRA VIS - Tensor Capture Tool")
    print("=" * 60)

    print("\nEnter the path to your GGUF model file:")
    print("(You can drag & drop the file here)\n")

    while True:
        model_path = input("> ").strip().strip('"').strip("'")

        if not model_path:
            print("Please enter a path.")
            continue

        path = Path(model_path)
        if not path.exists():
            print(f"File not found: {model_path}")
            print("Please try again.")
            continue

        if not path.suffix.lower() == '.gguf':
            print("Warning: File doesn't have .gguf extension. Continue anyway? (y/n)")
            if input("> ").strip().lower() != 'y':
                continue

        return str(path)


def get_model_family():
    """Get model family from user."""
    families = {
        '1': ('llama', 'Meta Llama (Llama 2, Llama 3, etc.)'),
        '2': ('granite', 'IBM Granite'),
        '3': ('qwen', 'Alibaba Qwen'),
        '4': ('phi', 'Microsoft Phi'),
        '5': ('mistral', 'Mistral / Mixtral'),
    }

    print("\nSelect your model family:\n")
    for key, (name, desc) in families.items():
        print(f"  {key}. {desc}")

    print()
    while True:
        choice = input("> ").strip()
        if choice in families:
            return families[choice][0]
        print("Please enter 1-5")


def get_prompts():
    """Get prompts from user."""
    print("\n" + "-" * 60)
    print("TRIAL PROMPTS")
    print("-" * 60)
    print("\nEach prompt = one trial run. Add multiple prompts to compare")
    print("tensor activations across different inference runs.")
    print("\nExample use cases:")
    print("  - Compare simple vs complex questions")
    print("  - Compare different languages")
    print("  - Compare code vs natural language")
    print("  - Compare short vs long context")
    print("\nEnter your prompts one at a time.")
    print("Press Enter on empty line when done, or type 'default' for samples.\n")

    prompts = []
    while True:
        line = input(f"Prompt {len(prompts) + 1}: ").strip()

        if line.lower() == 'default':
            return [
                "What is the capital of France?",
                "Explain how neural networks learn in simple terms.",
                "Write a short poem about artificial intelligence.",
            ]

        if not line:
            if prompts:
                break
            print("Enter at least one prompt, or type 'default'")
            continue

        prompts.append(line)

    return prompts


def main():
    """Run interactive tensor capture."""

    # Get configuration from user
    model_path = get_model_path()
    model_family = get_model_family()
    prompts = get_prompts()

    # Generate output filename
    model_name = Path(model_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/capture_{model_name}_{timestamp}.json"

    print("\n" + "-" * 60)
    print("Configuration:")
    print(f"  Model: {Path(model_path).name}")
    print(f"  Family: {model_family}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Output: {output_file}")
    print("-" * 60)

    input("\nPress Enter to start capture...")

    # Create collector
    print("\nInitializing TensorCollector...")
    collector = TensorCollector(model_family=model_family, verbose=True)
    callback = create_callback(collector)

    # Load model
    print(f"\nLoading model (this may take a moment)...")
    try:
        model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=-1,
            cb_eval=callback,
            cb_eval_user_data=None,
            verbose=False,
        )
    except TypeError as e:
        if "cb_eval" in str(e):
            print("\n[ERROR] Your llama-cpp-python doesn't have the cb_eval patch!")
            print("See README for patching instructions.")
            input("\nPress Enter to exit...")
            sys.exit(1)
        raise
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)

    print("Model loaded!\n")

    # Run trials
    print("=" * 60)
    print("Running inference trials...")
    print("=" * 60)

    for i, prompt in enumerate(prompts):
        print(f"\n[Trial {i + 1}/{len(prompts)}]")
        print(f"Prompt: {prompt[:70]}{'...' if len(prompt) > 70 else ''}")

        collector.start_trial(prompt)

        response = model(
            prompt,
            max_tokens=100,
            temperature=0.7,
            stop=["\n\n"],
        )

        generated = response["choices"][0]["text"].strip()
        trial = collector.end_trial(generated)

        print(f"Response: {generated[:70]}{'...' if len(generated) > 70 else ''}")
        print(f"Tensors: {len(trial.tensor_counts)} unique | Time: {trial.inference_time_ms:.0f}ms")

    # Save results
    print("\n" + "=" * 60)

    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)

    output_path = collector.save_to_json(output_file)

    stats = collector.get_statistics()
    print("\nCapture Complete!")
    print(f"  Total trials: {stats['total_trials']}")
    print(f"  Unique tensors: {stats['unique_tensors']}")
    print(f"  Total callbacks: {stats['total_callbacks']:,}")
    print(f"\nSaved to: {output_path}")

    print("\n" + "=" * 60)
    print("Next: Load this file in the Spectra Vis web interface!")
    print("=" * 60)

    input("\nPress Enter to exit...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(0)
